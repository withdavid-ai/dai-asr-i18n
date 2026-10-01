"""Optional Modal dispatcher for the canonical DAI-ASR-I18N inference runner."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from pathlib import Path

import modal

APP_NAME = "dai-asr-i18n-inference"
SECRET_NAMES = tuple(
    dict.fromkeys(
        value.strip()
        for value in os.environ.get("DAI_ASR_I18N_MODAL_SECRETS", "dai-asr-i18n-inference").split(",")
        if value.strip()
    )
)
VOLUME_NAME = os.environ.get("DAI_ASR_I18N_MODAL_VOLUME", "dai-asr-i18n-runs")
MOUNT = "/dai-asr-i18n"
DATASET_CACHE_VERSION = "hf-release-word-alignments-v1"


def _source_sha256() -> str:
    repository = Path(__file__).resolve().parents[1]
    root = repository / "src" / "dai_asr_i18n"
    digest = hashlib.sha256()
    package_files = (
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and (candidate.suffix in {".py", ".toml"} or candidate.name == "py.typed")
    )
    for path in sorted([Path(__file__).resolve(), *package_files]):
        digest.update(path.relative_to(repository).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return str(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"])


SOURCE_ENV = {
    "DAI_ASR_I18N_PACKAGE_VERSION": _package_version(),
    "DAI_ASR_I18N_SOURCE_SHA256": _source_sha256(),
}
HOSTED_BACKENDS = frozenset(
    {
        "deepgram",
        "elevenlabs",
        "gemini-transcribe",
        "mai-transcribe",
        "meta-muse",
        "mistral-voxtral",
        "openai-transcription",
        "xai-grok",
    }
)

app = modal.App(APP_NAME)
secrets = [modal.Secret.from_name(name) for name in SECRET_NAMES]
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

base_dependencies = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_sync(extras=["audio", "hf", "hosted", "speaker"], frozen=True)
    .env(
        {
            "DAI_ASR_I18N_MODAL_SECRETS": ",".join(SECRET_NAMES),
            "DAI_ASR_I18N_MODAL_QWEN": "0",
            **SOURCE_ENV,
        }
    )
)
base_image = base_dependencies.add_local_dir("src/dai_asr_i18n", remote_path="/root/dai_asr_i18n")
whisper_image = base_dependencies.uv_pip_install(
    "av==18.1.0",
    "ctranslate2==4.8.2",
    "faster-whisper==1.2.1",
    "tokenizers==0.23.2",
).add_local_dir("src/dai_asr_i18n", remote_path="/root/dai_asr_i18n")

QWEN_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
QWEN_MODEL_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
QWEN_MODEL_PATH = "/models/qwen3-asr-1.7b"


def _download_qwen_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(QWEN_MODEL_ID, revision=QWEN_MODEL_REVISION, local_dir=QWEN_MODEL_PATH)


ENABLE_QWEN = os.environ.get("DAI_ASR_I18N_MODAL_QWEN") == "1"
if ENABLE_QWEN:
    qwen_image = (
        modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
        .apt_install("build-essential", "ffmpeg", "libsndfile1", "sox")
        .pip_install(
            "torch",
            "qwen-asr==0.0.6",
            "huggingface-hub==0.36.0",
            "indic-nlp-library==0.92",
            "jiwer==4.0.0",
            "librosa==0.11.0",
            "more-itertools==11.1.0",
            "numpy==2.4.6",
            "opencc==1.4.2",
            "rapidfuzz==3.14.6",
            "regex==2026.6.28",
            "soundfile==0.14.0",
            "soxr==1.1.0",
            "whisper-normalizer==0.1.12",
        )
        .run_function(_download_qwen_model)
        .env(
            {
                "DAI_ASR_I18N_MODAL_SECRETS": ",".join(SECRET_NAMES),
                "DAI_ASR_I18N_MODAL_QWEN": "1",
                "DAI_ASR_I18N_QWEN_MODEL_PATH": QWEN_MODEL_PATH,
                **SOURCE_ENV,
            }
        )
        .add_local_dir("src/dai_asr_i18n", remote_path="/root/dai_asr_i18n")
    )


def _safe_name(value: str, field: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise ValueError(f"{field} must be a safe portable name")
    return value


def _filter_key(languages: tuple[str, ...], sample_ids: tuple[str, ...]) -> str:
    value = json.dumps(
        {"languages": sorted(languages), "sample_ids": sorted(sample_ids)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _dataset_root(repo_id: str, revision: str, languages: tuple[str, ...], sample_ids: tuple[str, ...]) -> Path:
    repo_slug = re.sub(r"[^A-Za-z0-9._-]+", "--", repo_id).strip("-.")
    return Path(MOUNT) / "datasets" / repo_slug / revision / DATASET_CACHE_VERSION / _filter_key(languages, sample_ids)


@app.function(
    image=base_image,
    secrets=secrets,
    volumes={MOUNT: volume},
    timeout=24 * 60 * 60,
    max_containers=1,
)
def prepare_dataset(
    repo_id: str,
    revision: str,
    languages: tuple[str, ...],
    sample_ids: tuple[str, ...],
) -> str:
    """Materialize one immutable HF selection once before fan-out."""

    from dai_asr_i18n.datasets import pull_huggingface_dataset

    destination = _dataset_root(repo_id, revision, languages, sample_ids)
    complete_destination = _dataset_root(repo_id, revision, (), ())
    for existing in (destination, complete_destination):
        if (existing / "selection.jsonl").is_file() and (existing / "dataset-receipt.json").is_file():
            return str(existing)
    pull_huggingface_dataset(
        repo_id=repo_id,
        revision=revision,
        output=destination,
        languages=languages,
        clip_ids=sample_ids,
        authenticated=True,
        dataset_format="release",
        workers=4,
    )
    volume.commit()
    return str(destination)


@app.function(image=base_image, volumes={MOUNT: volume}, timeout=24 * 60 * 60, max_containers=3)
def wait_for_complete_dataset(repo_id: str, revision: str) -> str:
    """Wait for a separately launched full-dataset preparation to commit."""

    import time

    destination = _dataset_root(repo_id, revision, (), ())
    while True:
        volume.reload()
        if (destination / "selection.jsonl").is_file() and (destination / "dataset-receipt.json").is_file():
            return str(destination)
        time.sleep(30)


def _run_shard(
    dataset_root: str,
    config_name: str,
    run_name: str,
    languages: tuple[str, ...],
    sample_ids: tuple[str, ...],
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    from dai_asr_i18n.inference import load_run_config, run_inference

    output = Path(MOUNT) / "runs" / _safe_name(run_name, "run_name") / "shards" / f"{shard_index:05d}"
    report = run_inference(
        dataset_root,
        load_run_config(config_name),
        output,
        languages=languages,
        sample_ids=sample_ids,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    volume.commit()
    return report.to_dict()


@app.function(
    image=base_image,
    secrets=secrets,
    volumes={MOUNT: volume},
    timeout=24 * 60 * 60,
    max_containers=100,
)
def run_hosted_shard(
    dataset_root: str,
    config_name: str,
    run_name: str,
    languages: tuple[str, ...],
    sample_ids: tuple[str, ...],
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    return _run_shard(dataset_root, config_name, run_name, languages, sample_ids, shard_index, shard_count)


@app.function(
    image=whisper_image,
    gpu="A10G",
    secrets=secrets,
    volumes={MOUNT: volume},
    timeout=24 * 60 * 60,
    max_containers=32,
)
def run_gpu_shard(
    dataset_root: str,
    config_name: str,
    run_name: str,
    languages: tuple[str, ...],
    sample_ids: tuple[str, ...],
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    return _run_shard(dataset_root, config_name, run_name, languages, sample_ids, shard_index, shard_count)


if ENABLE_QWEN:

    @app.function(
        image=qwen_image,
        gpu="A10G",
        secrets=secrets,
        volumes={MOUNT: volume},
        timeout=24 * 60 * 60,
        max_containers=32,
    )
    def run_qwen_shard(
        dataset_root: str,
        config_name: str,
        run_name: str,
        languages: tuple[str, ...],
        sample_ids: tuple[str, ...],
        shard_index: int,
        shard_count: int,
    ) -> dict[str, object]:
        return _run_shard(dataset_root, config_name, run_name, languages, sample_ids, shard_index, shard_count)


@app.function(image=base_image, volumes={MOUNT: volume}, timeout=60 * 60)
def finalize_run(run_name: str, shard_count: int) -> str:
    from dai_asr_i18n.inference import merge_run_shards

    root = Path(MOUNT) / "runs" / _safe_name(run_name, "run_name")
    shards = [root / "shards" / f"{index:05d}" for index in range(shard_count)]
    destination = root / "combined"
    destination = merge_run_shards(shards, destination, replace=True)
    volume.commit()
    return str(destination)


@app.local_entrypoint()
def main(
    repo_id: str,
    revision: str,
    config: str,
    run_name: str = "",
    languages: str = "",
    sample_ids: str = "",
    shards: int = 8,
    prepare_all: bool = False,
    prepare_only: bool = False,
    wait_for_complete: bool = False,
) -> None:
    """Prepare, fan out, and consolidate a pinned public benchmark run."""

    from dai_asr_i18n.inference import load_run_config

    if shards < 1:
        raise ValueError("shards must be positive")
    selected_languages = tuple(value.strip() for value in languages.split(",") if value.strip())
    selected_samples = tuple(value.strip() for value in sample_ids.split(",") if value.strip())
    selected_config = load_run_config(config)
    if not selected_config.source.startswith("builtin:"):
        raise ValueError("Modal runs require a packaged built-in config name")
    if not run_name:
        run_name = f"{selected_config.name}-{revision[:12]}"
    run_name = _safe_name(run_name, "run_name")
    if wait_for_complete:
        dataset_root = wait_for_complete_dataset.remote(repo_id, revision)
    else:
        dataset_root = prepare_dataset.remote(
            repo_id,
            revision,
            () if prepare_all else selected_languages,
            () if prepare_all else selected_samples,
        )
    if prepare_only:
        print(json.dumps({"dataset_root": dataset_root, "prepared_all": prepare_all}, indent=2))
        return
    backend = selected_config.model.backend
    if backend == "qwen3-asr":
        if not ENABLE_QWEN:
            raise ValueError("Qwen runs require DAI_ASR_I18N_MODAL_QWEN=1")
        function = run_qwen_shard
    elif backend == "faster-whisper":
        function = run_gpu_shard
    elif backend in HOSTED_BACKENDS:
        function = run_hosted_shard
    else:
        raise ValueError(
            f"unsupported general Modal backend {backend!r}; "
            "speaker-diarization configs use deploy/modal_diarization.py"
        )
    calls = [
        function.spawn(
            dataset_root,
            config,
            run_name,
            selected_languages,
            selected_samples,
            shard_index,
            shards,
        )
        for shard_index in range(shards)
    ]
    reports = modal.FunctionCall.gather(*calls)
    combined = finalize_run.remote(run_name, shards)
    print(json.dumps({"run_name": run_name, "combined_output": combined, "shards": reports}, indent=2))
