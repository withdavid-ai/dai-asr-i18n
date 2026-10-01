"""Modal fan-out for the four speaker-diarization benchmark systems."""

import hashlib
import json
import os
import re
import tomllib
from pathlib import Path

import modal

APP_NAME = "dai-asr-i18n-diarization"
SECRET_NAME = os.environ.get("DAI_ASR_I18N_MODAL_SECRET", "dai-asr-i18n-inference")
PYANNOTE_HF_SECRET_NAME = os.environ.get("DAI_ASR_I18N_PYANNOTE_HF_MODAL_SECRET", "dai-asr-i18n-pyannote-hf")
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
NEMO_SPEECH_COMMIT = "00278b0bd95bb2b8174b88012aa21b003c59d2e9"
DIARIZEN_COMMIT = "844f5555b0a98acd0931511fc641a8c5b8ba92c7"
SORTFORMER_MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
SORTFORMER_MODEL_REVISION = "cd03eee90fbec18297ac31b8c21546e596b7f71c"
SORTFORMER_MODEL_FILENAME = "diar_streaming_sortformer_4spk-v2.1.nemo"
NEMOTRON_MODEL_ID = "nvidia/Nemotron-3-Diarization"
NEMOTRON_MODEL_REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
NEMOTRON_MODEL_FILENAME = "Nemotron-3-Diarization.nemo"
PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"
PYANNOTE_MODEL_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
DIARIZEN_MODEL_ID = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
DIARIZEN_MODEL_REVISION = "027b3e221b9d81dce2be794084f1dbd3ba2da403"
DIARIZEN_EMBEDDING_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
DIARIZEN_EMBEDDING_REVISION = "837717ddb9ff5507820346191109dc79c958d614"
DIARIZEN_CACHE = "/opt/diarizen-models"
BACKEND_FILTER = os.environ.get("DAI_ASR_I18N_DIARIZATION_BACKEND", "all")
if BACKEND_FILTER not in {"all", "nemo-sortformer", "pyannote", "diarizen"}:
    raise ValueError(f"unsupported DAI_ASR_I18N_DIARIZATION_BACKEND={BACKEND_FILTER!r}")

app = modal.App(APP_NAME)
dataset_secret = modal.Secret.from_name(SECRET_NAME)
pyannote_hf_secret = modal.Secret.from_name(PYANNOTE_HF_SECRET_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


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


base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_sync(extras=["audio", "hf", "speaker"], frozen=True)
    .env(SOURCE_ENV)
    .add_local_dir("src/dai_asr_i18n", remote_path="/root/dai_asr_i18n")
)


def _load_sortformer_model():
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import SortformerEncLabelModel

    checkpoint = hf_hub_download(
        repo_id=SORTFORMER_MODEL_ID,
        filename=SORTFORMER_MODEL_FILENAME,
        revision=SORTFORMER_MODEL_REVISION,
    )
    return SortformerEncLabelModel.restore_from(checkpoint)


def _load_nemotron_model():
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import SortformerEncLabelModel

    checkpoint = hf_hub_download(
        repo_id=NEMOTRON_MODEL_ID,
        filename=NEMOTRON_MODEL_FILENAME,
        revision=NEMOTRON_MODEL_REVISION,
    )
    return SortformerEncLabelModel.restore_from(checkpoint)


def _download_nemo_models() -> None:
    _load_sortformer_model()
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=NEMOTRON_MODEL_ID,
        filename=NEMOTRON_MODEL_FILENAME,
        revision=NEMOTRON_MODEL_REVISION,
    )


nemo_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("ffmpeg", "libsndfile1", "sox", "build-essential", "g++", "cmake", "pkg-config", "git")
    .env({"CC": "gcc", "CXX": "g++", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **SOURCE_ENV})
    .pip_install("Cython", "packaging")
    .pip_install(
        f"nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git@{NEMO_SPEECH_COMMIT}",
        "soundfile==0.13.1",
    )
    .pip_install(
        "indic-nlp-library==0.92",
        "jiwer==4.0.0",
        "librosa==0.11.0",
        "more-itertools==11.1.0",
        "opencc==1.4.2",
        "rapidfuzz==3.14.6",
        "regex==2026.6.28",
        "soxr==1.1.0",
        "whisper-normalizer==0.1.12",
    )
    .run_function(_download_nemo_models)
    .add_local_dir("src/dai_asr_i18n", remote_path="/root/dai_asr_i18n")
)


def _hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _download_pyannote() -> None:
    from pyannote.audio import Pipeline

    if (
        Pipeline.from_pretrained(
            PYANNOTE_MODEL_ID,
            revision=PYANNOTE_MODEL_REVISION,
            token=_hf_token(),
        )
        is None
    ):
        raise RuntimeError(f"unable to download gated model {PYANNOTE_MODEL_ID}")


pyannote_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "indic-nlp-library==0.92",
        "jiwer==4.0.0",
        "librosa==0.11.0",
        "more-itertools==11.1.0",
        "opencc==1.4.2",
        "pyannote.audio==4.0.7",
        "rapidfuzz==3.14.6",
        "regex==2026.6.28",
        "soundfile==0.14.0",
        "soxr==1.1.0",
        "whisper-normalizer==0.1.12",
    )
    .run_function(_download_pyannote, secrets=[pyannote_hf_secret])
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **SOURCE_ENV})
    .add_local_dir("src/dai_asr_i18n", remote_path="/root/dai_asr_i18n")
)


def _load_diarizen(*, local_files_only: bool):
    from diarizen.pipelines.inference import DiariZenPipeline
    from huggingface_hub import hf_hub_download, snapshot_download

    model_root = snapshot_download(
        repo_id=DIARIZEN_MODEL_ID,
        revision=DIARIZEN_MODEL_REVISION,
        cache_dir=DIARIZEN_CACHE,
        local_files_only=local_files_only,
    )
    embedding = hf_hub_download(
        repo_id=DIARIZEN_EMBEDDING_MODEL_ID,
        filename="pytorch_model.bin",
        revision=DIARIZEN_EMBEDDING_REVISION,
        cache_dir=DIARIZEN_CACHE,
        local_files_only=local_files_only,
    )
    return DiariZenPipeline(diarizen_hub=Path(model_root), embedding_model=embedding)


def _download_diarizen() -> None:
    _load_diarizen(local_files_only=False)


diarizen_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "clang", "ffmpeg", "git", "libsndfile1")
    .run_commands(
        f"git clone --recursive https://github.com/BUTSpeechFIT/DiariZen.git /opt/DiariZen && "
        f"cd /opt/DiariZen && git checkout {DIARIZEN_COMMIT} && git submodule update --init --recursive"
    )
    .pip_install(
        "torch==2.1.1",
        "torchaudio==2.1.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        "cd /opt/DiariZen && pip install -r requirements.txt && pip install . && "
        "cd pyannote-audio && pip install . -c ../constraints.txt"
    )
    .pip_install(
        "indic-nlp-library==0.92",
        "jiwer==4.0.0",
        "more-itertools==11.1.0",
        "opencc==1.4.2",
        "rapidfuzz==3.14.6",
        "regex==2026.6.28",
        "whisper-normalizer==0.1.12",
    )
    .run_function(_download_diarizen)
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **SOURCE_ENV})
    .add_local_dir("src/dai_asr_i18n", remote_path="/root/dai_asr_i18n")
)


@app.function(
    image=base_image,
    secrets=[dataset_secret],
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
    from dai_asr_i18n.datasets import pull_huggingface_dataset

    repo_slug = re.sub(r"[^A-Za-z0-9._-]+", "--", repo_id).strip("-.")
    revision_root = Path(MOUNT) / "datasets" / repo_slug / revision / DATASET_CACHE_VERSION
    destination = revision_root / _filter_key(languages, sample_ids)
    complete_destination = revision_root / _filter_key((), ())
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
    )
    volume.commit()
    return str(destination)


def _segments(annotation) -> tuple:
    from dai_asr_i18n.inference import TranscriptSegment

    if not hasattr(annotation, "itertracks"):
        raise RuntimeError(f"diarizer returned unusable output {type(annotation).__name__}")
    return tuple(
        TranscriptSegment(text="", start_s=float(turn.start), end_s=float(turn.end), speaker=str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    )


class _Backend:
    def preflight(self) -> dict[str, object]:
        return {"ok": True, "identity": self.identity}


class _NemoBackend(_Backend):
    def __init__(self, model, config, applied_profile):
        self.model = model
        self.config = config
        self.applied_profile = applied_profile

    @property
    def identity(self) -> dict[str, object]:
        return {
            "backend": "nemo-sortformer",
            "backend_version": f"NVIDIA-NeMo/Speech@{NEMO_SPEECH_COMMIT}",
            "model_id": self.config.model.model_id,
            "model_revision": self.config.model.revision,
            "device": "cuda",
            "compute_type": "float32",
            "transcribes": False,
            "diarizes": True,
            "options": self.config.model.options,
        }

    def transcribe(self, audio_path: Path, *, language: str | None):
        import soundfile as sf
        import torch

        from dai_asr_i18n.inference import Transcript, TranscriptSegment

        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.model.device.type, dtype=self.model.dtype, enabled=False),
        ):
            predicted = self.model.diarize(audio=[str(audio_path)], batch_size=1)
        raw = predicted[0] if predicted else []
        segments = tuple(
            TranscriptSegment(text="", start_s=s[0], end_s=s[1], speaker=s[2])
            for value in raw
            if (s := _parse_nemo_segment(value)) is not None
        )
        return Transcript(
            text="",
            language=language,
            duration_s=float(sf.info(audio_path).duration),
            segments=segments,
            metadata={"streaming_config": self.applied_profile},
        )


def _parse_nemo_segment(value) -> tuple[float, float, str] | None:
    if isinstance(value, str):
        parts = value.split()
        if len(parts) < 3:
            return None
        try:
            return float(parts[0]), float(parts[1]), str(parts[2])
        except ValueError:
            return None
    if isinstance(value, dict):
        start = value.get("start", value.get("start_time"))
        end = value.get("end", value.get("end_time"))
        speaker = value.get("speaker", value.get("speaker_id", value.get("label")))
    else:
        start = getattr(value, "start", getattr(value, "start_time", None))
        end = getattr(value, "end", getattr(value, "end_time", None))
        speaker = getattr(value, "speaker", getattr(value, "speaker_id", getattr(value, "label", None)))
    if start is None or end is None or speaker is None:
        return None
    return float(start), float(end), str(speaker)


class _AnnotationBackend(_Backend):
    def __init__(self, pipeline, config, backend_version: str):
        self.pipeline = pipeline
        self.config = config
        self.backend_version = backend_version

    @property
    def identity(self) -> dict[str, object]:
        return {
            "backend": self.config.model.backend,
            "backend_version": self.backend_version,
            "model_id": self.config.model.model_id,
            "model_revision": self.config.model.revision,
            "device": "cuda",
            "compute_type": "float32",
            "transcribes": False,
            "diarizes": True,
            "options": self.config.model.options,
        }

    def transcribe(self, audio_path: Path, *, language: str | None):
        import soundfile as sf

        from dai_asr_i18n.inference import Transcript

        output = self.pipeline(str(audio_path))
        annotation = getattr(output, "speaker_diarization", output)
        return Transcript(
            text="",
            language=language,
            duration_s=float(sf.info(audio_path).duration),
            segments=_segments(annotation),
        )


def _run_shard(backend, dataset_root, config_name, run_name, languages, sample_ids, shard_index, shard_count):
    from dai_asr_i18n.inference import load_run_config, run_inference

    output = Path(MOUNT) / "runs" / _safe_name(run_name, "run_name") / "shards" / f"{shard_index:05d}"
    report = run_inference(
        dataset_root,
        load_run_config(config_name),
        output,
        backend=backend,
        languages=languages,
        sample_ids=sample_ids,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    volume.commit()
    return report.to_dict()


class NemoRunner:
    config_name: str = modal.parameter()

    @modal.enter()
    def load(self):
        from dai_asr_i18n.inference import load_run_config

        self.config = load_run_config(self.config_name)
        if self.config.model.model_id == SORTFORMER_MODEL_ID:
            self.model = _load_sortformer_model().eval().to("cuda")
        elif self.config.model.model_id == NEMOTRON_MODEL_ID:
            if self.config.model.revision != NEMOTRON_MODEL_REVISION:
                raise ValueError("the Nemotron run config must match the image-pinned model revision")
            self.model = _load_nemotron_model().eval().to("cuda")
        else:  # pragma: no cover - guarded by the packaged run configs
            raise ValueError(f"unsupported NeMo diarization model {self.config.model.model_id!r}")
        profile = dict(self.config.model.options["streaming"])
        modules = self.model.sortformer_modules
        for key, value in profile.items():
            if not hasattr(modules, key):
                raise RuntimeError(f"sortformer_modules has no {key!r}")
            setattr(modules, key, value)
        self.model._check_streaming_parameters()
        self.backend = _NemoBackend(self.model, self.config, profile)

    @modal.method()
    def run_shard(self, dataset_root, run_name, languages, sample_ids, shard_index, shard_count):
        return _run_shard(
            self.backend,
            dataset_root,
            self.config_name,
            run_name,
            languages,
            sample_ids,
            shard_index,
            shard_count,
        )


if BACKEND_FILTER in {"all", "nemo-sortformer"}:
    NemoRunner = app.cls(
        image=nemo_image,
        gpu="A100",
        volumes={MOUNT: volume},
        timeout=24 * 60 * 60,
        max_containers=64,
    )(NemoRunner)


class PyannoteRunner:
    @modal.enter()
    def load(self):
        import torch
        from pyannote.audio import Pipeline

        from dai_asr_i18n.inference import load_run_config

        self.config = load_run_config("pyannote-community-1")
        self.pipeline = Pipeline.from_pretrained(
            self.config.model.model_id,
            revision=self.config.model.revision,
            token=_hf_token(),
        )
        if self.pipeline is None:
            raise RuntimeError(f"unable to load gated model {self.config.model.model_id}")
        self.pipeline.to(torch.device("cuda"))
        self.backend = _AnnotationBackend(self.pipeline, self.config, "pyannote.audio==4.0.7")

    @modal.method()
    def run_shard(self, dataset_root, run_name, languages, sample_ids, shard_index, shard_count):
        return _run_shard(
            self.backend,
            dataset_root,
            "pyannote-community-1",
            run_name,
            languages,
            sample_ids,
            shard_index,
            shard_count,
        )


if BACKEND_FILTER in {"all", "pyannote"}:
    PyannoteRunner = app.cls(
        image=pyannote_image,
        gpu="A10G",
        secrets=[pyannote_hf_secret],
        volumes={MOUNT: volume},
        timeout=24 * 60 * 60,
        max_containers=64,
    )(PyannoteRunner)


class DiarizenRunner:
    @modal.enter()
    def load(self):
        from dai_asr_i18n.inference import load_run_config

        self.config = load_run_config("diarizen-wavlm-large-s80-md-v2")
        self.pipeline = _load_diarizen(local_files_only=True)
        self.backend = _AnnotationBackend(self.pipeline, self.config, f"DiariZen@{DIARIZEN_COMMIT}")

    @modal.method()
    def run_shard(self, dataset_root, run_name, languages, sample_ids, shard_index, shard_count):
        return _run_shard(
            self.backend,
            dataset_root,
            "diarizen-wavlm-large-s80-md-v2",
            run_name,
            languages,
            sample_ids,
            shard_index,
            shard_count,
        )


if BACKEND_FILTER in {"all", "diarizen"}:
    DiarizenRunner = app.cls(
        image=diarizen_image,
        gpu="A10G",
        volumes={MOUNT: volume},
        timeout=24 * 60 * 60,
        max_containers=64,
    )(DiarizenRunner)


@app.function(image=base_image, volumes={MOUNT: volume}, timeout=60 * 60)
def finalize_run(run_name: str, shard_count: int) -> str:
    from dai_asr_i18n.inference import merge_run_shards

    root = Path(MOUNT) / "runs" / _safe_name(run_name, "run_name")
    destination = root / "combined"
    destination = merge_run_shards(
        [root / "shards" / f"{index:05d}" for index in range(shard_count)],
        destination,
        replace=True,
    )
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
    shards: int = 32,
) -> None:
    from dai_asr_i18n.inference import load_run_config

    if shards < 1:
        raise ValueError("shards must be positive")
    selected_languages = tuple(value.strip() for value in languages.split(",") if value.strip())
    selected_samples = tuple(value.strip() for value in sample_ids.split(",") if value.strip())
    selected_config = load_run_config(config)
    if not selected_config.source.startswith("builtin:"):
        raise ValueError("Modal runs require a packaged built-in config name")
    if BACKEND_FILTER != "all" and selected_config.model.backend != BACKEND_FILTER:
        raise ValueError(
            f"config backend {selected_config.model.backend!r} does not match "
            f"DAI_ASR_I18N_DIARIZATION_BACKEND={BACKEND_FILTER!r}"
        )
    if selected_config.model.backend not in {"nemo-sortformer", "pyannote", "diarizen"}:
        raise ValueError(f"unsupported diarization backend {selected_config.model.backend!r}")
    if not run_name:
        run_name = f"{selected_config.name}-{revision[:12]}"
    run_name = _safe_name(run_name, "run_name")
    dataset_root = prepare_dataset.remote(repo_id, revision, selected_languages, selected_samples)
    if selected_config.model.backend == "nemo-sortformer":
        runner = NemoRunner(config_name=config)
    elif selected_config.model.backend == "pyannote":
        runner = PyannoteRunner()
    elif selected_config.model.backend == "diarizen":
        runner = DiarizenRunner()
    calls = [
        runner.run_shard.spawn(
            dataset_root,
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
