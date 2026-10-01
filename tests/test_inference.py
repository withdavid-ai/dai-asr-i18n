from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from dai_asr_i18n.audio import build_mono_mix
from dai_asr_i18n.datasets.models import RECEIPT_FILENAME, SELECTION_FILENAME
from dai_asr_i18n.inference import InferenceBackend, Transcript, TranscriptSegment, load_run_config
from dai_asr_i18n.inference.backends import FasterWhisperBackend
from dai_asr_i18n.inference.runner import _safe_error, preflight_run, run_inference

REVISION = "2" * 40


def _wav(path: Path, value: int = 0, frames: int = 160) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(value.to_bytes(2, "little", signed=True) * frames)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(root: Path, *, include_mono_hash: bool = False) -> Path:
    first = root / "audio/clip-1/ch1.wav"
    second = root / "audio/clip-1/ch2.wav"
    row = {
        "schema_version": 2,
        "clip_id": "clip-1",
        "language": "en-US",
        "duration_s": 0.01,
        "audio": {
            "ch1": {"path": "audio/clip-1/ch1.wav", "sha256": _wav(first, 100), "duration_s": 0.01},
            "ch2": {"path": "audio/clip-1/ch2.wav", "sha256": _wav(second, 200), "duration_s": 0.01},
        },
        "word_alignments": {
            "ch1": [
                {
                    "start_s": 0.0,
                    "end_s": 0.005,
                    "timing_source": "forced",
                    "words": [{"text": "hello", "start_s": 0.001, "end_s": 0.004}],
                }
            ],
            "ch2": [
                {
                    "start_s": 0.005,
                    "end_s": 0.01,
                    "timing_source": "forced",
                    "words": [{"text": "world", "start_s": 0.006, "end_s": 0.009}],
                }
            ],
        },
        "metadata": {"fixture": True},
    }
    if include_mono_hash:
        row["mono_mix_sha256"] = build_mono_mix(first, second).sha256
    (root / SELECTION_FILENAME).write_text(json.dumps(row) + "\n", encoding="utf-8")
    (root / RECEIPT_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_id": "example/public-asr",
                "revision": REVISION,
                "resolved_revision": REVISION,
                "source_manifest": "manifest/public-v1.jsonl",
            }
        ),
        encoding="utf-8",
    )
    return root


class FakeBackend(InferenceBackend):
    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []

    @property
    def identity(self) -> dict[str, object]:
        return {"backend": "fake", "model_id": "fake/model", "model_revision": REVISION}

    def preflight(self) -> dict[str, object]:
        return {"ok": True, "identity": self.identity}

    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        self.calls.append((audio_path.name, language))
        return Transcript(text="hello", language="en", duration_s=0.01, segments=(TranscriptSegment("hello", 0, 0.01),))


class FlakyBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.failures_remaining = 1

    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("temporary test failure")
        return super().transcribe(audio_path, language=language)


class AlwaysFailBackend(FakeBackend):
    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        raise RuntimeError("token=hf_1234567890SECRET")


def test_builtin_configs_are_pinned_and_distinct():
    portable = load_run_config("whisper-large-v3-portable")
    cuda = load_run_config("whisper-large-v3-cuda")
    smoke = load_run_config("whisper-tiny-cpu-smoke")
    assert len(portable.model.revision) == 40
    assert portable.model.authenticated is False
    assert portable.model.device == "auto"
    assert cuda.model.device == "cuda"
    assert smoke.model.device == "cpu"
    assert smoke.model.compute_type == "int8"
    assert portable.digest() != cuda.digest()


def test_diarization_run_configs_preserve_backend_and_access_contracts():
    sortformer = load_run_config("sortformer-4spk-v2.1")
    pyannote = load_run_config("pyannote-community-1")
    diarizen = load_run_config("diarizen-wavlm-large-s80-md-v2")
    nemotron = load_run_config("nemotron-3-diarization")

    assert sortformer.channels == ("mono",)
    assert len(sortformer.model.revision) == 40
    assert sortformer.model.backend == nemotron.model.backend == "nemo-sortformer"
    assert pyannote.model.backend == "pyannote"
    assert pyannote.model.authenticated is True
    assert pyannote.model.environment == ("HF_TOKEN",)
    assert diarizen.model.backend == "diarizen"
    assert diarizen.model.options["weights_license"] == "CC-BY-NC-4.0"
    assert nemotron.model.revision == "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
    assert nemotron.model.authenticated is False
    assert nemotron.model.environment == ()
    assert load_run_config("mai-transcribe-2").model.environment == (
        "AZURE_SPEECH_KEY",
        "AZURE_SPEECH_ENDPOINT",
    )


def test_custom_config_source_does_not_leak_its_local_path(tmp_path: Path):
    config_path = tmp_path / "custom-run.toml"
    config_path.write_text(
        Path("src/dai_asr_i18n/run_configs/whisper-tiny-cpu-smoke.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.source == "file:custom-run.toml"
    assert str(tmp_path) not in config.source


def test_run_configs_reject_embedded_credentials(tmp_path: Path):
    config_path = tmp_path / "unsafe.toml"
    config_path.write_text(
        Path("src/dai_asr_i18n/run_configs/deepgram-nova-3.toml")
        .read_text(encoding="utf-8")
        .replace("smart_format = true", 'api_key = "secret"'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain credentials"):
        load_run_config(config_path)


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf")])
def test_faster_whisper_rejects_nonfinite_temperatures(temperature):
    config = load_run_config("whisper-large-v3-portable").model
    invalid = replace(config, options={**config.options, "temperature": temperature})

    with pytest.raises(ValueError, match="finite non-negative"):
        FasterWhisperBackend(invalid)


def test_preflight_is_read_only_and_runner_resumes(tmp_path: Path):
    dataset = _dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    config = load_run_config("whisper-large-v3-portable")
    backend = FakeBackend()

    report = preflight_run(dataset, config, backend=backend)
    assert report.ok
    assert report.tasks == 2
    assert not output.exists()

    first = run_inference(dataset, config, output, backend=backend)
    assert first.completed == 2
    assert first.skipped == 0
    assert len(backend.calls) == 2
    rows = [json.loads(line) for line in (output / "hypotheses.jsonl").read_text().splitlines()]
    assert {row["channel"] for row in rows} == {"ch1", "ch2"}
    assert {row["sample_id"] for row in rows} == {"clip-1"}
    assert all("clip_id" not in row for row in rows)
    assert rows[0]["dataset_revision"] == REVISION
    assert rows[0]["dataset_source_type"] == "huggingface"
    assert rows[0]["dataset_source_id"] == "example/public-asr"
    assert rows[0]["reference"] in {"hello", "world"}
    for row in rows:
        expected_activity = {
            "ch1": [{"speaker": "ch1", "start_s": 0.0, "end_s": 0.005}],
            "ch2": [{"speaker": "ch2", "start_s": 0.005, "end_s": 0.01}],
        }
        assert row["reference_diarization"] == expected_activity[row["channel"]]

    second = run_inference(dataset, config, output, backend=backend)
    assert second.completed == 0
    assert second.skipped == 2
    assert len(backend.calls) == 2


def test_runner_constructs_the_configured_backend_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = _dataset(tmp_path / "dataset")
    created: list[FakeBackend] = []

    def backend_for(_config):
        backend = FakeBackend()
        created.append(backend)
        return backend

    monkeypatch.setattr("dai_asr_i18n.inference.runner._backend_for", backend_for)

    report = run_inference(
        dataset,
        load_run_config("whisper-large-v3-portable"),
        tmp_path / "output",
    )

    assert report.status == "complete"
    assert len(created) == 1
    assert len(created[0].calls) == 2


def test_preflight_rejects_missing_requested_channels(tmp_path: Path):
    dataset = _dataset(tmp_path / "dataset")
    row = json.loads((dataset / SELECTION_FILENAME).read_text(encoding="utf-8"))
    del row["audio"]["ch2"]
    del row["word_alignments"]["ch2"]
    (dataset / SELECTION_FILENAME).write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = load_run_config("whisper-large-v3-portable")

    report = preflight_run(dataset, config, backend=FakeBackend())

    assert not report.ok
    assert report.tasks == 1
    assert report.missing_channel_tasks == 1
    assert report.missing_channels == ({"sample_id": "clip-1", "missing": ["ch2"]},)


def test_distributed_run_writes_a_complete_manifest_for_an_empty_shard(tmp_path: Path):
    dataset = _dataset(tmp_path / "dataset")
    config = replace(load_run_config("whisper-tiny-cpu-smoke"), channels=("ch1",))
    empty_index = next(
        index
        for index in range(8)
        if preflight_run(dataset, config, backend=FakeBackend(), shard_index=index, shard_count=8).tasks == 0
    )

    report = run_inference(
        dataset,
        config,
        tmp_path / "empty-shard",
        backend=FakeBackend(),
        shard_index=empty_index,
        shard_count=8,
    )

    assert report.status == "complete"
    assert report.total == 0
    assert report.completed == 0


def test_runner_rejects_output_inside_the_read_only_dataset(tmp_path: Path):
    dataset = _dataset(tmp_path / "dataset")
    output = dataset / "results"

    with pytest.raises(ValueError, match="outside the read-only dataset"):
        run_inference(
            dataset,
            load_run_config("whisper-large-v3-portable"),
            output,
            backend=FakeBackend(),
        )
    assert not output.exists()


def test_runner_rejects_symlinks_inside_a_resume_directory(tmp_path: Path):
    dataset = _dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    outside = tmp_path / "outside.jsonl"
    output.mkdir()
    outside.write_text("unchanged", encoding="utf-8")
    (output / "hypotheses.jsonl").symlink_to(outside)

    with pytest.raises(ValueError, match="containing symbolic link"):
        run_inference(
            dataset,
            load_run_config("whisper-large-v3-portable"),
            output,
            backend=FakeBackend(),
        )
    assert outside.read_text(encoding="utf-8") == "unchanged"


def test_failures_are_retried_and_run_identity_cannot_mix(tmp_path: Path):
    dataset = _dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    config = load_run_config("whisper-large-v3-portable")
    backend = FlakyBackend()

    first = run_inference(dataset, config, output, backend=backend)
    assert first.status == "incomplete_with_failures"
    assert first.failed == 1
    assert first.manifest["coverage"]["remaining"] == 1  # type: ignore[index]

    second = run_inference(dataset, config, output, backend=backend)
    assert second.status == "complete"
    assert second.completed == 1
    assert second.skipped == 1
    assert second.manifest["coverage"]["failure_attempts_total"] == 1  # type: ignore[index]

    with pytest.raises(ValueError, match="different dataset selection or run config"):
        run_inference(dataset, load_run_config("whisper-tiny-cpu-smoke"), output, backend=backend)


def test_resume_rejects_outputs_from_a_different_source_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = _dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    config = load_run_config("whisper-large-v3-portable")
    monkeypatch.setenv("DAI_ASR_I18N_SOURCE_SHA256", "1" * 64)
    run_inference(dataset, config, output, backend=FakeBackend())

    monkeypatch.setenv("DAI_ASR_I18N_SOURCE_SHA256", "2" * 64)
    with pytest.raises(ValueError, match="different dataset selection or run config"):
        run_inference(dataset, config, output, backend=FakeBackend())


def test_systematic_failures_abort_and_secrets_are_redacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = _dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    config = load_run_config("whisper-large-v3-portable")
    monkeypatch.setattr("dai_asr_i18n.inference.runner.SYSTEMATIC_FAILURE_THRESHOLD", 2)

    report = run_inference(dataset, config, output, backend=AlwaysFailBackend())
    assert report.status == "incomplete_systematic_failure"
    assert report.failed == 2
    failures = (output / "failures.jsonl").read_text()
    assert "hf_1234567890SECRET" not in failures
    assert "[REDACTED]" in failures
    assert "secret-value" not in _safe_error(RuntimeError("api_key=secret-value"))
    assert "AKIAEXAMPLE" not in _safe_error(RuntimeError("aws_access_key_id=AKIAEXAMPLE"))


def test_mono_run_reconstructs_unpublished_mix_and_records_derived_digest(tmp_path: Path):
    config = replace(load_run_config("whisper-large-v3-portable"), channels=("mono",))
    dataset = _dataset(tmp_path / "dataset")
    report = preflight_run(dataset, config, backend=FakeBackend())
    assert report.ok
    backend = FakeBackend()
    result = run_inference(dataset, config, tmp_path / "output", backend=backend)

    assert result.status == "complete"
    assert backend.calls == [("mono.wav", "en-us")]
    derived = tmp_path / "output/derived-audio/clip-1/mono.wav"
    hypothesis = json.loads((tmp_path / "output/hypotheses.jsonl").read_text(encoding="utf-8"))
    assert hashlib.sha256(derived.read_bytes()).hexdigest() == hypothesis["audio_sha256"]
    assert result.manifest["audio_preparation"]["mono"]["policy"] == "mono-mix-v1"  # type: ignore[index]


def test_mono_resume_rebuilds_a_tampered_unpublished_mix(tmp_path: Path):
    config = replace(load_run_config("whisper-large-v3-portable"), channels=("mono",))
    dataset = _dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    backend = FakeBackend()

    first = run_inference(dataset, config, output, backend=backend)
    mono = output / "derived-audio/clip-1/mono.wav"
    original_digest = hashlib.sha256(mono.read_bytes()).hexdigest()
    mono.write_bytes(b"tampered")

    second = run_inference(dataset, config, output, backend=backend)

    assert first.status == second.status == "complete"
    assert second.completed == 0
    assert second.skipped == 1
    assert hashlib.sha256(mono.read_bytes()).hexdigest() == original_digest
    assert backend.calls == [("mono.wav", "en-us")]


def test_word_alignments_drive_word_timing_and_outer_diarization_intervals(tmp_path: Path):
    dataset = _dataset(tmp_path / "dataset")
    config = replace(load_run_config("whisper-large-v3-portable"), channels=("mono",))

    run_inference(dataset, config, tmp_path / "output", backend=FakeBackend())

    hypothesis = json.loads((tmp_path / "output/hypotheses.jsonl").read_text(encoding="utf-8"))
    assert hypothesis["reference_timed"] == [
        {
            "speaker": "ch1",
            "start_s": 0.001,
            "end_s": 0.004,
            "text": "hello",
            "timing_source": "forced",
        },
        {
            "speaker": "ch2",
            "start_s": 0.006,
            "end_s": 0.009,
            "text": "world",
            "timing_source": "forced",
        },
    ]
    assert hypothesis["reference_diarization"] == [
        {"speaker": "ch1", "start_s": 0.0, "end_s": 0.005},
        {"speaker": "ch2", "start_s": 0.005, "end_s": 0.01},
    ]


def test_mono_run_never_calls_the_model_or_writes_audio_on_a_digest_mismatch(tmp_path: Path):
    dataset = _dataset(tmp_path / "dataset", include_mono_hash=True)
    selection_path = dataset / SELECTION_FILENAME
    row = json.loads(selection_path.read_text(encoding="utf-8"))
    row["mono_mix_sha256"] = "f" * 64
    selection_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = replace(load_run_config("whisper-large-v3-portable"), channels=("mono",))
    backend = FakeBackend()

    result = run_inference(dataset, config, tmp_path / "output", backend=backend)

    assert result.status == "incomplete_with_failures"
    assert result.failed == 1
    assert backend.calls == []
    assert not (tmp_path / "output/derived-audio/clip-1/mono.wav").exists()
