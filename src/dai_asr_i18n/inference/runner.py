"""Read-only dataset execution with resumable, provenance-rich inference outputs."""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dai_asr_i18n.audio import MONO_MIX_POLICY_ID, build_mono_mix, mono_mix_manifest
from dai_asr_i18n.datasets import DatasetSelection, load_selection, verify_selection
from dai_asr_i18n.datasets.models import AudioAsset, BenchmarkClip
from dai_asr_i18n.datasets.validation import sha256_file
from dai_asr_i18n.inference.backends import (
    DeepgramBackend,
    ElevenLabsBackend,
    FasterWhisperBackend,
    GeminiTranscribeBackend,
    MaiTranscribeBackend,
    MetaMuseBackend,
    MistralVoxtralBackend,
    OpenAITranscriptionBackend,
    Qwen3AsrBackend,
    XaiGrokBackend,
)
from dai_asr_i18n.inference.base import InferenceBackend
from dai_asr_i18n.inference.config import RunConfig

HYPOTHESES_FILENAME = "hypotheses.jsonl"
FAILURES_FILENAME = "failures.jsonl"
RUN_MANIFEST_FILENAME = "run-manifest.json"
EFFECTIVE_CONFIG_FILENAME = "effective-run-spec.json"
INFERENCE_PROTOCOL_VERSION = "dai-asr-i18n-inference-v5"
SYSTEMATIC_FAILURE_THRESHOLD = 5
_HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{10,}")
_OPENAI_TOKEN = re.compile(r"sk-[A-Za-z0-9_-]{10,}")
_CREDENTIAL_VALUE = re.compile(
    r"(?i)(authorization|api[_-]?key|token|access[_-]?key(?:[_-]?id)?|secret[_-]?access[_-]?key)"
    r"(\s*[:=]\s*)([^\s,;&]+)"
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_error(error: Exception) -> str:
    text = _HF_TOKEN.sub("[REDACTED_HF_TOKEN]", str(error))
    text = _OPENAI_TOKEN.sub("[REDACTED_API_KEY]", text)
    return _CREDENTIAL_VALUE.sub(r"\1\2[REDACTED]", text)


def _backend_for(config: RunConfig) -> InferenceBackend:
    backends = {
        "deepgram": DeepgramBackend,
        "elevenlabs": ElevenLabsBackend,
        "faster-whisper": FasterWhisperBackend,
        "gemini-transcribe": GeminiTranscribeBackend,
        "mai-transcribe": MaiTranscribeBackend,
        "meta-muse": MetaMuseBackend,
        "mistral-voxtral": MistralVoxtralBackend,
        "openai-transcription": OpenAITranscriptionBackend,
        "qwen3-asr": Qwen3AsrBackend,
        "xai-grok": XaiGrokBackend,
    }
    backend = backends.get(config.model.backend)
    if backend is not None:
        return backend(config.model)
    raise ValueError(f"unsupported inference backend {config.model.backend!r}")


def _package_version() -> str:
    injected = os.environ.get("DAI_ASR_I18N_PACKAGE_VERSION")
    if injected:
        return injected
    try:
        return importlib.metadata.version("dai-asr-i18n")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def _implementation_identity() -> dict[str, str | None]:
    return {
        "package_version": _package_version(),
        "source_sha256": os.environ.get("DAI_ASR_I18N_SOURCE_SHA256") or None,
    }


def _stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _task_id(
    dataset: DatasetSelection,
    clip: BenchmarkClip,
    channel: str,
    audio_sha256: str,
    config: RunConfig,
    backend_identity: dict[str, object],
    implementation_identity: dict[str, str | None],
) -> str:
    return _stable_digest(
        {
            "dataset_source_type": dataset.source_type,
            "dataset_source_id": dataset.source_id,
            "dataset_revision": dataset.resolved_revision,
            "sample_id": clip.clip_id,
            "channel": channel,
            "audio_sha256": audio_sha256,
            "run_config_sha256": config.digest(),
            "backend": backend_identity,
            "implementation": implementation_identity,
            "inference_protocol": INFERENCE_PROTOCOL_VERSION,
        }
    )


def _append_jsonl(path: Path, value: object) -> None:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())


def _completed_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path} line {line_number}: {error.msg}") from error
            task_id = value.get("task_id") if isinstance(value, dict) else None
            if not isinstance(task_id, str):
                raise ValueError(f"missing task_id in {path} line {line_number}")
            if task_id in completed:
                raise ValueError(f"duplicate task_id {task_id!r} in {path}")
            completed.add(task_id)
    return completed


def _jsonl_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _hardware() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
    }


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    dataset: dict[str, object]
    backend: dict[str, object]
    config_sha256: str
    tasks: int
    audio_hours: float
    missing_channel_tasks: int
    missing_channels: tuple[dict[str, object], ...]
    selection: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "dataset": self.dataset,
            "backend": self.backend,
            "config_sha256": self.config_sha256,
            "tasks": self.tasks,
            "audio_hours": self.audio_hours,
            "missing_channel_tasks": self.missing_channel_tasks,
            "missing_channels": list(self.missing_channels),
            "selection": self.selection,
            "writes_performed": False,
        }


def _available_channels(clip: BenchmarkClip, configured: tuple[str, ...]) -> tuple[str, ...]:
    result = []
    for channel in configured:
        if channel in clip.audio:
            result.append(channel)
        elif channel == "mono" and {"ch1", "ch2"}.issubset(clip.audio):
            result.append(channel)
    return tuple(result)


def _validate_shard(shard_index: int, shard_count: int) -> None:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be between 0 and shard_count - 1")


def _in_shard(clip: BenchmarkClip, channel: str, shard_index: int, shard_count: int) -> bool:
    coordinate = f"{clip.clip_id}\0{channel}".encode()
    return int.from_bytes(hashlib.sha256(coordinate).digest()[:8], "big") % shard_count == shard_index


def _selected_clips(
    dataset: DatasetSelection,
    languages: tuple[str, ...],
    sample_ids: tuple[str, ...],
) -> tuple[BenchmarkClip, ...]:
    language_set = {language.replace("_", "-").lower() for language in languages}
    sample_set = set(sample_ids)
    clips = tuple(clip for clip in dataset.clips if not language_set or clip.language in language_set)
    if sample_set:
        available = {clip.clip_id for clip in clips}
        missing = sample_set - available
        if missing:
            raise ValueError(f"requested sample IDs were not found after filtering: {', '.join(sorted(missing))}")
        clips = tuple(clip for clip in clips if clip.clip_id in sample_set)
    if not clips:
        raise ValueError("the inference filters selected no clips")
    return clips


def _selected_tasks(
    dataset: DatasetSelection,
    config: RunConfig,
    *,
    languages: tuple[str, ...],
    sample_ids: tuple[str, ...],
    shard_index: int,
    shard_count: int,
) -> tuple[tuple[BenchmarkClip, str], ...]:
    _validate_shard(shard_index, shard_count)
    return tuple(
        (clip, channel)
        for clip in _selected_clips(dataset, languages, sample_ids)
        for channel in _available_channels(clip, config.channels)
        if _in_shard(clip, channel, shard_index, shard_count)
    )


def _inference_language(clip: BenchmarkClip, config: RunConfig) -> str:
    """Use an explicit dataset locale only for the GPT-Transcribe hint contract."""
    if config.model.backend != "openai-transcription" or config.model.model_id != "gpt-transcribe":
        return clip.language
    locales = [clip.metadata[key] for key in ("locale", "benchmark_locale") if clip.metadata.get(key)]
    normalized = []
    for locale in locales:
        if not isinstance(locale, str):
            raise ValueError("dataset locale must be a language code string")
        value = locale.replace("_", "-").lower()
        if value.split("-")[0] != clip.language.replace("_", "-").lower().split("-")[0]:
            raise ValueError("dataset locale conflicts with the clip language")
        normalized.append(value)
    if len(set(normalized)) > 1:
        raise ValueError("dataset locale and benchmark_locale disagree")
    return normalized[0] if normalized else clip.language


def preflight_run(
    dataset_root: str | Path,
    config: RunConfig,
    *,
    backend: InferenceBackend | None = None,
    languages: tuple[str, ...] = (),
    sample_ids: tuple[str, ...] = (),
    shard_index: int = 0,
    shard_count: int = 1,
) -> PreflightReport:
    """Verify all inputs and runtime dependencies without creating files or loading weights."""

    verification = verify_selection(dataset_root)
    missing: list[dict[str, object]] = []
    missing_channel_tasks = 0
    clips = _selected_clips(verification.dataset, languages, sample_ids)
    for clip in clips:
        from dai_asr_i18n.evaluation.connector import reference_from_clip

        reference_from_clip(clip)
        _inference_language(clip, config)
        channels = [
            channel
            for channel in config.channels
            if channel not in _available_channels(clip, (channel,))
            and _in_shard(clip, channel, shard_index, shard_count)
        ]
        if channels:
            missing.append({"sample_id": clip.clip_id, "missing": channels})
            missing_channel_tasks += len(channels)
    missing_channels = tuple(missing[:20])
    tasks = _selected_tasks(
        verification.dataset,
        config,
        languages=languages,
        sample_ids=sample_ids,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    selected_tasks = len(tasks)
    audio_seconds = sum(clip.duration_s for clip, _ in tasks)
    selected_backend = backend or _backend_for(config)
    backend_report = selected_backend.preflight()
    return PreflightReport(
        ok=bool(backend_report.get("ok")) and (selected_tasks > 0 or shard_count > 1) and missing_channel_tasks == 0,
        dataset=verification.to_dict(),
        backend=backend_report,
        config_sha256=config.digest(),
        tasks=selected_tasks,
        audio_hours=audio_seconds / 3600.0,
        missing_channel_tasks=missing_channel_tasks,
        missing_channels=missing_channels,
        selection={
            "languages": list(languages),
            "sample_ids": list(sample_ids),
            "shard_index": shard_index,
            "shard_count": shard_count,
        },
    )


def _reference_by_speaker(clip: BenchmarkClip, channel: str) -> dict[str, str]:
    channels = tuple(sorted(clip.audio)) if channel == "mono" else (channel,)
    return {speaker: clip.reference_text(speaker) for speaker in channels}


def _reference_timed(clip: BenchmarkClip, channel: str) -> list[dict[str, object]]:
    channels = tuple(sorted(clip.audio)) if channel == "mono" else (channel,)
    timed: list[dict[str, object]] = []
    for speaker in channels:
        alignments = clip.word_alignments.get(speaker, ())
        aligned_words = [
            {
                "speaker": speaker,
                "start_s": word.start_s,
                "end_s": word.end_s,
                "text": word.text,
                "timing_source": segment.timing_source,
            }
            for segment in alignments
            for word in segment.words
        ]
        if aligned_words:
            timed.extend(aligned_words)
        elif speaker in clip.references:  # legacy schema-version 1 manifests
            timed.extend(
                {
                    "speaker": speaker,
                    "start_s": segment.start_s,
                    "end_s": segment.end_s,
                    "text": segment.text,
                }
                for segment in clip.references[speaker]
            )
    return sorted(timed, key=lambda item: (item["start_s"], item["end_s"], item["speaker"]))


def _reference_diarization(clip: BenchmarkClip, channel: str) -> list[dict[str, object]]:
    """Return the same merged aligned-word activity used by diarization scoring."""

    channels = tuple(sorted(clip.audio)) if channel == "mono" else (channel,)
    activity: list[dict[str, object]] = []
    for speaker in channels:
        alignments = clip.word_alignments.get(speaker, ())
        if alignments or speaker in clip.word_alignments:
            from dai_asr_i18n.evaluation._reference_validation import merge_speech_intervals

            activity.extend(
                {"speaker": speaker, "start_s": start, "end_s": end}
                for start, end in merge_speech_intervals(
                    [(word.start_s, word.end_s) for segment in alignments for word in segment.words]
                )
            )
        elif speaker in clip.references:  # legacy schema-version 1 manifests
            activity.extend(
                {"speaker": speaker, "start_s": segment.start_s, "end_s": segment.end_s}
                for segment in clip.references[speaker]
            )
    return sorted(activity, key=lambda item: (item["start_s"], item["end_s"], item["speaker"]))


def _audio_for_task(
    dataset: DatasetSelection,
    clip: BenchmarkClip,
    channel: str,
    output: Path,
) -> tuple[Path, str]:
    if channel in clip.audio:
        asset: AudioAsset = clip.audio[channel]
        return dataset.asset_path(asset), asset.sha256
    if channel != "mono" or not {"ch1", "ch2"}.issubset(clip.audio):
        raise ValueError(f"clip {clip.clip_id!r} does not support channel {channel!r}")
    mono_path = output / "derived-audio" / clip.clip_id / "mono.wav"
    if mono_path.is_file() and clip.mono_mix_sha256 is not None:
        actual = sha256_file(mono_path)
        if hmac.compare_digest(actual, clip.mono_mix_sha256):
            return mono_path, actual
    result = build_mono_mix(
        dataset.asset_path(clip.audio["ch1"]),
        dataset.asset_path(clip.audio["ch2"]),
    )
    if clip.mono_mix_sha256 is not None and not hmac.compare_digest(result.sha256, clip.mono_mix_sha256):
        raise ValueError(
            f"mono mix checksum mismatch for clip {clip.clip_id!r}: "
            f"expected {clip.mono_mix_sha256}, got {result.sha256}"
        )
    if mono_path.is_file() and hmac.compare_digest(sha256_file(mono_path), result.sha256):
        return mono_path, result.sha256
    mono_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = mono_path.with_name(f".{mono_path.name}.tmp")
    if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
        raise ValueError(f"refusing to replace unsafe temporary mono output {temporary}")
    temporary.unlink(missing_ok=True)
    temporary.write_bytes(result.wav_bytes)
    temporary.replace(mono_path)
    return mono_path, result.sha256


@dataclass(frozen=True)
class RunReport:
    output: Path
    status: str
    completed: int
    skipped: int
    failed: int
    total: int
    manifest: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.status == "complete" and self.failed == 0,
            "output": str(self.output),
            "status": self.status,
            "completed": self.completed,
            "skipped": self.skipped,
            "failed": self.failed,
            "total": self.total,
            "manifest": self.manifest,
        }


def run_inference(
    dataset_root: str | Path,
    config: RunConfig,
    output: str | Path,
    *,
    resume: bool = True,
    backend: InferenceBackend | None = None,
    languages: tuple[str, ...] = (),
    sample_ids: tuple[str, ...] = (),
    shard_index: int = 0,
    shard_count: int = 1,
) -> RunReport:
    """Run a deterministic task shard, fsync each result, and safely resume completed tasks."""

    selected_backend = backend or _backend_for(config)
    preflight = preflight_run(
        dataset_root,
        config,
        backend=selected_backend,
        languages=languages,
        sample_ids=sample_ids,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    if not preflight.ok:
        if preflight.missing_channel_tasks:
            reason = f"{preflight.missing_channel_tasks} configured channel tasks are missing from the dataset"
        else:
            reason = str(preflight.backend.get("error") or "no eligible inference tasks")
        raise ValueError(f"preflight failed: {reason}")
    dataset = load_selection(dataset_root)
    destination = Path(output).expanduser().resolve()
    if destination == dataset.root or destination.is_relative_to(dataset.root):
        raise ValueError("inference output must be outside the read-only dataset directory")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"inference output path is not a directory: {destination}")
    if destination.exists():
        for existing in destination.rglob("*"):
            if existing.is_symlink():
                raise ValueError(f"refusing to use inference output containing symbolic link {existing}")
    hypotheses_path = destination / HYPOTHESES_FILENAME
    failures_path = destination / FAILURES_FILENAME
    run_spec_path = destination / EFFECTIVE_CONFIG_FILENAME
    destination_nonempty = destination.exists() and any(destination.iterdir())
    if destination_nonempty and not resume:
        raise ValueError(f"output directory {destination} is not empty; enable resume or choose a new directory")
    implementation_identity = _implementation_identity()
    run_spec = {
        "schema_version": 2,
        "dataset_source_type": dataset.source_type,
        "dataset_source_id": dataset.source_id,
        "dataset_revision": dataset.resolved_revision,
        "selection_sha256": dataset.digest(),
        "config": config.to_dict(),
        "config_sha256": config.digest(),
        "backend": selected_backend.identity,
        "implementation": implementation_identity,
        "inference_protocol": INFERENCE_PROTOCOL_VERSION,
        "selection": preflight.selection,
    }
    if destination_nonempty:
        if not run_spec_path.is_file():
            raise ValueError(f"cannot resume {destination}: {EFFECTIVE_CONFIG_FILENAME} is missing")
        try:
            existing_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"cannot resume {destination}: invalid {EFFECTIVE_CONFIG_FILENAME}") from error
        if existing_spec != run_spec:
            raise ValueError("cannot resume into an output created for a different dataset selection or run config")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        run_spec_path.write_text(
            json.dumps(run_spec, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    from dai_asr_i18n.normalization import normalization_manifest

    scoring_manifest = normalization_manifest(include_opencc_hashes=False)
    scoring_run_id = hashlib.sha256(
        json.dumps(
            {
                "source": dataset.source_id,
                "revision": dataset.resolved_revision,
                "selection": dataset.digest(),
                "scoring": scoring_manifest,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    completed_ids = _completed_task_ids(hypotheses_path) if resume else set()

    started_at = _utc_now()
    completed = 0
    skipped = 0
    failed = 0
    attempted = 0
    abort_reason: str | None = None
    last_failure_signature: tuple[str, str] | None = None
    consecutive_failures = 0
    tasks = _selected_tasks(
        dataset,
        config,
        languages=languages,
        sample_ids=sample_ids,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    for clip, channel in tasks:
        attempted += 1
        task_id: str | None = None
        try:
            audio_path, audio_sha256 = _audio_for_task(dataset, clip, channel, destination)
            task_id = _task_id(
                dataset,
                clip,
                channel,
                audio_sha256,
                config,
                selected_backend.identity,
                implementation_identity,
            )
            if task_id in completed_ids:
                skipped += 1
                continue
            started = time.monotonic()
            transcript = selected_backend.transcribe(audio_path, language=_inference_language(clip, config))
            elapsed_s = time.monotonic() - started
            hypothesis_by_speaker = transcript.text_by_speaker()
            if not hypothesis_by_speaker and channel != "mono":
                hypothesis_by_speaker = {channel: transcript.text}
            record = {
                "schema_version": 2,
                "scoring_run_id": scoring_run_id,
                "task_id": task_id,
                "sample_id": clip.clip_id,
                "language": clip.language,
                "channel": channel,
                "reference": clip.reference_text(channel),
                "hypothesis": transcript.text,
                "reference_by_speaker": _reference_by_speaker(clip, channel),
                "hypothesis_by_speaker": hypothesis_by_speaker,
                "reference_timed": _reference_timed(clip, channel),
                "reference_diarization": _reference_diarization(clip, channel),
                "hypothesis_timed": transcript.timed_speaker_segments(),
                "audio_sha256": audio_sha256,
                "dataset_source_type": dataset.source_type,
                "dataset_source_id": dataset.source_id,
                "dataset_revision": dataset.resolved_revision,
                "run_config_sha256": config.digest(),
                "model": config.model.model_id,
                "model_version": config.model.revision or "provider-managed",
                "backend": selected_backend.identity,
                "latency_s": elapsed_s,
                "transcript": transcript.to_dict(),
                "input": clip.metadata,
            }
            from dai_asr_i18n.evaluation.connector import score_connector_output

            try:
                record["scores"] = score_connector_output(clip, transcript, channel, config.model)
            except Exception as error:  # preserve paid inference even if scoring is unavailable
                record["scores"] = {
                    "language": clip.language.split("-")[0],
                    "condition": channel,
                    "metrics": {},
                    "unavailable": {"scoring": _safe_error(error)},
                }
            _append_jsonl(hypotheses_path, record)
            completed += 1
            last_failure_signature = None
            consecutive_failures = 0
        except Exception as error:  # preserve the item-level failure and continue the cohort
            failed += 1
            safe_error = _safe_error(error)
            signature = (type(error).__name__, safe_error)
            consecutive_failures = consecutive_failures + 1 if signature == last_failure_signature else 1
            last_failure_signature = signature
            _append_jsonl(
                failures_path,
                {
                    "schema_version": 2,
                    "sample_id": clip.clip_id,
                    "task_id": task_id,
                    "language": clip.language,
                    "channel": channel,
                    "run_config_sha256": config.digest(),
                    "error_type": type(error).__name__,
                    "error": safe_error,
                    "recorded_at": _utc_now(),
                },
            )
            if consecutive_failures >= SYSTEMATIC_FAILURE_THRESHOLD:
                abort_reason = (
                    f"aborted after {consecutive_failures} consecutive identical failures: "
                    f"{signature[0]}: {signature[1]}"
                )
                break
        if abort_reason is not None:
            break

    from dai_asr_i18n.evaluation.connector import write_run_scores

    run_records = (
        [json.loads(line) for line in hypotheses_path.read_text().splitlines() if line.strip()]
        if hypotheses_path.exists()
        else []
    )
    write_run_scores(run_records, destination)
    finished_at = _utc_now()
    successful_total = skipped + completed
    failure_attempts_total = _jsonl_row_count(failures_path)
    if successful_total == preflight.tasks:
        status = "complete"
    elif abort_reason is not None:
        status = "incomplete_systematic_failure"
    else:
        status = "incomplete_with_failures"
    manifest = {
        "schema_version": 2,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "dai_asr_i18n_version": _package_version(),
        "implementation": implementation_identity,
        "dataset": preflight.dataset,
        "config": config.to_dict(),
        "config_source": config.source,
        "config_sha256": config.digest(),
        "selection": preflight.selection,
        "backend": selected_backend.identity,
        "inference_protocol": INFERENCE_PROTOCOL_VERSION,
        "audio_preparation": {
            "source_channels": "identity",
            "mono": mono_mix_manifest() | {"policy": MONO_MIX_POLICY_ID},
        },
        "hardware": _hardware(),
        "coverage": {
            "total": preflight.tasks,
            "considered_this_run": attempted,
            "inference_attempts_this_run": completed + failed,
            "completed_this_run": completed,
            "resumed": skipped,
            "failed_this_run": failed,
            "successful_total": successful_total,
            "remaining": preflight.tasks - successful_total,
            "failure_attempts_total": failure_attempts_total,
        },
        "abort_reason": abort_reason,
        "scoring": scoring_manifest,
    }
    (destination / RUN_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return RunReport(
        output=destination,
        status=status,
        completed=completed,
        skipped=skipped,
        failed=failed,
        total=preflight.tasks,
        manifest=manifest,
    )
