"""Integrity validation for downloaded public benchmark selections."""

from __future__ import annotations

import hashlib
import json
import re
import wave
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from dai_asr_i18n.datasets.models import (
    RECEIPT_FILENAME,
    SELECTION_FILENAME,
    BenchmarkClip,
    DatasetSelection,
    huggingface_repo_id,
    safe_relative_path,
)

_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _load_json(value: str, *, source: str) -> Any:
    try:
        return json.loads(value, parse_constant=_reject_constant, object_pairs_hook=_object_without_duplicates)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {source}: {error.msg}") from error


def load_manifest(path: Path) -> tuple[BenchmarkClip, ...]:
    """Load and strictly validate a version-1 JSONL benchmark manifest."""

    clips: list[BenchmarkClip] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            clip = BenchmarkClip.from_dict(_load_json(line, source=f"{path}:{line_number}"), line_number=line_number)
            if clip.clip_id in seen:
                raise ValueError(f"duplicate sample_id {clip.clip_id!r} on line {line_number}")
            seen.add(clip.clip_id)
            clips.append(clip)
    if not clips:
        raise ValueError(f"manifest {path} contains no clips")
    return tuple(clips)


def load_selection(root: str | Path) -> DatasetSelection:
    """Load a materialized selection and its immutable source provenance receipt."""

    directory = Path(root).expanduser().resolve()
    receipt_path = directory / RECEIPT_FILENAME
    selection_path = directory / SELECTION_FILENAME
    if not receipt_path.is_file() or not selection_path.is_file():
        raise ValueError(f"{directory} is not a DAI-ASR-I18N dataset selection")
    if receipt_path.is_symlink() or selection_path.is_symlink():
        raise ValueError(f"{directory} contains symbolic-link selection metadata")
    receipt = _load_json(receipt_path.read_text(encoding="utf-8"), source=str(receipt_path))
    if not isinstance(receipt, dict):
        raise ValueError(f"{receipt_path} must contain an object")
    schema_version = receipt.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError(f"{receipt_path} uses an unsupported schema_version")
    if schema_version == 1:
        receipt = {
            **receipt,
            "source_type": "huggingface",
            "source_id": receipt.get("repo_id"),
            "revision_type": "git_commit",
        }
    required = [
        "source_type",
        "source_id",
        "revision",
        "resolved_revision",
        "revision_type",
        "source_manifest",
    ]
    if schema_version == 2:
        required.extend(("source_manifest_sha256", "selection_sha256"))
    missing = [field for field in required if not isinstance(receipt.get(field), str) or not receipt[field]]
    if missing:
        raise ValueError(f"{receipt_path} is missing fields: {', '.join(missing)}")
    source_type = receipt["source_type"]
    source_id = receipt["source_id"]
    resolved_revision = receipt["resolved_revision"].lower()
    revision_type = receipt["revision_type"]
    if source_type != "huggingface":
        raise ValueError(f"{receipt_path} has unsupported source_type {source_type!r}")
    try:
        source_id = huggingface_repo_id(source_id)
    except ValueError as error:
        raise ValueError(f"{receipt_path} has an invalid Hugging Face source_id") from error
    if revision_type != "git_commit" or not _FULL_COMMIT.fullmatch(resolved_revision):
        raise ValueError(f"{receipt_path} Hugging Face revision must be a full 40-character commit SHA")
    source_manifest = safe_relative_path(receipt["source_manifest"], "source_manifest")
    source_manifest_sha256 = receipt.get("source_manifest_sha256")
    expected_selection_sha256 = receipt.get("selection_sha256")
    if source_manifest_sha256 is not None and (
        not isinstance(source_manifest_sha256, str) or not _SHA256.fullmatch(source_manifest_sha256)
    ):
        raise ValueError(f"{receipt_path} has an invalid source_manifest_sha256")
    if expected_selection_sha256 is not None and (
        not isinstance(expected_selection_sha256, str) or not _SHA256.fullmatch(expected_selection_sha256)
    ):
        raise ValueError(f"{receipt_path} has an invalid selection_sha256")
    clips = load_manifest(selection_path)
    selection = DatasetSelection(
        root=directory,
        source_type=source_type,
        source_id=source_id,
        revision=receipt["revision"],
        resolved_revision=resolved_revision,
        revision_type=revision_type,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        selection_sha256="",
        clips=clips,
    )
    actual_selection_sha256 = selection.digest()
    if expected_selection_sha256 is not None and actual_selection_sha256 != expected_selection_sha256:
        raise ValueError(
            f"selection checksum mismatch: expected {expected_selection_sha256}, got {actual_selection_sha256}"
        )
    return replace(selection, selection_sha256=actual_selection_sha256)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_duration(
    actual_duration: float, expected_duration_s: float, tolerance: float, clip_id: str, channel: str
) -> None:
    if abs(actual_duration - expected_duration_s) > tolerance:
        raise ValueError(
            f"duration mismatch for clip {clip_id!r} channel {channel}: "
            f"manifest={expected_duration_s}, audio={actual_duration}"
        )


def _verify_wav(path: Path, *, expected_duration_s: float, clip_id: str, channel: str) -> None:
    try:
        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            frames = stream.getnframes()
            compression = stream.getcomptype()
    except (EOFError, wave.Error) as error:
        raise ValueError(f"invalid WAV for clip {clip_id!r} channel {channel}: {error}") from error
    if channels != 1 or sample_width not in {1, 2, 3, 4} or sample_rate < 1 or compression != "NONE":
        raise ValueError(
            f"clip {clip_id!r} channel {channel} must be an uncompressed mono PCM WAV; "
            f"got channels={channels}, sample_width={sample_width}, sample_rate={sample_rate}, "
            f"compression={compression}"
        )
    _verify_duration(frames / sample_rate, expected_duration_s, max(0.001, 1 / sample_rate), clip_id, channel)


def _verify_flac(path: Path, *, expected_duration_s: float, clip_id: str, channel: str) -> None:
    try:
        import soundfile as sf
    except ImportError as error:
        raise ValueError(
            'FLAC validation requires a dataset extra: pip install "dai-asr-i18n[hf]" or "dai-asr-i18n[local]"'
        ) from error
    try:
        info = sf.info(path)
    except RuntimeError as error:
        raise ValueError(f"invalid FLAC for clip {clip_id!r} channel {channel}: {error}") from error
    if info.format != "FLAC" or info.channels != 1 or info.subtype != "PCM_16" or info.samplerate < 1:
        raise ValueError(
            f"clip {clip_id!r} channel {channel} must be 16-bit mono FLAC; "
            f"got format={info.format}, channels={info.channels}, subtype={info.subtype}, "
            f"sample_rate={info.samplerate}"
        )
    _verify_duration(
        info.frames / info.samplerate,
        expected_duration_s,
        max(0.001, 1 / info.samplerate),
        clip_id,
        channel,
    )


def _verify_audio(path: Path, *, expected_duration_s: float, clip_id: str, channel: str) -> None:
    if path.suffix.lower() == ".wav":
        _verify_wav(path, expected_duration_s=expected_duration_s, clip_id=clip_id, channel=channel)
    elif path.suffix.lower() == ".flac":
        _verify_flac(path, expected_duration_s=expected_duration_s, clip_id=clip_id, channel=channel)
    else:  # guarded by the manifest model, retained as defense in depth
        raise ValueError(f"unsupported audio extension for clip {clip_id!r} channel {channel}: {path.suffix}")


@dataclass(frozen=True)
class VerificationReport:
    """Result of checking every selected clip and audio digest."""

    dataset: DatasetSelection
    files_checked: int
    total_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": True,
            "source_type": self.dataset.source_type,
            "source_id": self.dataset.source_id,
            "revision": self.dataset.revision,
            "resolved_revision": self.dataset.resolved_revision,
            "revision_type": self.dataset.revision_type,
            "source_manifest_sha256": self.dataset.source_manifest_sha256,
            "selection_sha256": self.dataset.digest(),
            "clips": len(self.dataset.clips),
            "languages": sorted({clip.language for clip in self.dataset.clips}),
            "files_checked": self.files_checked,
            "total_bytes": self.total_bytes,
        }


def verify_clip_assets(root: Path, clips: tuple[BenchmarkClip, ...]) -> tuple[int, int]:
    """Verify file containment, checksums, and the v1 WAV contract for supplied clips."""

    root = root.resolve()
    files_checked = 0
    total_bytes = 0
    for clip in clips:
        for channel, asset in clip.audio.items():
            relative_parts = PurePosixPath(asset.path).parts
            candidate = root.joinpath(*relative_parts)
            current = root
            for part in relative_parts:
                current /= part
                if current.is_symlink():
                    raise ValueError(f"clip {clip.clip_id!r} channel {channel} uses symbolic-link path {current}")
            path = candidate.resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"clip {clip.clip_id!r} channel {channel} escapes the dataset root")
            if not path.is_file():
                raise ValueError(f"missing audio for clip {clip.clip_id!r} channel {channel}: {asset.path}")
            actual = sha256_file(path)
            if actual != asset.sha256:
                raise ValueError(
                    f"checksum mismatch for clip {clip.clip_id!r} channel {channel}: "
                    f"expected {asset.sha256}, got {actual}"
                )
            _verify_audio(
                path,
                expected_duration_s=asset.duration_s,
                clip_id=clip.clip_id,
                channel=channel,
            )
            files_checked += 1
            total_bytes += path.stat().st_size
    return files_checked, total_bytes


def verify_selection(root: str | Path) -> VerificationReport:
    """Fail closed when an expected file is missing, escapes the root, or has changed."""

    dataset = load_selection(root)
    files_checked, total_bytes = verify_clip_assets(dataset.root, dataset.clips)
    return VerificationReport(dataset=dataset, files_checked=files_checked, total_bytes=total_bytes)
