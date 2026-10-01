"""Versioned, portable manifest models for public ASR benchmark clips."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

DATASET_SCHEMA_VERSION = 2
SUPPORTED_DATASET_SCHEMA_VERSIONS = {1, DATASET_SCHEMA_VERSION}
SELECTION_FILENAME = "selection.jsonl"
RECEIPT_FILENAME = "dataset-receipt.json"
_SAFE_CLIP_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
_LANGUAGE_TAG = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{1,8})*")
_REMOTE_NAME_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
_TIMING_TOLERANCE_S = 0.05


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def safe_relative_path(value: object, field_name: str) -> str:
    """Validate a repository-relative POSIX path before joining it to local storage."""

    text = _require_string(value, field_name)
    if "\\" in text:
        raise ValueError(f"{field_name} must be a safe repository-relative POSIX path")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must be a safe repository-relative path")
    return path.as_posix()


def huggingface_repo_id(value: object) -> str:
    """Validate the canonical ``namespace/name`` form accepted by the Hub."""

    text = _require_string(value, "repo_id")
    parts = text.split("/")
    if (
        len(text) > 96
        or len(parts) != 2
        or any(not _REMOTE_NAME_PART.fullmatch(part) for part in parts)
        or any(part.startswith(("-", ".")) or part.endswith(("-", ".")) for part in parts)
        or "--" in text
        or ".." in text
        or text.endswith(".git")
    ):
        raise ValueError("repo_id must be a valid Hugging Face organization/name")
    return text


def _require_sha256(value: object, field_name: str) -> str:
    text = _require_string(value, field_name).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _require_finite_number(value: object, field_name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{field_name} must be finite and >= {minimum}")
    return number


@dataclass(frozen=True)
class AudioAsset:
    """One immutable audio object referenced by a benchmark clip."""

    path: str
    sha256: str
    duration_s: float

    @classmethod
    def from_dict(cls, value: object, field_name: str) -> AudioAsset:
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be an object")
        return cls(
            path=safe_relative_path(value.get("path"), f"{field_name}.path"),
            sha256=_require_sha256(value.get("sha256"), f"{field_name}.sha256"),
            duration_s=_require_finite_number(value.get("duration_s"), f"{field_name}.duration_s"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "duration_s": self.duration_s}


@dataclass(frozen=True)
class ReferenceSegment:
    """An authoritative clip-relative reference segment."""

    text: str
    start_s: float
    end_s: float

    @classmethod
    def from_dict(cls, value: object, field_name: str) -> ReferenceSegment:
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be an object")
        text = value.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{field_name}.text must be a string")
        start_s = _require_finite_number(value.get("start_s"), f"{field_name}.start_s")
        end_s = _require_finite_number(value.get("end_s"), f"{field_name}.end_s")
        if end_s < start_s:
            raise ValueError(f"{field_name}.end_s must be >= start_s")
        return cls(text=text, start_s=start_s, end_s=end_s)

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "start_s": self.start_s, "end_s": self.end_s}


@dataclass(frozen=True)
class AlignedWord:
    """One word with clip-relative timing from the released dataset."""

    text: str
    start_s: float
    end_s: float

    @classmethod
    def from_dict(cls, value: object, field_name: str) -> AlignedWord:
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be an object")
        text = value.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{field_name}.text must be a string")
        start_s = _require_finite_number(value.get("start_s"), f"{field_name}.start_s")
        end_s = _require_finite_number(value.get("end_s"), f"{field_name}.end_s")
        if end_s < start_s:
            raise ValueError(f"{field_name}.end_s must be >= start_s")
        return cls(text=text, start_s=start_s, end_s=end_s)

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "start_s": self.start_s, "end_s": self.end_s}


@dataclass(frozen=True)
class WordAlignmentSegment:
    """One aligned reference segment and its word-level timing."""

    start_s: float
    end_s: float
    timing_source: str | None
    words: tuple[AlignedWord, ...]

    @classmethod
    def from_dict(cls, value: object, field_name: str) -> WordAlignmentSegment:
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be an object")
        start_s = _require_finite_number(value.get("start_s"), f"{field_name}.start_s")
        end_s = _require_finite_number(value.get("end_s"), f"{field_name}.end_s")
        if end_s < start_s:
            raise ValueError(f"{field_name}.end_s must be >= start_s")
        timing_source = value.get("timing_source")
        if timing_source is not None and not isinstance(timing_source, str):
            raise ValueError(f"{field_name}.timing_source must be a string or null")
        words_value = value.get("words")
        if not isinstance(words_value, list):
            raise ValueError(f"{field_name}.words must be an array")
        words = tuple(
            AlignedWord.from_dict(word, f"{field_name}.words[{index}]") for index, word in enumerate(words_value)
        )
        for index, word in enumerate(words):
            if word.start_s < start_s - 1e-6 or word.end_s > end_s + 1e-6:
                raise ValueError(f"{field_name}.words[{index}] falls outside its aligned segment")
        return cls(start_s=start_s, end_s=end_s, timing_source=timing_source, words=words)

    def to_dict(self) -> dict[str, object]:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "timing_source": self.timing_source,
            "words": [word.to_dict() for word in self.words],
        }


@dataclass(frozen=True)
class BenchmarkClip:
    """One public benchmark item with physical channels and scoring annotations."""

    clip_id: str
    language: str
    duration_s: float
    audio: dict[str, AudioAsset]
    references: dict[str, tuple[ReferenceSegment, ...]]
    word_alignments: dict[str, tuple[WordAlignmentSegment, ...]] = field(default_factory=dict)
    mono_mix_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: object, *, line_number: int | None = None) -> BenchmarkClip:
        where = f"manifest line {line_number}" if line_number is not None else "manifest row"
        if not isinstance(value, dict):
            raise ValueError(f"{where} must be an object")
        unknown = set(value) - {
            "schema_version",
            "sample_id",
            "clip_id",
            "language",
            "duration_s",
            "audio",
            "references",
            "word_alignments",
            "mono_mix_sha256",
            "metadata",
        }
        if unknown:
            raise ValueError(f"{where} has unknown fields: {', '.join(sorted(unknown))}")
        schema_version = value.get("schema_version", DATASET_SCHEMA_VERSION)
        if schema_version not in SUPPORTED_DATASET_SCHEMA_VERSIONS:
            raise ValueError(f"{where} uses unsupported schema_version {schema_version!r}")
        if value.get("sample_id") is not None and value.get("clip_id") is not None:
            raise ValueError(f"{where} must not contain both sample_id and clip_id")
        identifier_field = "sample_id" if value.get("sample_id") is not None else "clip_id"
        clip_id = _require_string(value.get(identifier_field), f"{where}.{identifier_field}")
        if not _SAFE_CLIP_ID.fullmatch(clip_id) or clip_id in {".", ".."}:
            raise ValueError(f"{where}.{identifier_field} must be a safe portable identifier")
        language = _require_string(value.get("language"), f"{where}.language").replace("_", "-").lower()
        if not _LANGUAGE_TAG.fullmatch(language):
            raise ValueError(f"{where}.language must be a simple BCP-47 language tag")
        duration_s = _require_finite_number(value.get("duration_s"), f"{where}.duration_s")
        if duration_s <= 0:
            raise ValueError(f"{where}.duration_s must be positive")

        audio_value = value.get("audio")
        if not isinstance(audio_value, dict) or not audio_value:
            raise ValueError(f"{where}.audio must be a non-empty object")
        allowed_channels = {"ch1", "ch2"}
        if set(audio_value) - allowed_channels:
            raise ValueError(f"{where}.audio supports only ch1 and ch2 source channels")
        audio = {
            channel: AudioAsset.from_dict(asset, f"{where}.audio.{channel}") for channel, asset in audio_value.items()
        }
        for channel, asset in audio.items():
            if Path(asset.path).suffix.lower() not in {".flac", ".wav"}:
                raise ValueError(f"{where}.audio.{channel}.path must name a FLAC or WAV file")
            if asset.duration_s <= 0:
                raise ValueError(f"{where}.audio.{channel}.duration_s must be positive")
            if abs(asset.duration_s - duration_s) > _TIMING_TOLERANCE_S:
                raise ValueError(f"{where}.audio.{channel}.duration_s must agree with clip duration within 50 ms")

        references_value = value.get("references", {})
        if not isinstance(references_value, dict):
            raise ValueError(f"{where}.references must be an object")
        if schema_version == 1 and set(references_value) != set(audio):
            raise ValueError(f"{where}.references must contain exactly the source audio channels")
        if schema_version == DATASET_SCHEMA_VERSION and "references" in value:
            raise ValueError(f"{where}.references is not allowed in schema_version {DATASET_SCHEMA_VERSION}")
        references: dict[str, tuple[ReferenceSegment, ...]] = {}
        for channel, segments_value in references_value.items():
            if not isinstance(segments_value, list):
                raise ValueError(f"{where}.references.{channel} must be an array")
            references[channel] = tuple(
                ReferenceSegment.from_dict(segment, f"{where}.references.{channel}[{index}]")
                for index, segment in enumerate(segments_value)
            )
            for index, segment in enumerate(references[channel]):
                if segment.end_s > duration_s + _TIMING_TOLERANCE_S:
                    raise ValueError(f"{where}.references.{channel}[{index}] exceeds clip duration")

        word_alignments_value = value.get("word_alignments", {})
        if not isinstance(word_alignments_value, dict):
            raise ValueError(f"{where}.word_alignments must be an object")
        if schema_version == DATASET_SCHEMA_VERSION and set(word_alignments_value) != set(audio):
            raise ValueError(f"{where}.word_alignments must contain exactly the source audio channels")
        if set(word_alignments_value) - set(audio):
            raise ValueError(f"{where}.word_alignments must contain only source audio channels")
        word_alignments: dict[str, tuple[WordAlignmentSegment, ...]] = {}
        for channel, segments_value in word_alignments_value.items():
            if not isinstance(segments_value, list):
                raise ValueError(f"{where}.word_alignments.{channel} must be an array")
            word_alignments[channel] = tuple(
                WordAlignmentSegment.from_dict(segment, f"{where}.word_alignments.{channel}[{index}]")
                for index, segment in enumerate(segments_value)
            )
            for index, segment in enumerate(word_alignments[channel]):
                if segment.end_s > duration_s + _TIMING_TOLERANCE_S:
                    raise ValueError(f"{where}.word_alignments.{channel}[{index}] exceeds clip duration")

        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{where}.metadata must be an object")
        mono_mix_sha256_value = value.get("mono_mix_sha256")
        mono_mix_sha256 = (
            None
            if mono_mix_sha256_value is None
            else _require_sha256(mono_mix_sha256_value, f"{where}.mono_mix_sha256")
        )
        if mono_mix_sha256 is not None and not {"ch1", "ch2"}.issubset(audio):
            raise ValueError(f"{where}.mono_mix_sha256 requires both ch1 and ch2 audio")
        return cls(
            clip_id=clip_id,
            language=language,
            duration_s=duration_s,
            audio=audio,
            references=references,
            word_alignments=word_alignments,
            mono_mix_sha256=mono_mix_sha256,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, object]:
        alignment_native = not self.references and set(self.word_alignments) == set(self.audio)
        value: dict[str, object] = {
            "schema_version": DATASET_SCHEMA_VERSION if alignment_native else 1,
            "sample_id": self.clip_id,
            "language": self.language,
            "duration_s": self.duration_s,
            "audio": {channel: asset.to_dict() for channel, asset in sorted(self.audio.items())},
            "metadata": self.metadata,
        }
        if self.references:
            value["references"] = {
                channel: [segment.to_dict() for segment in segments]
                for channel, segments in sorted(self.references.items())
            }
        if self.word_alignments:
            value["word_alignments"] = {
                channel: [segment.to_dict() for segment in segments]
                for channel, segments in sorted(self.word_alignments.items())
            }
        if self.mono_mix_sha256 is not None:
            value["mono_mix_sha256"] = self.mono_mix_sha256
        return value

    def reference_text(self, channel: str) -> str:
        if self.references:
            if channel in self.references:
                return " ".join(segment.text.strip() for segment in self.references[channel] if segment.text.strip())
            if channel == "mono":
                ordered_segments = sorted(
                    (
                        (segment.start_s, segment.end_s, source_channel, index, segment.text.strip())
                        for source_channel, segments in self.references.items()
                        for index, segment in enumerate(segments)
                        if segment.text.strip()
                    ),
                    key=lambda item: item[:4],
                )
                return " ".join(item[4] for item in ordered_segments)
            raise ValueError(f"clip {self.clip_id!r} has no reference for channel {channel!r}")
        if channel in self.word_alignments:
            ordered = sorted(
                (
                    (word.start_s, word.end_s, segment_index, word_index, word.text.strip())
                    for segment_index, segment in enumerate(self.word_alignments[channel])
                    for word_index, word in enumerate(segment.words)
                    if word.text.strip()
                ),
                key=lambda item: item[:4],
            )
            return " ".join(item[4] for item in ordered)
        if channel == "mono" and self.word_alignments:
            ordered_words = sorted(
                (
                    (
                        word.start_s,
                        word.end_s,
                        source_channel,
                        segment_index,
                        word_index,
                        word.text.strip(),
                    )
                    for source_channel, segments in self.word_alignments.items()
                    for segment_index, segment in enumerate(segments)
                    for word_index, word in enumerate(segment.words)
                    if word.text.strip()
                ),
                key=lambda item: item[:5],
            )
            return " ".join(item[5] for item in ordered_words)
        raise ValueError(f"clip {self.clip_id!r} has no reference for channel {channel!r}")


@dataclass(frozen=True)
class DatasetSelection:
    """A locally materialized, pinned selection from one remote dataset source."""

    root: Path
    source_type: str
    source_id: str
    revision: str
    resolved_revision: str
    revision_type: str
    source_manifest: str
    source_manifest_sha256: str | None
    selection_sha256: str
    clips: tuple[BenchmarkClip, ...]

    @property
    def repo_id(self) -> str:
        """Hugging Face repository identifier."""

        return self.source_id

    @property
    def selection_path(self) -> Path:
        return self.root / SELECTION_FILENAME

    @property
    def receipt_path(self) -> Path:
        return self.root / RECEIPT_FILENAME

    def asset_path(self, asset: AudioAsset) -> Path:
        return self.root.joinpath(*PurePosixPath(asset.path).parts)

    def digest(self) -> str:
        hasher = hashlib.sha256()
        for clip in self.clips:
            hasher.update(
                json.dumps(clip.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            )
            hasher.update(b"\n")
        return hasher.hexdigest()
