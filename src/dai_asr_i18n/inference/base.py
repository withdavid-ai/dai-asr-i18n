"""Backend-neutral ASR inference contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TranscriptWord:
    text: str
    start_s: float | None = None
    end_s: float | None = None
    confidence: float | None = None
    speaker: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "confidence": self.confidence,
            "speaker": self.speaker,
        }


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start_s: float | None = None
    end_s: float | None = None
    words: tuple[TranscriptWord, ...] = ()
    speaker: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "words": [word.to_dict() for word in self.words],
            "speaker": self.speaker,
        }


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None
    duration_s: float | None = None
    segments: tuple[TranscriptSegment, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def text_by_speaker(self) -> dict[str, str]:
        """Concatenate speaker-attributed text without inventing labels."""

        parts: dict[str, list[str]] = {}
        attributed_segments = [segment for segment in self.segments if segment.speaker is not None]
        if attributed_segments:
            for segment in attributed_segments:
                if segment.text.strip():
                    parts.setdefault(str(segment.speaker), []).append(segment.text.strip())
        else:
            for segment in self.segments:
                for word in segment.words:
                    if word.speaker is not None and word.text.strip():
                        parts.setdefault(str(word.speaker), []).append(word.text.strip())
        return {speaker: " ".join(texts) for speaker, texts in parts.items()}

    def timed_speaker_segments(self) -> list[dict[str, object]]:
        """Return complete speaker-attributed intervals suitable for time-constrained scoring."""

        return [
            {
                "speaker": str(segment.speaker),
                "start_s": segment.start_s,
                "end_s": segment.end_s,
                "text": segment.text,
            }
            for segment in self.segments
            if segment.speaker is not None and segment.start_s is not None and segment.end_s is not None
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "language": self.language,
            "duration_s": self.duration_s,
            "segments": [segment.to_dict() for segment in self.segments],
            "metadata": self.metadata,
        }


class InferenceBackend(ABC):
    """One model runtime with a stable output shape."""

    @property
    @abstractmethod
    def identity(self) -> dict[str, object]:
        """Return model, revision, backend, runtime, device, and precision identity."""

    @abstractmethod
    def preflight(self) -> dict[str, object]:
        """Check dependencies and hardware without downloading weights or writing files."""

    @abstractmethod
    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        """Run one inference call from a local audio file."""
