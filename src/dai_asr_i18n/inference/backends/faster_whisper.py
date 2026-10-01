"""Pinned local Whisper inference through faster-whisper/CTranslate2."""

from __future__ import annotations

import importlib.metadata
import math
from pathlib import Path
from typing import Any

from dai_asr_i18n.inference.base import InferenceBackend, Transcript, TranscriptSegment, TranscriptWord
from dai_asr_i18n.inference.config import ModelConfig

_SUPPORTED_OPTIONS = {
    "beam_size",
    "word_timestamps",
    "task",
    "temperature",
    "condition_on_previous_text",
    "vad_filter",
}


class FasterWhisperBackend(InferenceBackend):
    """Load one pinned converted Whisper checkpoint and transcribe local audio."""

    def __init__(self, config: ModelConfig) -> None:
        if config.backend != "faster-whisper":
            raise ValueError(f"FasterWhisperBackend cannot load backend {config.backend!r}")
        unknown = set(config.options) - _SUPPORTED_OPTIONS
        if unknown:
            raise ValueError(f"unsupported faster-whisper options: {', '.join(sorted(unknown))}")
        beam_size = config.options.get("beam_size")
        if isinstance(beam_size, bool) or not isinstance(beam_size, int) or beam_size < 1:
            raise ValueError("faster-whisper beam_size must be a positive integer")
        for option in ("word_timestamps", "condition_on_previous_text", "vad_filter"):
            if not isinstance(config.options.get(option), bool):
                raise ValueError(f"faster-whisper {option} must be true or false")
        if config.options.get("task") != "transcribe":
            raise ValueError("the benchmark faster-whisper task must be 'transcribe'")
        temperature = config.options.get("temperature")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("faster-whisper temperature must be a finite non-negative number")
        self.config = config
        self._model: Any | None = None

    @property
    def identity(self) -> dict[str, object]:
        def version(distribution: str) -> str | None:
            try:
                return importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                return None

        return {
            "backend": self.config.backend,
            "backend_version": version("faster-whisper"),
            "ctranslate2_version": version("ctranslate2"),
            "runtime_versions": {
                distribution: version(distribution)
                for distribution in (
                    "av",
                    "ctranslate2",
                    "faster-whisper",
                    "huggingface-hub",
                    "librosa",
                    "numpy",
                    "soundfile",
                    "tokenizers",
                )
            },
            "model_id": self.config.model_id,
            "model_revision": self.config.revision,
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "authenticated": self.config.authenticated,
            "options": self.config.options,
        }

    def preflight(self) -> dict[str, object]:
        try:
            import ctranslate2
            import faster_whisper  # noqa: F401
        except ImportError:
            return {
                "ok": False,
                "error": 'faster-whisper is not installed; run: pip install "dai-asr-i18n[local]"',
                "identity": self.identity,
            }
        cuda_devices = ctranslate2.get_cuda_device_count()
        device_ok = self.config.device != "cuda" or cuda_devices > 0
        return {
            "ok": device_ok,
            "error": None if device_ok else "the configuration requires CUDA but CTranslate2 found no CUDA device",
            "cuda_devices": cuda_devices,
            "identity": self.identity,
        }

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:
                raise ValueError('faster-whisper is not installed; run: pip install "dai-asr-i18n[local]"') from error
            self._model = WhisperModel(
                self.config.model_id,
                revision=self.config.revision,
                device=self.config.device,
                compute_type=self.config.compute_type,
                use_auth_token=True if self.config.authenticated else False,
            )
        return self._model

    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        model = self._load()
        options = dict(self.config.options)
        options["language"] = language.split("-")[0] if language else None
        segments, info = model.transcribe(str(audio_path), **options)
        shaped_segments: list[TranscriptSegment] = []
        texts: list[str] = []
        for segment in segments:
            texts.append(segment.text)
            words = tuple(
                TranscriptWord(
                    text=word.word,
                    start_s=float(word.start) if word.start is not None else None,
                    end_s=float(word.end) if word.end is not None else None,
                    confidence=float(word.probability) if word.probability is not None else None,
                )
                for word in (segment.words or ())
            )
            shaped_segments.append(
                TranscriptSegment(
                    text=segment.text,
                    start_s=float(segment.start),
                    end_s=float(segment.end),
                    words=words,
                )
            )
        return Transcript(
            text="".join(texts),
            language=getattr(info, "language", None),
            duration_s=float(info.duration) if getattr(info, "duration", None) is not None else None,
            segments=tuple(shaped_segments),
            metadata={
                "language_probability": (
                    float(info.language_probability)
                    if getattr(info, "language_probability", None) is not None
                    else None
                )
            },
        )
