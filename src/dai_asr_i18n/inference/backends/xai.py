"""xAI Grok Voice Transcribe file backend."""

from __future__ import annotations

import math
from pathlib import Path

from dai_asr_i18n.inference.backends.hosted import HostedBackend, checked_request, require_httpx, segments_from_words
from dai_asr_i18n.inference.base import Transcript, TranscriptWord
from dai_asr_i18n.inference.config import ModelConfig

ENDPOINT = "https://api.x.ai/v1/stt"
_OPTIONS = {"diarize", "format", "filler_words", "vad_threshold", "language_hint"}


class XaiGrokBackend(HostedBackend):
    provider = "xAI"

    def __init__(self, config: ModelConfig, *, credentials: dict[str, str] | None = None) -> None:
        super().__init__(config, credentials=credentials)
        if config.backend != "xai-grok":
            raise ValueError(f"XaiGrokBackend cannot load backend {config.backend!r}")
        unknown = set(config.options) - _OPTIONS
        if unknown:
            raise ValueError(f"unsupported xAI options: {', '.join(sorted(unknown))}")
        if "XAI_API_KEY" not in config.environment:
            raise ValueError("xAI configs must declare XAI_API_KEY")

    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        httpx = require_httpx()
        code = language.split("-")[0].lower() if language else None
        if code == "tl":
            code = "fil"
        parameters = {
            "model": self.config.model_id,
            "diarize": str(bool(self.config.options.get("diarize", True))).lower(),
            "format": str(bool(self.config.options.get("format", False))).lower(),
            "filler_words": str(bool(self.config.options.get("filler_words", True))).lower(),
            "vad_threshold": str(float(self.config.options.get("vad_threshold", 0.5))),
        }
        if self.config.options.get("language_hint", True) and code:
            parameters["language"] = code
        fields = [(name, (None, value)) for name, value in parameters.items()]
        fields.append(("file", (audio_path.name, audio_path.read_bytes(), "audio/wav")))
        response = checked_request(
            lambda: httpx.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {self.credential('XAI_API_KEY')}"},
                files=fields,
                timeout=1800,
            )
        )
        data = response.json()
        text = data.get("text")
        raw_words = data.get("words")
        if isinstance(text, str) and not text.strip() and raw_words is None:
            raw_words = []
        if not isinstance(text, str) or not isinstance(raw_words, list):
            raise ValueError("xAI response lacks transcript/word fields")
        if text.strip() and not raw_words:
            raise ValueError("xAI returned a non-empty transcript without diarized words")
        words: list[TranscriptWord] = []
        for item in raw_words:
            start, end = float(item["start"]), float(item["end"])
            if not (0 <= start <= end < math.inf) or item.get("speaker") is None:
                raise ValueError("xAI diarization word has invalid timing or no speaker")
            words.append(TranscriptWord(text=str(item["text"]), start_s=start, end_s=end, speaker=str(item["speaker"])))
        word_tuple = tuple(words)
        return Transcript(
            text=text,
            language=data.get("language") or code,
            duration_s=float(data["duration"]) if data.get("duration") is not None else None,
            segments=segments_from_words(word_tuple),
            metadata={"request": {"endpoint": ENDPOINT, "parameters": parameters}},
        )


__all__ = ["XaiGrokBackend"]
