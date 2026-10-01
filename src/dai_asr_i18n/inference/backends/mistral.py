"""Mistral Voxtral file-transcription backend with native diarization."""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

from dai_asr_i18n.inference.backends.hosted import HostedBackend, checked_request, require_httpx
from dai_asr_i18n.inference.base import Transcript, TranscriptSegment
from dai_asr_i18n.inference.config import ModelConfig

ENDPOINT = "https://api.mistral.ai/v1/audio/transcriptions"
_OPTIONS = {"diarize", "timestamp_granularities", "temperature", "stream", "language_hint"}


def _lexical_coverage_key(text: str) -> str:
    """Ignore presentation-only differences while preserving lexical Unicode content."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if unicodedata.category(character)[0] in "LMN")


class MistralVoxtralBackend(HostedBackend):
    provider = "Mistral"

    def __init__(self, config: ModelConfig, *, credentials: dict[str, str] | None = None) -> None:
        super().__init__(config, credentials=credentials)
        if config.backend != "mistral-voxtral":
            raise ValueError(f"MistralVoxtralBackend cannot load backend {config.backend!r}")
        unknown = set(config.options) - _OPTIONS
        if unknown:
            raise ValueError(f"unsupported Voxtral options: {', '.join(sorted(unknown))}")
        if "MISTRAL_API_KEY" not in config.environment:
            raise ValueError("Voxtral configs must declare MISTRAL_API_KEY")

    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        httpx = require_httpx()
        parameters = {
            "model": self.config.model_id,
            "diarize": bool(self.config.options.get("diarize", True)),
            "temperature": float(self.config.options.get("temperature", 0.0)),
            "stream": bool(self.config.options.get("stream", False)),
        }
        granularities = list(self.config.options.get("timestamp_granularities", ["segment"]))
        fields = [
            (name, (None, json.dumps(value) if not isinstance(value, str) else value))
            for name, value in parameters.items()
        ]
        fields.extend(("timestamp_granularities", (None, str(value))) for value in granularities)
        if self.config.options.get("language_hint") and language:
            fields.append(("language", (None, language.split("-")[0].lower())))
        fields.append(("file", (audio_path.name, audio_path.read_bytes(), "audio/wav")))
        response = checked_request(
            lambda: httpx.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {self.credential('MISTRAL_API_KEY')}"},
                files=fields,
                timeout=1800,
            )
        )
        data = response.json()
        if not isinstance(data.get("text"), str) or not isinstance(data.get("segments"), list):
            raise ValueError("Voxtral response lacks transcript/segments")
        if data.get("model") != self.config.model_id:
            raise ValueError("Voxtral response does not identify the requested model")
        segments: list[TranscriptSegment] = []
        for item in data["segments"]:
            text = item.get("text")
            speaker = item.get("speaker_id")
            if not isinstance(text, str) or (text.strip() and speaker is None):
                raise ValueError("Voxtral diarization segment lacks text or speaker")
            if not text.strip():
                continue
            start, end = item.get("start"), item.get("end")
            if any(
                value is not None
                and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value))
                for value in (start, end)
            ):
                raise ValueError("Voxtral returned an invalid timestamp")
            segments.append(TranscriptSegment(text=text, start_s=start, end_s=end, speaker=str(speaker)))
        segment_text = "".join(segment.text for segment in segments)
        exact_coverage = "".join(data["text"].split()) == "".join(segment_text.split())
        if _lexical_coverage_key(data["text"]) != _lexical_coverage_key(segment_text):
            raise ValueError("Voxtral speaker segments do not cover the returned transcript")
        return Transcript(
            text=data["text"],
            language=data.get("language") or (language.split("-")[0] if language else None),
            segments=tuple(segments),
            metadata={
                "request": {
                    "endpoint": ENDPOINT,
                    "parameters": {**parameters, "timestamp_granularities": granularities},
                },
                "segment_text_matches_top_level": exact_coverage,
            },
        )


__all__ = ["MistralVoxtralBackend"]
