"""ElevenLabs Scribe v2 batch transcription from a local audio file."""

from __future__ import annotations

import math
import mimetypes
from pathlib import Path

from dai_asr_i18n.inference.backends.hosted import HostedBackend, checked_request, require_httpx, segments_from_words
from dai_asr_i18n.inference.base import Transcript, TranscriptWord
from dai_asr_i18n.inference.config import ModelConfig

ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
_LANGUAGE_ALIASES = {"tl": "fil"}
_OPTIONS = {"diarize", "timestamps_granularity", "temperature", "seed", "tag_audio_events"}


class ElevenLabsBackend(HostedBackend):
    provider = "ElevenLabs"

    def __init__(self, config: ModelConfig, *, credentials: dict[str, str] | None = None) -> None:
        super().__init__(config, credentials=credentials)
        if config.backend != "elevenlabs":
            raise ValueError(f"ElevenLabsBackend cannot load backend {config.backend!r}")
        unknown = set(config.options) - _OPTIONS
        if unknown:
            raise ValueError(f"unsupported ElevenLabs options: {', '.join(sorted(unknown))}")
        if "ELEVENLABS_API_KEY" not in config.environment:
            raise ValueError("ElevenLabs configs must declare ELEVENLABS_API_KEY")
        for option in ("diarize", "tag_audio_events"):
            if option in config.options and not isinstance(config.options[option], bool):
                raise ValueError(f"ElevenLabs {option} must be true or false")
        if config.diarizes and config.options.get("diarize") is not True:
            raise ValueError("ElevenLabs diarization configs must set diarize = true")
        if config.options.get("timestamps_granularity") not in {None, "word", "character"}:
            raise ValueError("ElevenLabs timestamps_granularity must be word or character")
        if "seed" in config.options and (
            isinstance(config.options["seed"], bool) or not isinstance(config.options["seed"], int)
        ):
            raise ValueError("ElevenLabs seed must be an integer")
        if "temperature" in config.options and (
            isinstance(config.options["temperature"], bool)
            or not isinstance(config.options["temperature"], (int, float))
            or not math.isfinite(config.options["temperature"])
        ):
            raise ValueError("ElevenLabs temperature must be finite")

    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        httpx = require_httpx()
        code = language.split("-")[0].lower() if language else None
        code = _LANGUAGE_ALIASES.get(code, code)
        parameters = {"model_id": self.config.model_id, **self.config.options}
        if code:
            parameters["language_code"] = code
        media_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"

        def request():
            # Open the upload for every attempt so a retry never reuses an exhausted stream.
            with audio_path.open("rb") as audio:
                return httpx.post(
                    ENDPOINT,
                    headers={"xi-api-key": self.credential("ELEVENLABS_API_KEY")},
                    data={
                        key: str(value).lower() if isinstance(value, bool) else str(value)
                        for key, value in parameters.items()
                    },
                    files={"file": (audio_path.name, audio, media_type)},
                    timeout=300,
                )

        response = checked_request(request)
        data = response.json()
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("text"), str)
            or not isinstance(data.get("words"), list)
        ):
            raise ValueError("ElevenLabs response lacks transcript/word fields")
        if any(not isinstance(word, dict) for word in data["words"]):
            raise ValueError("ElevenLabs response contains an invalid word")
        words = tuple(
            TranscriptWord(
                text=str(word.get("text") or ""),
                start_s=float(word["start"]) if word.get("start") is not None else None,
                end_s=float(word["end"]) if word.get("end") is not None else None,
                confidence=(
                    math.exp(float(word["logprob"]))
                    if word.get("logprob") is not None
                    else float(data.get("language_probability", 1.0))
                ),
                speaker=str(word["speaker_id"]) if word.get("speaker_id") is not None else None,
            )
            for word in data["words"]
            if word.get("type") == "word"
        )
        if data["text"].strip() and self.config.diarizes and not words:
            raise ValueError("ElevenLabs returned a non-empty transcript without diarized words")
        return Transcript(
            text=data["text"],
            language=data.get("language_code") or code,
            segments=segments_from_words(words, require_speakers=self.config.diarizes),
            metadata={
                "language_probability": data.get("language_probability"),
                "request": {"endpoint": ENDPOINT, "parameters": parameters, "input": "file"},
            },
        )


__all__ = ["ElevenLabsBackend"]
