"""OpenAI file-transcription backends with a stable speaker-aware response."""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

from dai_asr_i18n.inference.backends.hosted import HostedBackend, checked_request, require_httpx
from dai_asr_i18n.inference.base import Transcript, TranscriptSegment, TranscriptWord
from dai_asr_i18n.inference.config import ModelConfig

ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
_OPTIONS = {"response_format", "chunking_strategy", "temperature", "upload_bitrate"}
_REQUEST_LIMIT_BYTES = 25 * 1024 * 1024


def _mp3(audio_path: Path, bitrate: str) -> bytes:
    try:
        process = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                bitrate,
                "-f",
                "mp3",
                "pipe:1",
            ],
            capture_output=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"ffmpeg could not prepare the OpenAI upload: {error}") from error
    if process.returncode or not process.stdout:
        raise ValueError(f"ffmpeg failed to prepare the OpenAI upload (exit {process.returncode})")
    if len(process.stdout) > _REQUEST_LIMIT_BYTES:
        raise ValueError("prepared OpenAI upload exceeds the 25 MB request limit")
    return process.stdout


class OpenAITranscriptionBackend(HostedBackend):
    provider = "OpenAI"

    def __init__(self, config: ModelConfig, *, credentials: dict[str, str] | None = None) -> None:
        super().__init__(config, credentials=credentials)
        if config.backend != "openai-transcription":
            raise ValueError(f"OpenAITranscriptionBackend cannot load backend {config.backend!r}")
        unknown = set(config.options) - _OPTIONS
        if unknown:
            raise ValueError(f"unsupported OpenAI transcription options: {', '.join(sorted(unknown))}")
        if "OPENAI_API_KEY" not in config.environment:
            raise ValueError("OpenAI transcription configs must declare OPENAI_API_KEY")
        response_format = config.options.get("response_format", "json")
        if response_format not in {"json", "text", "diarized_json"}:
            raise ValueError("unsupported OpenAI transcription response_format")
        if config.diarizes and response_format != "diarized_json":
            raise ValueError("OpenAI diarization configs must request diarized_json")
        if config.options.get("chunking_strategy") not in {None, "auto"}:
            raise ValueError("OpenAI chunking_strategy must be auto when provided")
        if not re.fullmatch(r"[1-9][0-9]*k", str(config.options.get("upload_bitrate", "128k"))):
            raise ValueError("OpenAI upload_bitrate must be a positive integer followed by k")
        temperature = config.options.get("temperature")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 1
        ):
            raise ValueError("OpenAI temperature must be finite and between 0 and 1")

    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        httpx = require_httpx()
        bitrate = str(self.config.options.get("upload_bitrate", "128k"))
        audio = _mp3(audio_path, bitrate)
        parameters = {"model": self.config.model_id}
        parameters.update({key: value for key, value in self.config.options.items() if key != "upload_bitrate"})
        if language:
            parameters["language"] = language.split("-")[0].lower()
        response = checked_request(
            lambda: httpx.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {self.credential('OPENAI_API_KEY')}"},
                data=parameters,
                files={"file": ("audio.mp3", audio, "audio/mpeg")},
                timeout=900,
            )
        )
        if parameters.get("response_format") == "text":
            return Transcript(
                text=response.text.strip(),
                language=language.split("-")[0] if language else None,
                metadata={
                    "request": {
                        "endpoint": ENDPOINT,
                        "parameters": parameters,
                        "input": {"codec": "mp3", "sample_rate_hz": 16000, "bitrate": bitrate},
                    }
                },
            )
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("OpenAI transcription response must be an object")
        raw_words = data.get("words", [])
        raw_segments = data.get("segments", [])
        if not isinstance(raw_words, list) or not isinstance(raw_segments, list):
            raise ValueError("OpenAI transcription response has invalid word/segment fields")
        if any(not isinstance(word, dict) for word in raw_words):
            raise ValueError("OpenAI transcription response contains an invalid word")
        if data.get("text") is not None and not isinstance(data["text"], str):
            raise ValueError("OpenAI transcription response has an invalid text field")
        words = tuple(
            TranscriptWord(
                text=str(word.get("word") or word.get("text") or ""),
                start_s=float(word["start"]) if word.get("start") is not None else None,
                end_s=float(word["end"]) if word.get("end") is not None else None,
                speaker=str(word["speaker"]) if word.get("speaker") is not None else None,
            )
            for word in raw_words
        )
        shaped_segments: list[TranscriptSegment] = []
        for segment in raw_segments:
            if not isinstance(segment, dict):
                raise ValueError("OpenAI transcription response contains an invalid segment")
            text = str(segment.get("text") or "").strip()
            if text and self.config.diarizes and segment.get("speaker") is None:
                raise ValueError("OpenAI diarization segment lacks a speaker label")
            shaped_segments.append(
                TranscriptSegment(
                    text=text,
                    start_s=float(segment["start"]) if segment.get("start") is not None else None,
                    end_s=float(segment["end"]) if segment.get("end") is not None else None,
                    speaker=str(segment["speaker"]) if segment.get("speaker") is not None else None,
                )
            )
        segments = tuple(shaped_segments)
        text = str(data.get("text") or " ".join(segment.text for segment in segments)).strip()
        if self.config.diarizes and text and not any(segment.text for segment in segments):
            raise ValueError("OpenAI returned a non-empty transcript without diarized segments")
        if not self.config.diarizes and "text" not in data:
            raise ValueError("OpenAI transcription response lacks text")
        return Transcript(
            text=text,
            language=data.get("language") or (language.split("-")[0] if language else None),
            duration_s=float(data["duration"]) if data.get("duration") is not None else None,
            segments=segments or (() if not words else (TranscriptSegment(text=text, words=words),)),
            metadata={
                "request": {
                    "endpoint": ENDPOINT,
                    "parameters": parameters,
                    "input": {"codec": "mp3", "sample_rate_hz": 16000, "bitrate": bitrate},
                }
            },
        )


__all__ = ["OpenAITranscriptionBackend"]
