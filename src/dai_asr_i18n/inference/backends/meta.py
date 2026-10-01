"""Meta Muse Voice Transcribe one-shot file backend."""

from __future__ import annotations

import json
import math
import os
import subprocess
import uuid
from pathlib import Path

from dai_asr_i18n.inference.backends.hosted import HostedBackend, checked_request, require_httpx
from dai_asr_i18n.inference.base import Transcript, TranscriptSegment
from dai_asr_i18n.inference.config import ModelConfig
from dai_asr_i18n.inference.windowing import planned_audio_windows, stitch_transcripts

ENDPOINT = "https://api.meta.ai/v1/asr/transcribe"
_REQUEST_LIMIT_BYTES = 32_000_000
_OPTIONS = {"mode", "audio_encoding", "language_bias", "max_input_s", "window_target_factor"}
_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mr": "Marathi",
    "pt": "Portuguese",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Mandarin Chinese",
}


def _wav24k(audio_path: Path) -> bytes:
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
            "24000",
            "-c:a",
            "pcm_s16le",
            "-map_metadata",
            "-1",
            "-f",
            "wav",
            "pipe:1",
        ],
        capture_output=True,
        timeout=600,
        check=False,
    )
    if process.returncode or not process.stdout:
        raise ValueError(f"ffmpeg failed to prepare the Meta upload (exit {process.returncode})")
    if len(process.stdout) > _REQUEST_LIMIT_BYTES:
        raise ValueError("prepared Meta upload exceeds the 32 MB request limit")
    return process.stdout


def _turn_interval(turn: dict[str, object]) -> tuple[float, float]:
    start = turn.get("startMs")
    end = turn.get("endMs")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (start, end)):
        raise ValueError("Meta Muse turn lacks numeric timestamps")
    start_s, end_s = float(start) / 1000.0, float(end) / 1000.0
    if not 0 <= start_s <= end_s < math.inf:
        raise ValueError("Meta Muse turn has invalid timestamps")
    return start_s, end_s


class MetaMuseBackend(HostedBackend):
    provider = "Meta"

    def __init__(self, config: ModelConfig, *, credentials: dict[str, str] | None = None) -> None:
        super().__init__(config, credentials=credentials)
        if config.backend != "meta-muse":
            raise ValueError(f"MetaMuseBackend cannot load backend {config.backend!r}")
        unknown = set(config.options) - _OPTIONS
        if unknown:
            raise ValueError(f"unsupported Meta Muse options: {', '.join(sorted(unknown))}")

    def _api_key(self) -> str:
        for name in ("MODEL_API_KEY", "META_API_KEY", "MUSE_KEY"):
            value = self._credentials.get(name) or os.environ.get(name)
            if value:
                return value
        raise ValueError("missing Meta Model API key (MODEL_API_KEY, META_API_KEY, or MUSE_KEY)")

    def preflight(self) -> dict[str, object]:
        try:
            require_httpx()
            self._api_key()
        except ValueError as error:
            return {"ok": False, "error": str(error), "identity": self.identity}
        return {"ok": True, "error": None, "identity": self.identity}

    def _transcribe_one(self, audio_path: Path, language: str | None) -> Transcript:
        httpx = require_httpx()
        code = language.split("-")[0].lower() if language else None
        request = {
            "model": self.config.model_id,
            "mode": str(self.config.options.get("mode", "DIARIZATION")),
            "audioEncoding": str(self.config.options.get("audio_encoding", "WAV")),
        }
        if self.config.options.get("language_bias", True) and code in _LANGUAGE_NAMES:
            request["languageBias"] = [_LANGUAGE_NAMES[code]]
        response = checked_request(
            lambda: httpx.post(
                ENDPOINT,
                params={"sessionId": str(uuid.uuid4())},
                headers={"Authorization": f"Bearer {self._api_key()}", "Accept": "application/json"},
                files={
                    "request": (None, json.dumps(request), "application/json"),
                    "audio": ("audio.wav", _wav24k(audio_path), "audio/wav"),
                },
                timeout=900,
            )
        )
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("turns"), list):
            raise ValueError("Meta Muse response lacks diarization turns")
        if data.get("transcript") is not None and not isinstance(data["transcript"], str):
            raise ValueError("Meta Muse response has an invalid transcript")
        segments: list[TranscriptSegment] = []
        for turn in data["turns"]:
            if not isinstance(turn, dict) or not isinstance(turn.get("transcript"), str):
                raise ValueError("Meta Muse response contains an invalid turn")
            text = turn["transcript"].strip()
            if text and turn.get("speaker") is None:
                raise ValueError("Meta Muse diarization turn lacks a speaker label")
            start_s, end_s = _turn_interval(turn)
            segments.append(
                TranscriptSegment(
                    text=text,
                    start_s=start_s,
                    end_s=end_s,
                    speaker=str(turn["speaker"]) if turn.get("speaker") is not None else None,
                )
            )
        transcript_text = data.get("transcript") or " ".join(segment.text for segment in segments)
        if transcript_text.strip() and not any(segment.text for segment in segments):
            raise ValueError("Meta Muse returned a non-empty transcript without diarization turns")
        duration_ms = data.get("audioDurationMs")
        if duration_ms is not None and (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, (int, float))
            or not 0 <= float(duration_ms) < math.inf
        ):
            raise ValueError("Meta Muse returned an invalid audio duration")
        return Transcript(
            text=transcript_text.strip(),
            language=code,
            duration_s=(float(duration_ms) / 1000.0 if duration_ms is not None else None),
            segments=tuple(segments),
            metadata={
                "request": {
                    "endpoint": ENDPOINT,
                    "parameters": request,
                    "input": {"codec": "wav", "sample_rate_hz": 24000},
                }
            },
        )

    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        maximum = float(self.config.options.get("max_input_s", 600.0))
        factor = float(self.config.options.get("window_target_factor", 0.98))
        with planned_audio_windows(audio_path, max_input_s=maximum, target_factor=factor) as windows:
            results = [(window, self._transcribe_one(window.path, language)) for window in windows]
        return results[0][1] if len(results) == 1 else stitch_transcripts(results, language=language)


__all__ = ["MetaMuseBackend"]
