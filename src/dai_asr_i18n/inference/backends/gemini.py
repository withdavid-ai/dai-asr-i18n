"""Google Gemini 3.5 Transcribe via the dedicated Interactions API."""

from __future__ import annotations

import logging
import mimetypes
import time
from pathlib import Path

from dai_asr_i18n.inference.backends.hosted import HostedBackend, checked_request, require_httpx, segments_from_words
from dai_asr_i18n.inference.base import Transcript, TranscriptWord
from dai_asr_i18n.inference.config import ModelConfig

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
SUPPORTED_LANGUAGES = frozenset("en es ar fr de hi it ja ko pt tr vi ru zh id tl th bn mr te".split())
_LANGUAGE_ALIASES = {"tl": "fil", "zh": "cmn"}
_OPTIONS = {"mode", "diarization_mode", "timestamp_granularities"}


def _google_genai():
    try:
        from google import genai
    except ImportError as error:
        raise ValueError('Gemini inference requires: pip install "dai-asr-i18n[hosted]"') from error
    return genai


def _seconds(value: object) -> float:
    number = float(value.removesuffix("s") if isinstance(value, str) else value)
    if not 0 <= number < float("inf"):
        raise ValueError("Gemini returned an invalid time offset")
    return number


class GeminiTranscribeBackend(HostedBackend):
    provider = "Google"

    def __init__(self, config: ModelConfig, *, credentials: dict[str, str] | None = None) -> None:
        super().__init__(config, credentials=credentials)
        if config.backend != "gemini-transcribe":
            raise ValueError(f"GeminiTranscribeBackend cannot load backend {config.backend!r}")
        unknown = set(config.options) - _OPTIONS
        if unknown:
            raise ValueError(f"unsupported Gemini options: {', '.join(sorted(unknown))}")
        if "GEMINI_API_KEY" not in config.environment:
            raise ValueError("Gemini configs must declare GEMINI_API_KEY")
        if config.options.get("mode") != "verbatim":
            raise ValueError("Gemini mode must be verbatim")
        if config.options.get("diarization_mode") != "speaker":
            raise ValueError("Gemini diarization_mode must be speaker")
        if config.options.get("timestamp_granularities") != ["word"]:
            raise ValueError("Gemini timestamp_granularities must be [word]")

    def preflight(self) -> dict[str, object]:
        report = super().preflight()
        if not report["ok"]:
            return report
        try:
            _google_genai()
        except ValueError as error:
            return {**report, "ok": False, "error": str(error)}
        return report

    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        if not language:
            raise ValueError("Gemini Transcribe requires a language hint")
        language = language.split("-")[0].lower()
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Gemini Transcribe does not support language {language!r}")
        code = _LANGUAGE_ALIASES.get(language, language)
        mime_type = {
            ".flac": "audio/flac",
            ".wav": "audio/wav",
            ".mp3": "audio/mp3",
        }.get(audio_path.suffix.lower(), mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream")
        generation_config = {
            "transcription_config": {
                "mode": {
                    "type": self.config.options["mode"],
                    "diarization_mode": self.config.options["diarization_mode"],
                    "timestamp_granularities": self.config.options["timestamp_granularities"],
                },
                "language_codes": [code],
            }
        }
        genai = _google_genai()
        client = genai.Client(api_key=self.credential("GEMINI_API_KEY"))
        upload = None
        try:
            upload = client.files.upload(file=audio_path, config={"mime_type": mime_type})
            deadline = time.monotonic() + 300
            while str(getattr(upload.state, "name", upload.state)) == "PROCESSING":
                if time.monotonic() > deadline:
                    raise TimeoutError("Gemini audio upload did not become active")
                time.sleep(2)
                upload = client.files.get(name=upload.name)
            if str(getattr(upload.state, "name", upload.state)) != "ACTIVE":
                raise ValueError("Gemini audio upload did not become ACTIVE")
            httpx = require_httpx()
            response = checked_request(
                lambda: httpx.post(
                    ENDPOINT,
                    headers={"x-goog-api-key": self.credential("GEMINI_API_KEY")},
                    json={
                        "model": self.config.model_id,
                        "input": [{"type": "audio", "uri": upload.uri, "mime_type": mime_type}],
                        "generation_config": generation_config,
                    },
                    timeout=1800,
                )
            )
            raw = response.json()
            if raw.get("status") != "completed":
                raise ValueError(f"Gemini transcription did not complete (status={raw.get('status')!r})")
            words = []
            texts = []
            for step in raw.get("steps", []):
                if step.get("type") != "model_output":
                    continue
                for content in step.get("content", []):
                    if content.get("type") != "text":
                        continue
                    texts.append(str(content.get("text") or ""))
                    for word in content.get("annotations", []):
                        if word.get("type") != "word_info":
                            continue
                        if word.get("speaker") is None:
                            raise ValueError("Gemini word lacks speaker attribution")
                        words.append(
                            TranscriptWord(
                                text=str(word["text"]),
                                start_s=_seconds(word["start_offset"]),
                                end_s=_seconds(word["end_offset"]),
                                speaker=str(word["speaker"]),
                            )
                        )
            text = str(raw.get("output_text") or "".join(texts)).strip()
            if text and not words:
                raise ValueError("Gemini returned text without requested word/speaker annotations")
            word_tuple = tuple(words)
            return Transcript(
                text=text,
                language=language,
                segments=segments_from_words(word_tuple),
                metadata={
                    "request": {
                        "endpoint": ENDPOINT,
                        "parameters": generation_config,
                        "input": {"upload": "Gemini Files API", "mime_type": mime_type},
                    }
                },
            )
        finally:
            if upload is not None:
                try:
                    client.files.delete(name=upload.name)
                except Exception:
                    logging.getLogger(__name__).warning("Could not remove the temporary Gemini upload")
            client.close()


__all__ = ["GeminiTranscribeBackend", "SUPPORTED_LANGUAGES"]
