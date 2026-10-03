"""Microsoft MAI-Transcribe-2 from a local audio file."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

from dai_asr_i18n.inference.backends.hosted import HostedBackend, checked_request, require_httpx
from dai_asr_i18n.inference.base import Transcript, TranscriptSegment, TranscriptWord
from dai_asr_i18n.inference.config import ModelConfig
from dai_asr_i18n.inference.word_coverage import word_text_coverage_complete

_OPTIONS = {"api_version", "transcribe_style", "timestamps"}
_LANGUAGE_ALIASES = {"tl": "fil"}
_PUBLIC_ENDPOINT = "https://<azure-speech-resource>/speechtotext/transcriptions:transcribe"


def _interval(value: dict[str, object]) -> tuple[float, float]:
    start = float(value["offsetMilliseconds"]) / 1000
    end = start + float(value["durationMilliseconds"]) / 1000
    if not 0 <= start <= end < float("inf"):
        raise ValueError("MAI returned an invalid time interval")
    return start, end


class MaiTranscribeBackend(HostedBackend):
    provider = "Microsoft"

    def __init__(self, config: ModelConfig, *, credentials: dict[str, str] | None = None) -> None:
        super().__init__(config, credentials=credentials)
        if config.backend != "mai-transcribe":
            raise ValueError(f"MaiTranscribeBackend cannot load backend {config.backend!r}")
        unknown = set(config.options) - _OPTIONS
        if unknown:
            raise ValueError(f"unsupported MAI options: {', '.join(sorted(unknown))}")
        if not {"AZURE_SPEECH_KEY", "AZURE_SPEECH_ENDPOINT"}.issubset(config.environment):
            raise ValueError("MAI configs must declare AZURE_SPEECH_KEY and AZURE_SPEECH_ENDPOINT")
        if config.options.get("api_version") != "2025-10-15":
            raise ValueError("MAI api_version must be 2025-10-15")
        if config.options.get("transcribe_style") != "verbatim":
            raise ValueError("MAI transcribe_style must be verbatim")
        if config.options.get("timestamps") != "word":
            raise ValueError("MAI timestamps must be word")

    def _endpoint(self) -> str:
        endpoint = self.credential("AZURE_SPEECH_ENDPOINT").rstrip("/")
        parsed = urlparse(endpoint)
        if (
            parsed.scheme != "https"
            or parsed.path not in {"", "/"}
            or not parsed.hostname
            or not parsed.hostname.endswith((".cognitiveservices.azure.com", ".api.cognitive.microsoft.com"))
        ):
            raise ValueError("AZURE_SPEECH_ENDPOINT must be an HTTPS Azure Speech resource endpoint")
        return f"{endpoint}/speechtotext/transcriptions:transcribe"

    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        if not language:
            raise ValueError("MAI requires a language hint")
        code = _LANGUAGE_ALIASES.get(language.split("-")[0].lower(), language.split("-")[0].lower())
        definition = {
            "locales": [code],
            "enhancedMode": {
                "enabled": True,
                "model": self.config.model_id,
                "modelOptions": {
                    "timestamps": self.config.options["timestamps"],
                    "transcribeStyle": self.config.options["transcribe_style"],
                },
            },
            "diarization": {"enabled": True},
        }
        endpoint = self._endpoint()
        media_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
        httpx = require_httpx()

        def request():
            with audio_path.open("rb") as audio:
                return httpx.post(
                    endpoint,
                    params={"api-version": self.config.options["api_version"]},
                    headers={"Ocp-Apim-Subscription-Key": self.credential("AZURE_SPEECH_KEY")},
                    data={"definition": json.dumps(definition, separators=(",", ":"))},
                    files={"audio": (audio_path.name, audio, media_type)},
                    timeout=1800,
                )

        raw = checked_request(request).json()
        if not isinstance(raw.get("phrases"), list) or not isinstance(raw.get("combinedPhrases"), list):
            raise ValueError("MAI response lacks phrase fields")
        segments: list[TranscriptSegment] = []
        for phrase in raw["phrases"]:
            if not isinstance(phrase, dict) or phrase.get("speaker") is None:
                raise ValueError("MAI diarization response lacks speaker labels")
            start, end = _interval(phrase)
            speaker = str(phrase["speaker"])
            phrase_text = "" if phrase.get("text") is None else phrase["text"]
            phrase_words = phrase.get("words")
            if not isinstance(phrase_text, str) or (phrase_words is not None and not isinstance(phrase_words, list)):
                raise ValueError("MAI phrase requires text and a word list")
            if not phrase_text.strip() and not phrase_words:
                continue
            words = []
            for word in phrase_words or []:
                if not isinstance(word, dict) or not isinstance(word.get("text"), str):
                    raise ValueError("MAI word requires an object with string text")
                word_start, word_end = _interval(word)
                words.append(
                    TranscriptWord(
                        text=str(word["text"]),
                        start_s=word_start,
                        end_s=word_end,
                        confidence=(float(phrase["confidence"]) if phrase.get("confidence") is not None else None),
                        speaker=speaker,
                    )
                )
            segments.append(
                TranscriptSegment(
                    text=phrase_text.strip() or " ".join(w.text for w in words),
                    start_s=start,
                    end_s=end,
                    words=tuple(words),
                    speaker=speaker,
                )
            )
        if any(
            not isinstance(phrase, dict) or (phrase.get("text") is not None and not isinstance(phrase["text"], str))
            for phrase in raw["combinedPhrases"]
        ):
            raise ValueError("MAI combined phrase requires an object with string text")
        combined_text = " ".join(
            str(phrase.get("text") or "").strip()
            for phrase in raw["combinedPhrases"]
            if isinstance(phrase, dict) and str(phrase.get("text") or "").strip()
        )
        text = combined_text or " ".join(segment.text for segment in segments if segment.text)
        return Transcript(
            text=text,
            language=code,
            duration_s=(float(raw["durationMilliseconds"]) / 1000 if raw.get("durationMilliseconds") else None),
            segments=tuple(segments),
            metadata={
                "response": raw,
                **(
                    {"word_alignment_complete": False}
                    if not word_text_coverage_complete(
                        ((s.speaker, s.text) for s in segments),
                        ((w.speaker, w.text) for s in segments for w in s.words),
                    )
                    else {}
                ),
                "request": {
                    "endpoint": _PUBLIC_ENDPOINT,
                    "parameters": definition,
                    "api_version": self.config.options["api_version"],
                    "input": "file",
                },
            },
        )


__all__ = ["MaiTranscribeBackend"]
