"""Deepgram pre-recorded transcription from a local audio file."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from dai_asr_i18n.inference.backends.hosted import HostedBackend, checked_request, require_httpx, segments_from_words
from dai_asr_i18n.inference.base import Transcript, TranscriptWord
from dai_asr_i18n.inference.config import ModelConfig

ENDPOINT = "https://api.deepgram.com/v1/listen"
_OPTIONS = {
    "smart_format",
    "diarize",
    "diarize_model",
    "paragraphs",
    "utterances",
    "filler_words",
    "mip_opt_out",
}


class DeepgramBackend(HostedBackend):
    provider = "Deepgram"

    def __init__(self, config: ModelConfig, *, credentials: dict[str, str] | None = None) -> None:
        super().__init__(config, credentials=credentials)
        if config.backend != "deepgram":
            raise ValueError(f"DeepgramBackend cannot load backend {config.backend!r}")
        unknown = set(config.options) - _OPTIONS
        if unknown:
            raise ValueError(f"unsupported Deepgram options: {', '.join(sorted(unknown))}")
        if "DEEPGRAM_API_KEY" not in config.environment:
            raise ValueError("Deepgram configs must declare DEEPGRAM_API_KEY")
        boolean_options = set(config.options) - {"diarize_model"}
        if any(not isinstance(config.options[name], bool) for name in boolean_options):
            raise ValueError("Deepgram benchmark options must be booleans")
        if config.options.get("diarize_model") not in {None, "latest", "v1", "v2"}:
            raise ValueError("Deepgram diarize_model must be latest, v1, or v2")
        if config.diarizes and not (
            config.options.get("diarize") is True or config.options.get("diarize_model") is not None
        ):
            raise ValueError("Deepgram diarization configs must select a diarization model")

    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        httpx = require_httpx()
        code = language.split("-")[0].lower() if language else None
        parameters = {"model": self.config.model_id, **self.config.options}
        if code:
            parameters["language"] = code
        media_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
        response = checked_request(
            lambda: httpx.post(
                ENDPOINT,
                headers={
                    "Authorization": f"Token {self.credential('DEEPGRAM_API_KEY')}",
                    "Content-Type": media_type,
                },
                params={
                    key: str(value).lower() if isinstance(value, bool) else str(value)
                    for key, value in parameters.items()
                },
                content=audio_path.read_bytes(),
                timeout=300,
            )
        )
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("results"), dict):
            raise ValueError("Deepgram response lacks results")
        channels = data["results"].get("channels")
        if not isinstance(channels, list) or not channels or not isinstance(channels[0], dict):
            raise ValueError("Deepgram response lacks channel results")
        alternatives = channels[0].get("alternatives")
        if not isinstance(alternatives, list) or not alternatives or not isinstance(alternatives[0], dict):
            raise ValueError("Deepgram response lacks transcription alternatives")
        alternative = alternatives[0]
        if not isinstance(alternative.get("transcript"), str) or not isinstance(alternative.get("words"), list):
            raise ValueError("Deepgram response lacks transcript/word fields")
        if any(not isinstance(word, dict) for word in alternative["words"]):
            raise ValueError("Deepgram response contains an invalid word")
        words = tuple(
            TranscriptWord(
                text=str(word.get("punctuated_word") or word.get("word") or ""),
                start_s=float(word["start"]) if word.get("start") is not None else None,
                end_s=float(word["end"]) if word.get("end") is not None else None,
                confidence=float(word["confidence"]) if word.get("confidence") is not None else None,
                speaker=str(word["speaker"]) if word.get("speaker") is not None else None,
            )
            for word in alternative["words"]
        )
        if alternative["transcript"].strip() and self.config.diarizes and not words:
            raise ValueError("Deepgram returned a non-empty transcript without diarized words")
        return Transcript(
            text=alternative["transcript"],
            language=code,
            segments=segments_from_words(words, require_speakers=self.config.diarizes),
            metadata={"request": {"endpoint": ENDPOINT, "parameters": parameters, "input": "file"}},
        )


__all__ = ["DeepgramBackend"]
