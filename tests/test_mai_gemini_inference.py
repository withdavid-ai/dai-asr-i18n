from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dai_asr_i18n.inference.backends.gemini import GeminiTranscribeBackend
from dai_asr_i18n.inference.backends.mai import MaiTranscribeBackend
from dai_asr_i18n.inference.config import load_run_config


class Response:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def json(self) -> dict[str, object]:
        return self.value


def test_mai_backend_parses_speaker_phrases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    def post(*_: object, **__: object) -> Response:
        return Response(
            {
                "durationMilliseconds": 1000,
                "combinedPhrases": [],
                "phrases": [
                    {
                        "speaker": 1,
                        "text": "hello world",
                        "offsetMilliseconds": 0,
                        "durationMilliseconds": 1000,
                        "confidence": 0.9,
                        "words": [
                            {"text": "hello", "offsetMilliseconds": 0, "durationMilliseconds": 400},
                            {"text": "world", "offsetMilliseconds": 500, "durationMilliseconds": 500},
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr("dai_asr_i18n.inference.backends.mai.require_httpx", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = MaiTranscribeBackend(
        load_run_config("mai-transcribe-2").model,
        credentials={
            "AZURE_SPEECH_KEY": "secret",
            "AZURE_SPEECH_ENDPOINT": "https://example.cognitiveservices.azure.com",
        },
    )

    transcript = backend.transcribe(audio, language="en-US")

    assert transcript.text == "hello world"
    assert transcript.text_by_speaker() == {"1": "hello world"}
    assert transcript.segments[0].words[0].confidence == 0.9
    assert "secret" not in str(transcript.metadata)
    assert "example.cognitiveservices.azure.com" not in str(transcript.metadata)
    assert transcript.metadata["request"]["endpoint"] == (
        "https://<azure-speech-resource>/speechtotext/transcriptions:transcribe"
    )


def test_gemini_backend_parses_speaker_words_and_cleans_up_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    deleted: list[str] = []

    class Files:
        def upload(self, **_: object):
            return SimpleNamespace(name="files/1", uri="gemini://files/1", state=SimpleNamespace(name="ACTIVE"))

        def get(self, **_: object):
            raise AssertionError("an ACTIVE upload should not be polled")

        def delete(self, *, name: str):
            deleted.append(name)

    class Client:
        files = Files()

        def close(self):
            return None

    def post(*_: object, **__: object) -> Response:
        return Response(
            {
                "status": "completed",
                "output_text": "hello world",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "text",
                                "text": "hello world",
                                "annotations": [
                                    {
                                        "type": "word_info",
                                        "text": "hello",
                                        "start_offset": "0s",
                                        "end_offset": "0.4s",
                                        "speaker": "A",
                                    },
                                    {
                                        "type": "word_info",
                                        "text": "world",
                                        "start_offset": "0.5s",
                                        "end_offset": "0.9s",
                                        "speaker": "B",
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr(
        "dai_asr_i18n.inference.backends.gemini._google_genai", lambda: SimpleNamespace(Client=lambda **_: Client())
    )
    monkeypatch.setattr("dai_asr_i18n.inference.backends.gemini.require_httpx", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = GeminiTranscribeBackend(
        load_run_config("gemini-transcribe-3.5").model,
        credentials={"GEMINI_API_KEY": "secret"},
    )

    transcript = backend.transcribe(audio, language="en")

    assert transcript.text_by_speaker() == {"A": "hello", "B": "world"}
    assert deleted == ["files/1"]
    assert "secret" not in str(transcript.metadata)
