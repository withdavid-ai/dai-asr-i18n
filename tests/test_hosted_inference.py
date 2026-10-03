from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dai_asr_i18n.inference.backends.deepgram import DeepgramBackend
from dai_asr_i18n.inference.backends.elevenlabs import ElevenLabsBackend
from dai_asr_i18n.inference.backends.hosted import checked_request
from dai_asr_i18n.inference.backends.openai import OpenAITranscriptionBackend
from dai_asr_i18n.inference.config import load_run_config


class Response:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def json(self) -> dict[str, object]:
        return self.value


def test_hosted_retry_honors_retry_after_and_reopens_the_call(monkeypatch: pytest.MonkeyPatch):
    responses = [
        SimpleNamespace(status_code=429, headers={"retry-after": "7"}),
        SimpleNamespace(status_code=200, headers={}),
    ]
    sleeps: list[float] = []
    monkeypatch.setattr(
        "dai_asr_i18n.inference.backends.hosted.require_httpx",
        lambda: SimpleNamespace(TransportError=OSError),
    )
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.random.uniform", lambda *_: 0.0)
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.time.sleep", sleeps.append)

    response = checked_request(lambda: responses.pop(0))

    assert response.status_code == 200
    assert sleeps == [7.0]
    assert responses == []


def test_elevenlabs_backend_parses_speaker_words_without_exposing_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    observed: dict[str, object] = {}

    def post(*args: object, **kwargs: object) -> Response:
        observed.update({"args": args, **kwargs})
        return Response(
            {
                "text": "hello world",
                "language_code": "en",
                "language_probability": 0.99,
                "words": [
                    {"type": "word", "text": "hello", "start": 0.0, "end": 0.4, "speaker_id": "A"},
                    {"type": "word", "text": "world", "start": 0.5, "end": 0.9, "speaker_id": "B"},
                ],
            }
        )

    monkeypatch.setattr("dai_asr_i18n.inference.backends.elevenlabs.require_httpx", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = ElevenLabsBackend(
        load_run_config("elevenlabs-scribe-v2").model,
        credentials={"ELEVENLABS_API_KEY": "secret"},
    )

    transcript = backend.transcribe(audio, language="en-US")

    assert transcript.text_by_speaker() == {"A": "hello", "B": "world"}
    assert observed["headers"] == {"xi-api-key": "secret"}
    assert "secret" not in str(backend.identity)
    assert observed["data"]["model_id"] == "scribe_v2"  # type: ignore[index]


def test_elevenlabs_retry_reopens_audio_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    uploads: list[bytes] = []

    def post(*_: object, **kwargs: object) -> Response:
        uploads.append(kwargs["files"]["file"][1].read())  # type: ignore[index, union-attr]
        return Response({"text": "", "words": []})

    def call_twice(call):
        call()
        return call()

    monkeypatch.setattr("dai_asr_i18n.inference.backends.elevenlabs.require_httpx", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr("dai_asr_i18n.inference.backends.elevenlabs.checked_request", call_twice)
    backend = ElevenLabsBackend(
        load_run_config("elevenlabs-scribe-v2").model,
        credentials={"ELEVENLABS_API_KEY": "secret"},
    )

    backend.transcribe(audio, language="en")

    assert uploads == [b"audio", b"audio"]


def test_elevenlabs_rejects_nonempty_text_without_diarized_words(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        "dai_asr_i18n.inference.backends.elevenlabs.require_httpx",
        lambda: SimpleNamespace(post=lambda *_, **__: Response({"text": "hello", "words": []})),
    )
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = ElevenLabsBackend(
        load_run_config("elevenlabs-scribe-v2").model,
        credentials={"ELEVENLABS_API_KEY": "secret"},
    )

    with pytest.raises(ValueError, match="without diarized words"):
        backend.transcribe(audio, language="en")


def test_deepgram_backend_parses_diarized_words(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    def post(*_: object, **__: object) -> Response:
        return Response(
            {
                "results": {
                    "channels": [
                        {
                            "alternatives": [
                                {
                                    "transcript": "hello world",
                                    "words": [
                                        {"word": "hello", "start": 0.0, "end": 0.4, "speaker": 0},
                                        {"word": "world", "start": 0.5, "end": 0.9, "speaker": 1},
                                    ],
                                }
                            ]
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr("dai_asr_i18n.inference.backends.deepgram.require_httpx", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = DeepgramBackend(load_run_config("deepgram-nova-3").model, credentials={"DEEPGRAM_API_KEY": "secret"})

    transcript = backend.transcribe(audio, language="en")

    assert transcript.text_by_speaker() == {"0": "hello", "1": "world"}


def test_deepgram_rejects_words_without_required_speaker_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    def post(*_: object, **__: object) -> Response:
        return Response(
            {"results": {"channels": [{"alternatives": [{"transcript": "hello", "words": [{"word": "hello"}]}]}]}}
        )

    monkeypatch.setattr("dai_asr_i18n.inference.backends.deepgram.require_httpx", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = DeepgramBackend(load_run_config("deepgram-nova-3").model, credentials={"DEEPGRAM_API_KEY": "secret"})

    with pytest.raises(ValueError, match="without a speaker label"):
        backend.transcribe(audio, language="en")


def test_openai_backend_parses_diarized_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    def post(*_: object, **__: object) -> Response:
        return Response(
            {
                "text": "hello world",
                "segments": [
                    {"speaker": "A", "start": 0.0, "end": 0.4, "text": "hello"},
                    {"speaker": "B", "start": 0.5, "end": 0.9, "text": "world"},
                ],
            }
        )

    monkeypatch.setattr("dai_asr_i18n.inference.backends.openai.require_httpx", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    monkeypatch.setattr("dai_asr_i18n.inference.backends.openai._mp3", lambda *_: b"mp3")
    backend = OpenAITranscriptionBackend(
        load_run_config("openai-gpt-4o-transcribe-diarize").model,
        credentials={"OPENAI_API_KEY": "secret"},
    )

    transcript = backend.transcribe(audio, language="en")

    assert transcript.text_by_speaker() == {"A": "hello", "B": "world"}
    assert transcript.timed_speaker_segments()[1]["speaker"] == "B"
    assert transcript.metadata["request"]["parameters"]["language"] == "en"
    assert "languages[]" not in transcript.metadata["request"]["parameters"]


def test_openai_rejects_nonempty_diarized_text_without_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    monkeypatch.setattr(
        "dai_asr_i18n.inference.backends.openai.require_httpx",
        lambda: SimpleNamespace(post=lambda *_, **__: Response({"text": "hello", "segments": []})),
    )
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    monkeypatch.setattr("dai_asr_i18n.inference.backends.openai._mp3", lambda *_: b"mp3")
    backend = OpenAITranscriptionBackend(
        load_run_config("openai-gpt-4o-transcribe-diarize").model,
        credentials={"OPENAI_API_KEY": "secret"},
    )

    with pytest.raises(ValueError, match="without diarized segments"):
        backend.transcribe(audio, language="en")


def test_hosted_preflight_reports_missing_environment_without_secret_values(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = DeepgramBackend(load_run_config("deepgram-nova-3").model)

    report = backend.preflight()

    assert not report["ok"]
    assert report["error"] == "missing required environment variables: DEEPGRAM_API_KEY"
