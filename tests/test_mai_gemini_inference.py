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


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("missing_words", [None, []])
def test_mai_marks_missing_word_annotations_without_losing_phrase_text(tmp_path, monkeypatch, partial, missing_words):
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    phrases = [
        {"speaker": 2, "text": "world", "offsetMilliseconds": 500, "durationMilliseconds": 500, "words": missing_words}
    ]
    if partial:
        phrases.insert(
            0,
            {
                "speaker": 1,
                "text": "hello",
                "offsetMilliseconds": 0,
                "durationMilliseconds": 500,
                "words": [{"text": "hello", "offsetMilliseconds": 0, "durationMilliseconds": 500}],
            },
        )
    raw = {"phrases": phrases, "combinedPhrases": []}
    monkeypatch.setattr(
        "dai_asr_i18n.inference.backends.mai.require_httpx",
        lambda: SimpleNamespace(post=lambda *a, **kw: Response(raw)),
    )
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = MaiTranscribeBackend(
        load_run_config("mai-transcribe-2").model,
        credentials={
            "AZURE_SPEECH_KEY": "secret",
            "AZURE_SPEECH_ENDPOINT": "https://example.cognitiveservices.azure.com",
        },
    )
    transcript = backend.transcribe(audio, language="en")
    assert transcript.metadata["word_alignment_complete"] is False
    assert transcript.text_by_speaker() == ({"1": "hello", "2": "world"} if partial else {"2": "world"})
    assert transcript.text == ("hello world" if partial else "world")
    assert transcript.metadata["response"] == raw


@pytest.mark.parametrize(
    "text,words,valid",
    [(None, None, True), ("", [], True), (123, [], False), ("word", {}, False), ("word", "word", False)],
)
def test_mai_empty_and_malformed_phrases(tmp_path, monkeypatch, text, words, valid):
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    raw = {
        "phrases": [
            {"text": text, "words": words, "speaker": 1, "offsetMilliseconds": 0, "durationMilliseconds": 1000}
        ],
        "combinedPhrases": [],
    }
    monkeypatch.setattr(
        "dai_asr_i18n.inference.backends.mai.require_httpx",
        lambda: SimpleNamespace(post=lambda *a, **kw: Response(raw)),
    )
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = MaiTranscribeBackend(
        load_run_config("mai-transcribe-2").model,
        credentials={
            "AZURE_SPEECH_KEY": "secret",
            "AZURE_SPEECH_ENDPOINT": "https://example.cognitiveservices.azure.com",
        },
    )
    if not valid:
        with pytest.raises(ValueError, match="word list"):
            backend.transcribe(audio, language="en")
    else:
        transcript = backend.transcribe(audio, language="en")
        assert not transcript.text and not transcript.segments
        assert "word_alignment_complete" not in transcript.metadata
        assert transcript.metadata["response"] == raw


@pytest.mark.parametrize(
    "case,complete,expected",
    [
        ("truncated", False, {"1": "hello world"}),
        ("blank_word", False, {"1": "hello"}),
        ("empty_phrase", True, {"1": "hello"}),
        ("case", True, {"1": "Hello!"}),
        ("same_speaker_boundary", True, {"1": "hello world"}),
        ("different_speaker_boundary", False, {"1": "hello", "2": "world"}),
    ],
)
def test_mai_validates_complete_speaker_streams(tmp_path, monkeypatch, case, complete, expected):
    from copy import deepcopy

    from dai_asr_i18n.evaluation.connector import _blob
    from dai_asr_i18n.evaluation.hypotheses import hypothesis_timeline

    text, word_texts = {
        "truncated": ("hello world", ["hello"]),
        "blank_word": ("hello", [""]),
        "empty_phrase": ("", ["hello"]),
        "case": ("Hello!", ["hello"]),
    }.get(case, ("hello", ["hello world"]))
    phrases = [
        {
            "text": text,
            "speaker": 1,
            "offsetMilliseconds": 0,
            "durationMilliseconds": 1000,
            "words": [
                {"text": t, "offsetMilliseconds": i * 100, "durationMilliseconds": 100}
                for i, t in enumerate(word_texts)
            ],
        }
    ]
    if case.endswith("boundary"):
        phrases.append(
            {
                "text": "world",
                "speaker": 1 if case.startswith("same") else 2,
                "offsetMilliseconds": 1000,
                "durationMilliseconds": 1000,
                "words": [],
            }
        )
    raw = {"phrases": phrases, "combinedPhrases": []}
    original = deepcopy(raw)
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        "dai_asr_i18n.inference.backends.mai.require_httpx",
        lambda: SimpleNamespace(post=lambda *a, **kw: Response(raw)),
    )
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = MaiTranscribeBackend(
        load_run_config("mai-transcribe-2").model,
        credentials={
            "AZURE_SPEECH_KEY": "secret",
            "AZURE_SPEECH_ENDPOINT": "https://example.cognitiveservices.azure.com",
        },
    )
    transcript = backend.transcribe(audio, language="en")
    assert transcript.text_by_speaker() == expected
    assert (transcript.metadata.get("word_alignment_complete") is not False) == complete
    blob = _blob(transcript, diarizes=True, word_activity=True, check_word_coverage=True)
    if complete:
        assert hypothesis_timeline(blob)
    else:
        with pytest.raises(ValueError, match="incomplete"):
            hypothesis_timeline(blob)
    assert raw == original == transcript.metadata["response"]


@pytest.mark.parametrize(
    "left,right,complete",
    [
        ("Hello!", "hello", True),
        ("cafe\u0301", "café", True),
        ("STRASSE", "Straße", True),
        ("Ａ", "A", False),
        ("one", "1", False),
        ("a\u0301", "a", False),
        ("$1", "£1", False),
        ("漢", "汉", False),
        ("1", "2", False),
    ],
)
def test_coverage_comparison_does_not_apply_scoring_normalization(left, right, complete):
    from dai_asr_i18n.inference.word_coverage import word_text_coverage_complete

    assert word_text_coverage_complete([("A", left)], [("A", right)]) == complete
