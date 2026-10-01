from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from dai_asr_i18n.inference import load_run_config
from dai_asr_i18n.inference.backends.meta import MetaMuseBackend
from dai_asr_i18n.inference.backends.mistral import MistralVoxtralBackend
from dai_asr_i18n.inference.backends.qwen import Qwen3AsrBackend
from dai_asr_i18n.inference.backends.xai import XaiGrokBackend
from dai_asr_i18n.inference.windowing import planned_audio_windows


class Response:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, value: dict):
        self.value = value

    def json(self) -> dict:
        return self.value


def _audio(path: Path, duration_s: float = 0.1) -> Path:
    sf.write(path, np.zeros(round(16_000 * duration_s), dtype=np.float32), 16_000, subtype="PCM_16")
    return path


def test_meta_muse_parses_turns_and_accepts_modal_secret_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    def post(*args, **kwargs):
        captured.update(kwargs)
        return Response(
            {
                "transcript": "hello world",
                "audioDurationMs": 1000,
                "turns": [
                    {"transcript": "hello", "startMs": 0, "endMs": 400, "speaker": "A"},
                    {"transcript": "world", "startMs": 500, "endMs": 1000, "speaker": "B"},
                ],
            }
        )

    monkeypatch.setattr("dai_asr_i18n.inference.backends.meta._wav24k", lambda _: b"wav")
    monkeypatch.setattr("dai_asr_i18n.inference.backends.meta.require_httpx", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = MetaMuseBackend(
        load_run_config("meta-muse-voice-transcribe").model,
        credentials={"MUSE_KEY": "MUSE_KEY_VALUE"},
    )
    assert backend.preflight()["ok"] is True
    result = backend.transcribe(_audio(tmp_path / "audio.wav"), language="en")
    assert result.text_by_speaker() == {"A": "hello", "B": "world"}
    request = captured["files"]["request"][1]
    assert '"languageBias": ["English"]' in request
    assert "MUSE_KEY_VALUE" not in str(backend.identity)


def test_meta_muse_rejects_nonempty_turn_without_a_speaker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def post(*args, **kwargs):
        return Response(
            {
                "transcript": "hello",
                "audioDurationMs": 500,
                "turns": [{"transcript": "hello", "startMs": 0, "endMs": 500}],
            }
        )

    monkeypatch.setattr("dai_asr_i18n.inference.backends.meta._wav24k", lambda _: b"wav")
    monkeypatch.setattr("dai_asr_i18n.inference.backends.meta.require_httpx", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = MetaMuseBackend(
        load_run_config("meta-muse-voice-transcribe").model,
        credentials={"MUSE_KEY": "x"},
    )

    with pytest.raises(ValueError, match="lacks a speaker label"):
        backend.transcribe(_audio(tmp_path / "audio.wav"), language="en")


def test_mistral_voxtral_uploads_local_file_and_parses_speakers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    def post(*args, **kwargs):
        captured.update(kwargs)
        return Response(
            {
                "model": "voxtral-mini-2602",
                "text": "Hello, world!",
                "language": "en",
                "segments": [
                    {"text": "hello ", "start": 0.0, "end": 0.4, "speaker_id": "0"},
                    {"text": "world", "start": 0.5, "end": 1.0, "speaker_id": "1"},
                ],
            }
        )

    monkeypatch.setattr("dai_asr_i18n.inference.backends.mistral.require_httpx", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = MistralVoxtralBackend(
        load_run_config("mistral-voxtral-mini-transcribe-2").model,
        credentials={"MISTRAL_API_KEY": "x"},
    )
    result = backend.transcribe(_audio(tmp_path / "audio.wav"), language="en")
    assert result.text_by_speaker() == {"0": "hello", "1": "world"}
    assert result.metadata["segment_text_matches_top_level"] is False
    assert captured["files"][-1][0] == "file"
    assert not any(name == "language" for name, _ in captured["files"])


def test_mistral_voxtral_rejects_missing_lexical_segment_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def post(*args, **kwargs):
        return Response(
            {
                "model": "voxtral-mini-2602",
                "text": "hello world",
                "language": "en",
                "segments": [{"text": "hello", "start": 0.0, "end": 0.4, "speaker_id": "0"}],
            }
        )

    monkeypatch.setattr("dai_asr_i18n.inference.backends.mistral.require_httpx", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = MistralVoxtralBackend(
        load_run_config("mistral-voxtral-mini-transcribe-2").model,
        credentials={"MISTRAL_API_KEY": "x"},
    )

    with pytest.raises(ValueError, match="do not cover"):
        backend.transcribe(_audio(tmp_path / "audio.wav"), language="en")


def test_xai_uses_current_grok_model_and_file_is_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    def post(*args, **kwargs):
        captured.update(kwargs)
        return Response(
            {
                "text": "hello world",
                "language": "en",
                "duration": 1.0,
                "words": [
                    {"text": "hello", "start": 0.0, "end": 0.4, "speaker": 0},
                    {"text": "world", "start": 0.5, "end": 1.0, "speaker": 1},
                ],
            }
        )

    monkeypatch.setattr("dai_asr_i18n.inference.backends.xai.require_httpx", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = XaiGrokBackend(
        load_run_config("xai-grok-voice-transcribe-2").model,
        credentials={"XAI_API_KEY": "x"},
    )
    result = backend.transcribe(_audio(tmp_path / "audio.wav"), language="tl")
    fields = captured["files"]
    assert fields[-1][0] == "file"
    assert dict((name, value[1]) for name, value in fields[:-1])["model"] == "grok-voice-transcribe-2.0"
    assert dict((name, value[1]) for name, value in fields[:-1])["language"] == "fil"
    assert result.text_by_speaker() == {"0": "hello", "1": "world"}


@pytest.mark.parametrize("words", [pytest.param("omitted", id="omitted"), pytest.param(None, id="null")])
def test_xai_accepts_explicit_empty_transcript_without_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, words: str | None
):
    def post(*args, **kwargs):
        payload = {"text": "", "language": "it", "duration": 1.0}
        if words != "omitted":
            payload["words"] = words
        return Response(payload)

    monkeypatch.setattr("dai_asr_i18n.inference.backends.xai.require_httpx", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    backend = XaiGrokBackend(
        load_run_config("xai-grok-voice-transcribe-2").model,
        credentials={"XAI_API_KEY": "x"},
    )

    result = backend.transcribe(_audio(tmp_path / "silence.wav"), language="it")

    assert result.text == ""
    assert result.segments == ()


def test_qwen_backend_loads_pinned_local_path_and_auto_detects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: dict = {}

    class Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.update(path=path, **kwargs)
            return cls()

        def transcribe(self, **kwargs):
            calls["transcribe"] = kwargs
            return [SimpleNamespace(text="hello", language="English")]

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "qwen_asr", SimpleNamespace(Qwen3ASRModel=Model))
    monkeypatch.setenv("DAI_ASR_I18N_QWEN_MODEL_PATH", "/models/pinned")
    monkeypatch.setattr("importlib.metadata.version", lambda _: "0.0.6")
    backend = Qwen3AsrBackend(load_run_config("qwen3-asr-1.7b").model)
    assert backend.preflight()["ok"] is True
    assert backend.identity["model_loading_policy"] == "pinned-snapshot-v1"
    assert calls == {}
    result = backend.transcribe(_audio(tmp_path / "audio.wav"), language="en")
    assert calls["path"] == "/models/pinned"
    assert calls["transcribe"]["language"] is None
    assert result.text == "hello"


def test_qwen_backend_downloads_the_configured_revision_when_no_local_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: dict = {}

    class Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.update(path=path, **kwargs)
            return cls()

        def transcribe(self, **kwargs):
            return [SimpleNamespace(text="hello", language="English")]

    def snapshot_download(model_id, *, revision):
        calls["snapshot"] = (model_id, revision)
        return "/cache/qwen-pinned"

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "qwen_asr", SimpleNamespace(Qwen3ASRModel=Model))
    monkeypatch.delenv("DAI_ASR_I18N_QWEN_MODEL_PATH", raising=False)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    config = load_run_config("qwen3-asr-1.7b").model

    result = Qwen3AsrBackend(config).transcribe(_audio(tmp_path / "audio.wav"), language="en")

    assert calls["snapshot"] == (config.model_id, config.revision)
    assert calls["path"] == "/cache/qwen-pinned"
    assert result.text == "hello"


def test_qwen_backend_rejects_an_unpinned_model_config():
    config = replace(load_run_config("qwen3-asr-1.7b").model, revision=None)

    with pytest.raises(ValueError, match="must pin model.revision"):
        Qwen3AsrBackend(config)


def test_window_planner_keeps_every_slice_under_margin(tmp_path: Path):
    path = _audio(tmp_path / "long.wav", duration_s=2.2)
    with planned_audio_windows(path, max_input_s=1.0, target_factor=0.98) as windows:
        assert len(windows) == 3
        assert all(window.path.is_file() for window in windows)
        assert all(window.end_s - window.start_s <= 0.981 for window in windows)
