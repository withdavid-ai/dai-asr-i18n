"""Exercise the upload codec's real clock without contacting a provider."""

import hashlib
import io
import json
import shutil
import wave
from dataclasses import replace

import pytest
from test_inference import FakeBackend, _dataset

from dai_asr_i18n.inference import load_run_config
from dai_asr_i18n.inference.backends.openai import (
    AUDIO_ENCODING_PROTOCOL,
    OpenAITranscriptionBackend,
    _decoded_duration,
    _mp3,
)
from dai_asr_i18n.inference.runner import EFFECTIVE_CONFIG_FILENAME, run_inference


@pytest.mark.parametrize(
    "language,expected",
    [("zh-CN", "zh-cn"), ("zh", "zh"), ("en-US", "en"), ("es-419", "es"), ("ja-JP", "ja")],
)
def test_gpt_transcribe_uses_plural_language_hint(tmp_path, monkeypatch, language, expected):
    from types import SimpleNamespace

    import httpx
    from test_hosted_inference import Response

    captured = []

    def post(url, **kwargs):
        captured.append(httpx.Request("POST", url, **{k: kwargs[k] for k in ("data", "files")}))
        return Response({"text": "hello"})

    module = "dai_asr_i18n.inference.backends.openai"
    monkeypatch.setattr(module + ".require_httpx", lambda: SimpleNamespace(post=post))
    monkeypatch.setattr("dai_asr_i18n.inference.backends.hosted.require_httpx", lambda: SimpleNamespace())
    monkeypatch.setattr(module + "._mp3", lambda *_: b"mp3")
    model = replace(
        load_run_config("openai-gpt-4o-transcribe-diarize").model,
        model_id="gpt-transcribe",
        diarizes=False,
        options={"response_format": "json"},
    )
    path = tmp_path / "source.wav"
    path.write_bytes(b"audio")
    output = OpenAITranscriptionBackend(model, credentials={"OPENAI_API_KEY": "test"}).transcribe(
        path, language=language
    )
    request = output.metadata["request"]
    assert request["parameters"]["languages[]"] == [expected]
    assert "language" not in request["parameters"]
    wire = captured[0].read()
    assert b'name="languages[]"' in wire
    assert b'name="language"' not in wire
    assert ("\r\n\r\n" + expected + "\r\n").encode() in wire
    assert request["input"]["source_sha256"] == hashlib.sha256(b"audio").hexdigest()
    assert request["input"]["submitted_sha256"] == hashlib.sha256(b"mp3").hexdigest()


def test_run_passes_explicit_locale_without_changing_scoring_language(tmp_path):
    from dai_asr_i18n.datasets.models import SELECTION_FILENAME

    dataset = _dataset(tmp_path / "data")
    path = dataset / SELECTION_FILENAME
    row = json.loads(path.read_text())
    row["language"] = "zh"
    row["metadata"]["locale"] = "zh-CN"
    path.write_text(json.dumps(row) + "\n")
    base = load_run_config("openai-gpt-4o-transcribe-diarize")
    config = replace(base, channels=("ch1",), model=replace(base.model, model_id="gpt-transcribe", diarizes=False))
    backend = FakeBackend()
    output = tmp_path / "run"
    run_inference(dataset, config, output, backend=backend)
    assert backend.calls == [("ch1.wav", "zh-cn")]
    assert json.loads((output / "hypotheses.jsonl").read_text())["language"] == "zh"
    row["metadata"]["locale"] = "ja-JP"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="locale conflicts"):
        run_inference(dataset, config, tmp_path / "bad-run", backend=backend)
    assert len(backend.calls) == 1


def test_seekable_mp3_preserves_synthetic_signal_clock(tmp_path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import correlate, correlation_lags

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for the OpenAI file-transcription backend")
    rate = 16000
    source = np.zeros(rate * 3, dtype=np.int16)
    source[rate : rate + 4000] = np.random.default_rng(28).integers(-10000, 10000, size=4000, dtype=np.int16)
    path = tmp_path / "source.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(source.tobytes())
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    submitted = _mp3(path, "128k")
    decoded, decoded_rate = sf.read(io.BytesIO(submitted), dtype="int16")
    assert decoded_rate == rate
    assert len(decoded) == len(source)
    lag = correlation_lags(len(decoded), len(source))[
        np.argmax(correlate(decoded.astype(float), source.astype(float), method="fft"))
    ]
    assert lag == 0
    assert _decoded_duration(submitted) == 3.0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_encoding_identity_prevents_resuming_an_old_upload_protocol(tmp_path):
    model = load_run_config("openai-gpt-4o-transcribe-diarize").model
    identity = OpenAITranscriptionBackend(model).identity
    assert identity["audio_encoding"]["protocol"] == AUDIO_ENCODING_PROTOCOL
    assert identity["audio_encoding"]["gapless_metadata"] is True

    class Backend(FakeBackend):
        @property
        def identity(self):
            return identity

    dataset = _dataset(tmp_path / "data")
    config = replace(load_run_config("whisper-large-v3-portable"), channels=("ch1",))
    output = tmp_path / "run"
    backend = Backend()
    run_inference(dataset, config, output, backend=backend)
    spec_path = output / EFFECTIVE_CONFIG_FILENAME
    spec = json.loads(spec_path.read_text())
    spec["backend"].pop("audio_encoding")
    spec_path.write_text(json.dumps(spec))
    calls = len(backend.calls)
    with pytest.raises(ValueError, match="different dataset selection or run config"):
        run_inference(dataset, config, output, backend=backend)
    assert len(backend.calls) == calls
