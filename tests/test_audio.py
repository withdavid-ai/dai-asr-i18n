"""Deterministic reconstruction of the mixed-speaker inference input."""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pytest
import soundfile as sf

from dai_asr_i18n import (
    MONO_MIX_POLICY_ID,
    MONO_SAMPLE_RATE,
    build_mono_mix,
    mix_mono_arrays,
    mono_mix_manifest,
)
from dai_asr_i18n.cli import main


def _wav_bytes(samples, sample_rate: int) -> bytes:
    output = io.BytesIO()
    sf.write(output, np.asarray(samples, dtype=np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _float_wav_bytes(samples, sample_rate: int) -> bytes:
    output = io.BytesIO()
    sf.write(output, np.asarray(samples, dtype=np.float32), sample_rate, format="WAV", subtype="FLOAT")
    return output.getvalue()


def test_mix_mono_arrays_pads_and_sums_without_averaging():
    result = mix_mono_arrays([0.25, -0.25], [0.5])
    np.testing.assert_array_equal(result, np.asarray([0.75, -0.25], dtype=np.float32))


def test_mix_mono_arrays_peak_normalizes_only_when_needed():
    result = mix_mono_arrays([0.75, -0.5], [0.75, -0.25])
    np.testing.assert_allclose(result, [1.0, -0.5], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(
        mix_mono_arrays([0.25, -0.25], [0.25, -0.25]),
        np.asarray([0.5, -0.5], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("channel", "message"),
    [
        ([[0.25, -0.25]], "one-dimensional mono array"),
        ([np.nan], "non-finite samples"),
        ([np.inf], "non-finite samples"),
    ],
)
def test_mix_mono_arrays_rejects_invalid_mono_samples(channel, message):
    with pytest.raises(ValueError, match=message):
        mix_mono_arrays(channel, [0.0])


def test_build_mono_mix_rejects_non_finite_decoded_audio():
    channel = _float_wav_bytes([0.25, np.nan], MONO_SAMPLE_RATE)

    with pytest.raises(ValueError, match="channel1 contains non-finite samples"):
        build_mono_mix(channel, _wav_bytes([0.0, 0.0], MONO_SAMPLE_RATE))


def test_build_mono_mix_is_deterministic_and_pcm16():
    first = _wav_bytes([0.25, 0.0, -0.25], MONO_SAMPLE_RATE)
    second = _wav_bytes([0.25, -0.25], MONO_SAMPLE_RATE)

    one = build_mono_mix(first, second)
    two = build_mono_mix(first, second)

    assert one == two
    assert one.policy == MONO_MIX_POLICY_ID
    assert one.sample_rate == MONO_SAMPLE_RATE
    assert one.sample_count == 3
    assert one.sha256 == hashlib.sha256(one.wav_bytes).hexdigest()
    assert one.sha256 == "6d9f79602a2aa7608a98f19a82fa708a69048a14a1f2ae965e5e4f95bc73ad0e"
    with sf.SoundFile(io.BytesIO(one.wav_bytes)) as output:
        assert output.channels == 1
        assert output.samplerate == MONO_SAMPLE_RATE
        assert output.subtype == "PCM_16"


def test_build_mono_mix_resamples_each_channel_before_padding():
    result = build_mono_mix(
        _wav_bytes(np.zeros(8_000, dtype=np.float32), 8_000),
        _wav_bytes(np.zeros(32_000, dtype=np.float32), 16_000),
    )
    assert result.sample_count == 32_000
    assert result.duration_s == 2.0


def test_mono_mix_manifest_describes_the_complete_contract():
    manifest = mono_mix_manifest()
    assert manifest["policy"] == "mono-mix-v1"
    assert manifest["sample_rate"] == 16_000
    assert manifest["resampler"] == "librosa.resample default (soxr_hq)"
    assert manifest["input_validation"] == "reject non-finite decoded samples and non-mono array inputs"
    assert manifest["mix"] == "sample-wise sum without averaging or ducking"
    assert manifest["encoding"] == "mono PCM16 WAV"
    assert manifest["libsndfile"]
    assert all(manifest["packages"][name] for name in ("numpy", "soundfile", "librosa", "soxr"))


def test_mix_mono_cli_writes_only_after_checksum_validation(tmp_path, capsys):
    channel1 = tmp_path / "speaker-1.wav"
    channel2 = tmp_path / "speaker-2.wav"
    output = tmp_path / "derived" / "mono.wav"
    channel1.write_bytes(_wav_bytes([0.25, 0.0], MONO_SAMPLE_RATE))
    channel2.write_bytes(_wav_bytes([0.25, -0.25], MONO_SAMPLE_RATE))
    expected = build_mono_mix(channel1, channel2)

    assert (
        main(
            [
                "mix-mono",
                "--channel-1",
                str(channel1),
                "--channel-2",
                str(channel2),
                "--output",
                str(output),
                "--expected-sha256",
                expected.sha256.upper(),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert output.read_bytes() == expected.wav_bytes
    assert payload["sha256"] == expected.sha256
    assert payload["manifest"]["policy"] == "mono-mix-v1"

    output.unlink()
    with pytest.raises(SystemExit) as error:
        main(
            [
                "mix-mono",
                "--channel-1",
                str(channel1),
                "--channel-2",
                str(channel2),
                "--output",
                str(output),
                "--expected-sha256",
                "0" * 64,
            ]
        )
    assert error.value.code == 2
    assert "checksum mismatch" in capsys.readouterr().err
    assert not output.exists()
