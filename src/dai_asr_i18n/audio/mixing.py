"""Canonical reconstruction of the benchmark's mixed-speaker mono input."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import os
from dataclasses import dataclass
from numbers import Integral
from typing import Any

MONO_MIX_POLICY_ID = "mono-mix-v1"
MONO_SAMPLE_RATE = 16_000

AudioSource = str | os.PathLike[str] | bytes | bytearray


@dataclass(frozen=True, slots=True)
class MonoMixResult:
    """Canonical PCM16 WAV plus the identity needed to verify it."""

    wav_bytes: bytes
    sha256: str
    sample_rate: int
    sample_count: int
    policy: str = MONO_MIX_POLICY_ID

    @property
    def duration_s(self) -> float:
        return self.sample_count / self.sample_rate

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "policy": self.policy,
            "sha256": self.sha256,
            "sample_rate": self.sample_rate,
            "sample_count": self.sample_count,
            "duration_s": self.duration_s,
        }


def _numpy():
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - exercised by installs without the audio extra
        raise RuntimeError("mono mixing requires the 'dai-asr-i18n[audio]' extra") from error
    return np


def _soundfile():
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover - exercised by installs without the audio extra
        raise RuntimeError("mono mixing requires the 'dai-asr-i18n[audio]' extra") from error
    return sf


def _mono_float32(samples: Any, *, label: str):
    np = _numpy()
    with np.errstate(invalid="ignore", over="ignore"):
        values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional mono array")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{label} contains non-finite samples")
    return values


def _resample(samples: Any, source_rate: int, target_rate: int = MONO_SAMPLE_RATE, *, label: str = "audio"):
    np = _numpy()
    values = _mono_float32(samples, label=label)
    if isinstance(source_rate, bool) or not isinstance(source_rate, Integral) or source_rate <= 0:
        raise ValueError("source sample rate must be a positive integer")
    if source_rate == target_rate:
        return values
    try:
        import librosa
    except ImportError as error:  # pragma: no cover - exercised by installs without the audio extra
        raise RuntimeError("resampling requires the 'dai-asr-i18n[audio]' extra") from error
    return librosa.resample(values, orig_sr=source_rate, target_sr=target_rate).astype(np.float32)


def mix_mono_arrays(channel1: Any, channel2: Any):
    """Sum two synchronized mono arrays, padding and peak-normalizing exactly as production.

    Inputs must already share a sample rate. The shorter channel is zero-padded. Relative levels
    are preserved: the channels are summed, never averaged or speech-dependently ducked. The full
    mixture is scaled only when its absolute peak exceeds one. Inputs must be one-dimensional and
    contain only finite values.
    """

    np = _numpy()
    first = _mono_float32(channel1, label="channel1")
    second = _mono_float32(channel2, label="channel2")
    length = max(first.shape[0], second.shape[0])
    if first.shape[0] < length:
        first = np.pad(first, (0, length - first.shape[0]))
    if second.shape[0] < length:
        second = np.pad(second, (0, length - second.shape[0]))
    with np.errstate(invalid="ignore", over="ignore"):
        mixed = first + second
    if not bool(np.all(np.isfinite(mixed))):
        raise ValueError("mixed audio contains non-finite samples")
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def _decode_audio(source: AudioSource) -> tuple[Any, int]:
    sf = _soundfile()
    readable = io.BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray)) else source
    samples, sample_rate = sf.read(readable, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
    return mono, int(sample_rate)


def _encode_pcm16_wav(samples: Any, sample_rate: int = MONO_SAMPLE_RATE) -> bytes:
    sf = _soundfile()
    np = _numpy()
    output = io.BytesIO()
    values = np.clip(_mono_float32(samples, label="mixed audio"), -1.0, 1.0)
    sf.write(output, values, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _build_mono_mix_from_arrays(
    channel1: Any,
    channel1_sample_rate: int,
    channel2: Any,
    channel2_sample_rate: int,
) -> MonoMixResult:
    """Build the canonical mix from decoded mono arrays after independent resampling."""

    first = _resample(channel1, channel1_sample_rate, label="channel1")
    second = _resample(channel2, channel2_sample_rate, label="channel2")
    mixed = mix_mono_arrays(first, second)
    wav_bytes = _encode_pcm16_wav(mixed)
    return MonoMixResult(
        wav_bytes=wav_bytes,
        sha256=hashlib.sha256(wav_bytes).hexdigest(),
        sample_rate=MONO_SAMPLE_RATE,
        sample_count=int(mixed.shape[0]),
    )


def build_mono_mix(channel1: AudioSource, channel2: AudioSource) -> MonoMixResult:
    """Decode two released channels and return the canonical ``mono-mix-v1`` WAV."""

    first, first_rate = _decode_audio(channel1)
    second, second_rate = _decode_audio(channel2)
    return _build_mono_mix_from_arrays(first, first_rate, second, second_rate)


def mono_mix_manifest() -> dict[str, object]:
    """Return the complete versioned contract for reconstructing a mono input."""

    packages: dict[str, str | None] = {}
    for name in ("numpy", "soundfile", "librosa", "soxr"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    libsndfile_version: str | None = None
    try:
        libsndfile_version = str(_soundfile().__libsndfile_version__)
    except RuntimeError:
        pass
    return {
        "policy": MONO_MIX_POLICY_ID,
        "sample_rate": MONO_SAMPLE_RATE,
        "decode_dtype": "float32",
        "resampler": "librosa.resample default (soxr_hq)",
        "channel_downmix": "arithmetic mean for an unexpected multichannel source",
        "input_validation": "reject non-finite decoded samples and non-mono array inputs",
        "length_alignment": "zero-pad the shorter channel at the end",
        "mix": "sample-wise sum without averaging or ducking",
        "peak_policy": "divide the full mix by abs peak only when peak > 1.0",
        "encoding": "mono PCM16 WAV",
        "libsndfile": libsndfile_version,
        "packages": packages,
    }


__all__ = [
    "MONO_MIX_POLICY_ID",
    "MONO_SAMPLE_RATE",
    "MonoMixResult",
    "build_mono_mix",
    "mix_mono_arrays",
    "mono_mix_manifest",
]
