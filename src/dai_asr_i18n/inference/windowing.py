"""Audio-only window planning for models with hard input-duration limits."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from dai_asr_i18n.inference.base import Transcript, TranscriptSegment, TranscriptWord


@dataclass(frozen=True)
class AudioWindow:
    path: Path
    start_s: float
    end_s: float
    index: int


def _require_audio():
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as error:
        raise ValueError('windowed inference requires: pip install "dai-asr-i18n[audio]"') from error
    return np, sf


def _silence_gaps(samples, sample_rate: int) -> list[tuple[float, float]]:
    """Find conservative silence gaps using the production energy-detector constants."""

    np, _ = _require_audio()
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    total_s = len(x) / float(sample_rate)
    frame = max(1, int(sample_rate * 0.030))
    hop = max(1, int(sample_rate * 0.010))
    if len(x) < frame:
        return [(0.0, total_s)] if total_s else []

    starts = hop * np.arange(max(1, (len(x) - frame) // hop + 1))
    ends = starts + frame
    squared = np.concatenate([[0.0], np.cumsum(x * x)])
    rms = np.sqrt((squared[ends] - squared[starts]) / frame)
    db = 20.0 * np.log10(np.maximum(rms, 3.16e-5))
    active = db > (max(float(np.percentile(db, 10.0)), -90.0) + 18.0)
    frame_times = (starts + frame / 2.0) / float(sample_rate)

    regions: list[list[float]] = []
    start_s: float | None = None
    for index, is_active in enumerate(active):
        if is_active and start_s is None:
            start_s = max(0.0, float(frame_times[index] - 0.005))
        elif not is_active and start_s is not None:
            end_s = min(total_s, float(frame_times[index - 1] + 0.005))
            if end_s - start_s >= 0.025:
                regions.append([max(0.0, start_s - 0.010), min(total_s, end_s + 0.010)])
            start_s = None
    if start_s is not None:
        regions.append([max(0.0, start_s - 0.010), total_s])

    # Pauses shorter than one second remain inside one speech region.
    merged: list[list[float]] = []
    for region in regions:
        if merged and region[0] <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], region[1])
        else:
            merged.append(region)
    if not merged:
        return [(0.0, total_s)] if total_s else []

    gaps: list[tuple[float, float]] = []
    if merged[0][0] > 0:
        gaps.append((0.0, merged[0][0]))
    gaps.extend((left[1], right[0]) for left, right in zip(merged, merged[1:]) if right[0] > left[1])
    if merged[-1][1] < total_s:
        gaps.append((merged[-1][1], total_s))
    return [gap for gap in gaps if gap[1] - gap[0] >= 0.2]


def _plan_boundaries(samples, sample_rate: int, max_window_s: float) -> list[tuple[float, float]]:
    total_s = len(samples) / float(sample_rate)
    if total_s <= max_window_s:
        return [(0.0, total_s)]
    gaps = _silence_gaps(samples, sample_rate)
    boundaries: list[tuple[float, float]] = []
    current = 0.0
    while total_s - current > max_window_s:
        chunks_left = math.ceil((total_s - current) / max_window_s)
        target = current + (total_s - current) / chunks_left
        ceiling = current + max_window_s
        candidates = [(start + end) / 2.0 for start, end in gaps if current + 15.0 <= (start + end) / 2.0 <= ceiling]
        cut = min(candidates, key=lambda value: abs(value - target)) if candidates else ceiling
        boundaries.append((current, cut))
        current = cut
    boundaries.append((current, total_s))
    return boundaries


@contextmanager
def planned_audio_windows(
    audio_path: Path, *, max_input_s: float, target_factor: float = 0.98
) -> Iterator[list[AudioWindow]]:
    """Yield silence-aware WAV slices, keeping a margin below a model's hard limit."""

    if max_input_s <= 0 or not 0 < target_factor <= 1:
        raise ValueError("window limits must be positive and target_factor must be in (0, 1]")
    np, sf = _require_audio()
    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1, dtype=np.float32)
    boundaries = _plan_boundaries(mono, sample_rate, max_input_s * target_factor)
    if len(boundaries) == 1:
        yield [AudioWindow(audio_path, boundaries[0][0], boundaries[0][1], 0)]
        return

    with tempfile.TemporaryDirectory(prefix="dai-asr-i18n-windows-") as directory:
        windows: list[AudioWindow] = []
        for index, (start_s, end_s) in enumerate(boundaries):
            path = Path(directory) / f"window-{index:03d}.wav"
            start = int(round(start_s * sample_rate))
            end = int(round(end_s * sample_rate))
            sf.write(path, samples[start:end], sample_rate, subtype="PCM_16", format="WAV")
            windows.append(AudioWindow(path, start_s, end_s, index))
        yield windows


def stitch_transcripts(windows: list[tuple[AudioWindow, Transcript]], *, language: str | None) -> Transcript:
    """Rebase independently transcribed windows without claiming cross-window speaker identity."""

    words: list[TranscriptWord] = []
    segments: list[TranscriptSegment] = []
    texts: list[str] = []
    for window, transcript in windows:
        if transcript.text.strip():
            texts.append(transcript.text.strip())
        for segment in transcript.segments:
            segment_words = tuple(
                TranscriptWord(
                    text=word.text,
                    start_s=None if word.start_s is None else word.start_s + window.start_s,
                    end_s=None if word.end_s is None else word.end_s + window.start_s,
                    confidence=word.confidence,
                    speaker=(f"window{window.index}:{word.speaker}" if word.speaker is not None else None),
                )
                for word in segment.words
            )
            words.extend(segment_words)
            segments.append(
                TranscriptSegment(
                    text=segment.text,
                    start_s=None if segment.start_s is None else segment.start_s + window.start_s,
                    end_s=None if segment.end_s is None else segment.end_s + window.start_s,
                    words=segment_words,
                    speaker=(f"window{window.index}:{segment.speaker}" if segment.speaker is not None else None),
                )
            )
    if words and not segments:
        segments.append(TranscriptSegment(text=" ".join(texts), words=tuple(words)))
    return Transcript(
        text=" ".join(texts),
        language=language,
        duration_s=max((window.end_s for window, _ in windows), default=0.0),
        segments=tuple(segments),
        metadata={
            "windowing": "audio_energy_v2",
            "speaker_identity_scope": "independent_windows",
            "windows": [
                {
                    "index": window.index,
                    "start_s": window.start_s,
                    "end_s": window.end_s,
                    "response": transcript.metadata,
                }
                for window, transcript in windows
            ],
        },
    )


__all__ = ["AudioWindow", "planned_audio_windows", "stitch_transcripts"]
