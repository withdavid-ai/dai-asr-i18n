"""Speaker activity from native word intervals, independent of transcript grouping."""

import math
from collections import defaultdict
from collections.abc import Mapping

from dai_asr_i18n.scoring.diarization import DiarizationSegment

WORD_ACTIVITY_MERGE_GAP_S = 0.200
_BOUNDARY_EPSILON_S = 1e-12


def speaker_word_activity(blob: Mapping, *, omit_reversed_words: bool = False) -> tuple[DiarizationSegment, ...]:
    """Merge only positive-duration words of the same speaker within 200 ms.

    The input's lexical segments are never modified. Words from different speakers
    can overlap. Zero-duration words create no activity and cannot bridge a gap.
    Only Gemini's explicitly selected policy permits omitting reversed intervals.
    """
    for metadata in (blob.get("provider_meta") or {}).values():
        if isinstance(metadata, Mapping) and metadata.get("word_alignment_complete") is False:
            raise ValueError("provider word alignment incomplete; lexical transcript retained")
    words = blob.get("words")
    if not isinstance(words, list):
        raise ValueError("speaker-word activity requires native words")
    if not words and any(s.get("text", "").strip() for s in blob["verbatim"]["segments"]):
        raise ValueError("nonempty speaker-word transcript has no native word timing")
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for word in words:
        speaker = word.get("speaker_external_id")
        if speaker is None or not str(speaker).strip():
            raise ValueError("native activity word has no speaker label")
        start, end = word.get("start_ms"), word.get("end_ms")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)):
            raise ValueError("native activity word requires finite timestamps")
        if start < 0 or end < 0:
            raise ValueError("native activity word requires nonnegative timestamps")
        if end < start:
            if omit_reversed_words:
                continue
            raise ValueError("native activity word has reversed timestamps")
        if end > start:
            grouped[str(speaker)].append((start / 1000.0, end / 1000.0))
    result = []
    for speaker, intervals in sorted(grouped.items()):
        merged: list[list[float]] = []
        for start, end in sorted(intervals):
            if merged and start - merged[-1][1] <= WORD_ACTIVITY_MERGE_GAP_S + _BOUNDARY_EPSILON_S:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        result.extend(DiarizationSegment(speaker, start, end) for start, end in merged)
    return tuple(sorted(result, key=lambda s: (s.start_s, s.end_s, s.speaker)))
