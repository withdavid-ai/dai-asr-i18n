"""Read aligned references without changing text or timing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from dai_asr_i18n.evaluation._reference_validation import forced_speech_regions
from dai_asr_i18n.scoring.diarization import DiarizationSegment


@dataclass(frozen=True)
class BenchmarkReference:
    """Prepared transcript and speaker-activity scoring inputs."""

    language: str
    text_by_speaker: dict[str, str]
    mixed_text: str
    speaker_count_timeline: tuple[DiarizationSegment, ...]
    forced_timeline: tuple[DiarizationSegment, ...]
    evaluation_duration_s: float
    pseudo_segments: int
    merge_gap_s: float = 0.2


def reference_from_alignment(
    document: Mapping,
    *,
    channel_durations_s: Mapping[str, float],
) -> BenchmarkReference:
    """Validate the supplied reference and retain its text and word timestamps.

    Reference word gaps of at most 0.20 seconds form speech regions. Durations
    are measured from the two physical audio channels. No alignment is rerun.
    """
    durations = {}
    for ch in ("ch1", "ch2"):
        value = channel_durations_s.get(ch)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"a finite positive audio duration is required for {ch}")
        durations[ch[2:]] = float(value)
    job, language = document["jobId"], document["lang"]
    annotations = document.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("aligned reference requires annotations")
    groups = {"1": [], "2": []}
    for a in annotations:
        speaker = str(a.get("speakerId"))
        aid = a.get("annotationId", "")
        prefix = f"{job}:ch{speaker}:"
        index = aid.removeprefix(prefix)
        if speaker not in groups or not aid.startswith(prefix) or not index.isdigit():
            raise ValueError("aligned annotation identity does not match its job and physical speaker")
        if not isinstance(a.get("silverText"), str):
            raise ValueError("aligned reference segment is missing its transcript text")
        DiarizationSegment(speaker, a["audioStart"], a["audioEnd"])
        groups[speaker].append((int(index), a))
    ordered = {speaker: [a for _, a in sorted(rows, key=lambda row: row[0])] for speaker, rows in groups.items()}
    payload = {
        "jobId": job,
        "lang": language,
        "speakers": [{"sid": int(speaker), "chunks": rows} for speaker, rows in ordered.items()],
    }
    forced = forced_speech_regions(document, payload, durations)
    return BenchmarkReference(
        language=language,
        text_by_speaker={speaker: " ".join(a["silverText"] for a in rows).strip() for speaker, rows in ordered.items()},
        mixed_text=" ".join(
            a["silverText"]
            for a in sorted([a for rows in ordered.values() for a in rows], key=lambda a: a["audioStart"])
        ).strip(),
        speaker_count_timeline=tuple(
            DiarizationSegment(speaker, a["audioStart"], a["audioEnd"])
            for speaker, rows in ordered.items()
            for a in rows
        ),
        forced_timeline=tuple(DiarizationSegment(speaker, a / 1000, b / 1000) for speaker, a, b in forced),
        evaluation_duration_s=max(durations.values()),
        pseudo_segments=sum(a.get("timing_source") == "pseudo" for a in annotations),
    )
