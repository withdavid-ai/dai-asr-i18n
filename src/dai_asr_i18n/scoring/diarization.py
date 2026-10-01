"""Exact-interval diarization metrics aligned with the production evaluation methodology."""

from __future__ import annotations

import importlib.metadata
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from dai_asr_i18n.policies import DER_POLICY_VERSION, JER_POLICY_VERSION, SPEAKER_COUNT_POLICY_VERSION


@dataclass(frozen=True, slots=True)
class DiarizationSegment:
    """One half-open speaker-activity interval, measured in seconds."""

    speaker: str
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.speaker, str) or not self.speaker.strip():
            raise ValueError("diarization segment requires a nonempty speaker label")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in (self.start_s, self.end_s)
            )
            or self.start_s < 0
            or self.end_s < self.start_s
        ):
            raise ValueError("diarization segment requires finite, nonnegative, ordered timestamps")


SegmentInput = DiarizationSegment | tuple[str, float, float] | Mapping[str, object]
SpeakerAssignment = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class DiarizationScore:
    """NIST-style DER and its exact speaker-time sufficient statistics."""

    policy: str
    scorer: str
    assignment_objective: str
    collar_s: float
    overlap: str
    same_speaker_support: str
    zero_reference_policy: str
    evaluation_duration_s: float | None
    reference_timeline: tuple[DiarizationSegment, ...]
    hypothesis_timeline: tuple[DiarizationSegment, ...]
    speaker_assignment: SpeakerAssignment
    missed_s: float
    false_alarm_s: float
    confusion_s: float
    reference_speaker_time_s: float

    @property
    def numerator(self) -> float:
        return self.missed_s + self.false_alarm_s + self.confusion_s

    @property
    def denominator(self) -> float:
        return self.reference_speaker_time_s

    @property
    def value(self) -> float | None:
        """Return DER, or ``None`` when reference speaker-time is zero."""

        return self.numerator / self.denominator if self.denominator else None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["speaker_assignment"] = [list(pair) for pair in self.speaker_assignment]
        value["numerator"] = self.numerator
        value["denominator"] = self.denominator
        value["value"] = self.value
        return value


@dataclass(frozen=True, slots=True)
class JaccardErrorScore:
    """Speaker-balanced JER under minimum-Jaccard-error assignment."""

    policy: str
    scorer: str
    assignment_objective: str
    collar_s: float
    overlap: str
    same_speaker_support: str
    evaluation_duration_s: float | None
    reference_timeline: tuple[DiarizationSegment, ...]
    hypothesis_timeline: tuple[DiarizationSegment, ...]
    speaker_assignment: SpeakerAssignment
    error_sum: float
    reference_speakers: int

    @property
    def numerator(self) -> float:
        return self.error_sum

    @property
    def denominator(self) -> int:
        return self.reference_speakers

    @property
    def value(self) -> float | None:
        """Return JER, or ``None`` when the reference has no active speakers."""

        return self.error_sum / self.reference_speakers if self.reference_speakers else None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["speaker_assignment"] = [list(pair) for pair in self.speaker_assignment]
        value["numerator"] = self.numerator
        value["denominator"] = self.denominator
        value["value"] = self.value
        return value


@dataclass(frozen=True, slots=True)
class SpeakerCountScore:
    """Per-recording speaker-count accuracy and absolute error."""

    policy: str
    evaluation_duration_s: float | None
    reference_speakers: int
    hypothesis_speakers: int

    @property
    def correct(self) -> bool:
        return self.reference_speakers == self.hypothesis_speakers

    @property
    def accuracy(self) -> float:
        return float(self.correct)

    @property
    def absolute_error(self) -> int:
        return abs(self.hypothesis_speakers - self.reference_speakers)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "correct": self.correct,
            "accuracy": self.accuracy,
            "absolute_error": self.absolute_error,
        }


def _coerce_segments(value: Sequence[SegmentInput], side: str) -> tuple[DiarizationSegment, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{side} must be a sequence of diarization segments")
    result: list[DiarizationSegment] = []
    for segment in value:
        if isinstance(segment, DiarizationSegment):
            result.append(segment)
        elif isinstance(segment, tuple) and len(segment) == 3:
            result.append(DiarizationSegment(*segment))
        elif isinstance(segment, Mapping):
            missing = [field for field in ("speaker", "start_s", "end_s") if field not in segment]
            if missing:
                raise ValueError(f"{side} diarization segment is missing fields: {', '.join(missing)}")
            result.append(DiarizationSegment(segment["speaker"], segment["start_s"], segment["end_s"]))
        else:
            raise TypeError(f"{side} must contain DiarizationSegment values, three-item tuples, or segment mappings")
    return tuple(result)


def _duration(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("evaluation duration must be finite and positive")
    return float(value)


def _canonical_speech(segments: Sequence[DiarizationSegment]) -> tuple[DiarizationSegment, ...]:
    """Union touching or overlapping support per speaker without filling silence gaps."""

    by_speaker: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for segment in segments:
        if segment.end_s > segment.start_s:
            by_speaker[segment.speaker].append((float(segment.start_s), float(segment.end_s)))
    result: list[DiarizationSegment] = []
    for speaker, spans in sorted(by_speaker.items()):
        merged: list[tuple[float, float]] = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        result.extend(DiarizationSegment(speaker, start, end) for start, end in merged)
    return tuple(result)


def _prepare_timeline(
    value: Sequence[SegmentInput],
    side: str,
    evaluation_duration_s: float | None,
) -> tuple[DiarizationSegment, ...]:
    segments = _coerce_segments(value, side)
    if evaluation_duration_s is not None:
        segments = tuple(
            DiarizationSegment(segment.speaker, max(0.0, segment.start_s), min(evaluation_duration_s, segment.end_s))
            for segment in segments
            if min(evaluation_duration_s, segment.end_s) > max(0.0, segment.start_s)
        )
    return _canonical_speech(segments)


def _speaker_intervals(
    reference: Sequence[DiarizationSegment],
    hypothesis: Sequence[DiarizationSegment],
    *,
    collar_s: float = 0.0,
):
    """Sweep exact half-open intervals with constant active-speaker sets."""

    events: dict[float, list[tuple[int, str | None, int]]] = defaultdict(list)
    for side, timeline in enumerate((reference, hypothesis)):
        for segment in timeline:
            events[segment.start_s].append((side, segment.speaker, 1))
            events[segment.end_s].append((side, segment.speaker, -1))
    if collar_s:
        for segment in reference:
            for boundary in (segment.start_s, segment.end_s):
                events[boundary - collar_s].append((2, None, 1))
                events[boundary + collar_s].append((2, None, -1))
    active: list[Counter[str | None]] = [Counter(), Counter(), Counter()]
    previous: float | None = None
    for boundary in sorted(events):
        if previous is not None and boundary > previous and not active[2]:
            yield boundary - previous, frozenset(active[0]), frozenset(active[1])
        for side, speaker, delta in events[boundary]:
            active[side][speaker] += delta
            if active[side][speaker] == 0:
                del active[side][speaker]
        previous = boundary


def _linear_assignment(matrix: Sequence[Sequence[float]], *, maximize: bool) -> tuple[tuple[int, int], str]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - exercised by installs without the extra
        raise RuntimeError('diarization scoring requires: pip install "dai-asr-i18n[speaker]"') from error
    scorer = f"scipy-{importlib.metadata.version('scipy')}-linear_sum_assignment"
    if not matrix or not matrix[0]:
        return (), scorer
    rows, columns = linear_sum_assignment(matrix, maximize=maximize)
    return tuple((int(row), int(column)) for row, column in zip(rows, columns, strict=True)), scorer


def diarization_error_rate(
    reference: Sequence[SegmentInput],
    hypothesis: Sequence[SegmentInput],
    *,
    collar_s: float = 0.0,
    evaluation_duration_s: float | None = None,
) -> DiarizationScore:
    """Score exact-interval NIST DER with overlap included.

    The collar is a no-score half-width around canonical reference boundaries. It removes both
    errors and reference speaker-time from the result. Hypothesis boundaries never create collar
    regions. The returned miss, false-alarm, confusion, numerator, and denominator values are all
    measured in seconds.
    """

    if (
        isinstance(collar_s, bool)
        or not isinstance(collar_s, (int, float))
        or not math.isfinite(collar_s)
        or collar_s < 0
    ):
        raise ValueError("DER collar must be finite and nonnegative")
    duration = _duration(evaluation_duration_s)
    ref = _prepare_timeline(reference, "reference", duration)
    hyp = _prepare_timeline(hypothesis, "hypothesis", duration)
    intervals = list(_speaker_intervals(ref, hyp, collar_s=float(collar_s)))
    reference_speakers = sorted({speaker for _, speakers, _ in intervals for speaker in speakers})
    hypothesis_speakers = sorted({speaker for _, _, speakers in intervals for speaker in speakers})
    reference_index = {speaker: index for index, speaker in enumerate(reference_speakers)}
    hypothesis_index = {speaker: index for index, speaker in enumerate(hypothesis_speakers)}
    overlap = [[0.0 for _ in hypothesis_speakers] for _ in reference_speakers]
    missed = false_alarm = common = total_reference = 0.0
    for interval_duration, active_reference, active_hypothesis in intervals:
        reference_count = len(active_reference)
        hypothesis_count = len(active_hypothesis)
        total_reference += interval_duration * reference_count
        missed += interval_duration * max(0, reference_count - hypothesis_count)
        false_alarm += interval_duration * max(0, hypothesis_count - reference_count)
        common += interval_duration * min(reference_count, hypothesis_count)
        for reference_speaker in active_reference:
            for hypothesis_speaker in active_hypothesis:
                overlap[reference_index[reference_speaker]][hypothesis_index[hypothesis_speaker]] += interval_duration
    indices, scorer = _linear_assignment(overlap, maximize=True)
    correctly_attributed = sum(overlap[row][column] for row, column in indices)
    confusion = max(0.0, common - correctly_attributed)
    assignment = tuple((reference_speakers[row], hypothesis_speakers[column]) for row, column in indices)
    return DiarizationScore(
        policy=DER_POLICY_VERSION,
        scorer=scorer,
        assignment_objective="maximum_correct_speaker_time",
        collar_s=float(collar_s),
        overlap="include",
        same_speaker_support="union_touching_and_overlapping",
        zero_reference_policy="undefined_rate_retain_primitives",
        evaluation_duration_s=duration,
        reference_timeline=ref,
        hypothesis_timeline=hyp,
        speaker_assignment=assignment,
        missed_s=missed,
        false_alarm_s=false_alarm,
        confusion_s=confusion,
        reference_speaker_time_s=total_reference,
    )


def jaccard_error_rate(
    reference: Sequence[SegmentInput],
    hypothesis: Sequence[SegmentInput],
    *,
    evaluation_duration_s: float | None = None,
) -> JaccardErrorScore:
    """Score dscore/DIHARD JER with exact intervals, zero collar, and overlap included."""

    duration = _duration(evaluation_duration_s)
    ref = _prepare_timeline(reference, "reference", duration)
    hyp = _prepare_timeline(hypothesis, "hypothesis", duration)
    reference_speakers = sorted({segment.speaker for segment in ref})
    hypothesis_speakers = sorted({segment.speaker for segment in hyp})
    reference_index = {speaker: index for index, speaker in enumerate(reference_speakers)}
    hypothesis_index = {speaker: index for index, speaker in enumerate(hypothesis_speakers)}
    reference_duration = [0.0] * len(reference_speakers)
    hypothesis_duration = [0.0] * len(hypothesis_speakers)
    overlap = [[0.0 for _ in hypothesis_speakers] for _ in reference_speakers]
    for interval_duration, active_reference, active_hypothesis in _speaker_intervals(ref, hyp):
        for reference_speaker in active_reference:
            reference_duration[reference_index[reference_speaker]] += interval_duration
        for hypothesis_speaker in active_hypothesis:
            hypothesis_duration[hypothesis_index[hypothesis_speaker]] += interval_duration
        for reference_speaker in active_reference:
            for hypothesis_speaker in active_hypothesis:
                overlap[reference_index[reference_speaker]][hypothesis_index[hypothesis_speaker]] += interval_duration
    costs = [
        [
            1.0
            - overlap[reference_index_value][hypothesis_index_value]
            / (
                reference_duration[reference_index_value]
                + hypothesis_duration[hypothesis_index_value]
                - overlap[reference_index_value][hypothesis_index_value]
            )
            for hypothesis_index_value in range(len(hypothesis_speakers))
        ]
        for reference_index_value in range(len(reference_speakers))
    ]
    indices, scorer = _linear_assignment(costs, maximize=False)
    errors = [1.0] * len(reference_speakers)
    for row, column in indices:
        errors[row] = costs[row][column]
    assignment = tuple((reference_speakers[row], hypothesis_speakers[column]) for row, column in indices)
    return JaccardErrorScore(
        policy=JER_POLICY_VERSION,
        scorer=scorer,
        assignment_objective="minimum_jaccard_error",
        collar_s=0.0,
        overlap="include",
        same_speaker_support="union_touching_and_overlapping",
        evaluation_duration_s=duration,
        reference_timeline=ref,
        hypothesis_timeline=hyp,
        speaker_assignment=assignment,
        error_sum=sum(errors),
        reference_speakers=len(reference_speakers),
    )


def speaker_count(
    reference: Sequence[SegmentInput],
    hypothesis: Sequence[SegmentInput],
    *,
    evaluation_duration_s: float | None = None,
) -> SpeakerCountScore:
    """Count active timeline speakers for exact-match accuracy and count MAE."""

    duration = _duration(evaluation_duration_s)
    ref = _prepare_timeline(reference, "reference", duration)
    hyp = _prepare_timeline(hypothesis, "hypothesis", duration)
    return SpeakerCountScore(
        policy=SPEAKER_COUNT_POLICY_VERSION,
        evaluation_duration_s=duration,
        reference_speakers=len({segment.speaker for segment in ref}),
        hypothesis_speakers=len({segment.speaker for segment in hyp}),
    )


der = diarization_error_rate
jer = jaccard_error_rate


__all__ = [
    "DiarizationScore",
    "DiarizationSegment",
    "JaccardErrorScore",
    "SegmentInput",
    "SpeakerCountScore",
    "der",
    "diarization_error_rate",
    "jer",
    "jaccard_error_rate",
    "speaker_count",
]
