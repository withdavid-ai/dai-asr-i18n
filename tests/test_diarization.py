"""Exact-interval DER, JER, and speaker-count methodology."""

from __future__ import annotations

import json
import random
from itertools import permutations

import pytest

from dai_asr_i18n import (
    DER_POLICY_VERSION,
    JER_POLICY_VERSION,
    SPEAKER_COUNT_POLICY_VERSION,
    DiarizationSegment,
    der,
    diarization_error_rate,
    jer,
    speaker_count,
)


def test_der_decomposes_miss_false_alarm_and_confusion_exactly():
    reference = [("A", 0.0, 1.0), ("B", 1.0, 2.0)]
    hypothesis = [("X", 0.0, 1.5)]
    score = der(reference, hypothesis)
    assert score.policy == DER_POLICY_VERSION == "dai-asr-i18n-der-v8"
    assert score.assignment_objective == "maximum_correct_speaker_time"
    assert score.same_speaker_support == "union_touching_and_overlapping"
    assert score.zero_reference_policy == "undefined_rate_retain_primitives"
    assert score.missed_s == pytest.approx(0.5)
    assert score.false_alarm_s == 0
    assert score.confusion_s == pytest.approx(0.5)
    assert score.denominator == pytest.approx(2.0)
    assert score.value == pytest.approx(0.5)
    assert score.speaker_assignment == (("A", "X"),)


def test_der_includes_overlap_and_optimally_maps_speakers():
    reference = [("A", 0.0, 1.0), ("B", 0.0, 1.0)]
    hypothesis = [("Y", 0.0, 1.0), ("X", 0.0, 1.0)]
    score = diarization_error_rate(reference, hypothesis)
    assert score.value == 0
    assert score.denominator == 2
    assert set(score.speaker_assignment) == {("A", "X"), ("B", "Y")}


def test_extra_active_hypothesis_speaker_is_false_alarm():
    score = der([("A", 0.0, 1.0)], [("X", 0.0, 1.0), ("Y", 0.0, 1.0)])
    assert score.false_alarm_s == 1
    assert score.missed_s == score.confusion_s == 0
    assert score.value == 1


def test_reference_collar_removes_errors_and_denominator():
    reference = [("A", 0.0, 5.0)]
    hypothesis = [("X", 0.2, 5.0)]
    strict = der(reference, hypothesis)
    collared = der(reference, hypothesis, collar_s=0.25)
    assert strict.missed_s == pytest.approx(0.2)
    assert strict.denominator == 5
    assert collared.value == 0
    assert collared.denominator == pytest.approx(4.5)


def test_touching_same_speaker_turns_do_not_manufacture_collar_boundaries():
    split = [("A", 0.0, 0.5), ("A", 0.5, 1.0)]
    one = [("A", 0.0, 1.0)]
    assert der(split, [], collar_s=0.25) == der(one, [], collar_s=0.25)
    assert der(split, [], collar_s=0.25).denominator == pytest.approx(0.5)


def test_evaluation_duration_clips_all_diarization_companions():
    reference = [("A", 0.0, 1.0)]
    hypothesis = [("X", 0.0, 1.0), ("outside", 1.1, 1.2)]
    assert der(reference, hypothesis, evaluation_duration_s=1).value == 0
    assert jer(reference, hypothesis, evaluation_duration_s=1).value == 0
    count = speaker_count(reference, hypothesis, evaluation_duration_s=1)
    assert count.correct
    assert count.reference_speakers == count.hypothesis_speakers == 1


def test_zero_reference_rate_is_json_null_but_false_alarm_is_retained():
    score = der([], [("X", 0.0, 1.0)])
    assert score.value is None
    assert score.numerator == score.false_alarm_s == 1
    assert score.denominator == 0
    assert json.loads(json.dumps(score.to_dict(), allow_nan=False))["value"] is None


def test_jer_is_speaker_balanced_and_uses_jaccard_assignment():
    balanced = jer([("long", 0.0, 99.0), ("short", 99.0, 100.0)], [("H", 0.0, 99.0)])
    partial = jer([("A", 0.0, 2.0)], [("H", 1.0, 3.0)])
    assert balanced.policy == JER_POLICY_VERSION == "dai-asr-i18n-jer-v2"
    assert balanced.assignment_objective == "minimum_jaccard_error"
    assert balanced.collar_s == 0
    assert balanced.overlap == "include"
    assert (balanced.error_sum, balanced.reference_speakers, balanced.value) == (1, 2, 0.5)
    assert partial.value == pytest.approx(2 / 3)


def test_jer_does_not_separately_penalize_extra_hypothesis_speakers():
    score = jer(
        [("A", 0.0, 1.0), ("B", 0.0, 1.0)],
        [("X", 0.0, 1.0), ("Y", 0.0, 1.0), ("extra", 0.0, 10.0)],
    )
    assert score.value == 0
    assert len(score.speaker_assignment) == 2


def test_speaker_count_uses_active_timeline_labels():
    score = speaker_count(
        [("A", 0.0, 1.0), ("B", 1.0, 2.0)],
        [("X", 0.0, 2.0), ("unused", 2.0, 2.0)],
    )
    assert score.policy == SPEAKER_COUNT_POLICY_VERSION == "dai-asr-i18n-speaker-count-v3"
    assert not score.correct
    assert score.accuracy == 0
    assert score.absolute_error == 1


@pytest.mark.parametrize(
    "segment",
    [
        ("", 0.0, 1.0),
        ("A", 2.0, 1.0),
        ("A", 0.0, float("nan")),
        ("A", True, 1.0),
        ("A", -0.1, 1.0),
    ],
)
def test_invalid_diarization_segments_are_rejected(segment):
    with pytest.raises((TypeError, ValueError)):
        der([segment], [])


@pytest.mark.parametrize("collar", [-1, float("inf"), True])
def test_invalid_der_collars_are_rejected(collar):
    with pytest.raises(ValueError, match="collar"):
        der([], [], collar_s=collar)


def _independent_canonical(timeline):
    result = []
    for speaker in sorted({speaker for speaker, _, _ in timeline}):
        merged = []
        for start, end in sorted((start, end) for label, start, end in timeline if label == speaker):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        result.extend((speaker, start, end) for start, end in merged)
    return result


def _independent_der(reference, hypothesis, collar_s=0.0):
    """Small exhaustive implementation using a different assignment strategy."""

    reference = _independent_canonical(reference)
    hypothesis = _independent_canonical(hypothesis)
    collars = [
        (boundary - collar_s, boundary + collar_s)
        for _, start, end in reference
        for boundary in (start, end)
        if collar_s
    ]
    cuts = sorted(
        {value for _, start, end in reference + hypothesis for value in (start, end)}
        | {value for start, end in collars for value in (start, end)}
    )
    reference_speakers = sorted({speaker for speaker, _, _ in reference})
    hypothesis_speakers = sorted({speaker for speaker, _, _ in hypothesis})
    padded_hypothesis = hypothesis_speakers + [None] * max(0, len(reference_speakers) - len(hypothesis_speakers))
    assignments = permutations(padded_hypothesis, len(reference_speakers)) if reference_speakers else [()]
    best = None
    for permutation in assignments:
        hypothesis_to_reference = {
            hypothesis_speaker: reference_speaker
            for reference_speaker, hypothesis_speaker in zip(reference_speakers, permutation, strict=True)
            if hypothesis_speaker is not None
        }
        errors = total = 0.0
        for start, end in zip(cuts, cuts[1:], strict=False):
            midpoint = (start + end) / 2
            if any(collar_start <= midpoint < collar_end for collar_start, collar_end in collars):
                continue
            active_reference = {
                speaker for speaker, segment_start, segment_end in reference if segment_start <= midpoint < segment_end
            }
            active_hypothesis = {
                speaker for speaker, segment_start, segment_end in hypothesis if segment_start <= midpoint < segment_end
            }
            mapped_hypothesis = active_hypothesis & hypothesis_to_reference.keys()
            mapped = {hypothesis_to_reference[speaker] for speaker in mapped_hypothesis}
            reference_count = len(active_reference)
            hypothesis_count = len(active_hypothesis)
            duration = end - start
            total += duration * reference_count
            errors += duration * (
                max(0, reference_count - hypothesis_count)
                + max(0, hypothesis_count - reference_count)
                + min(reference_count, hypothesis_count)
                - len(active_reference & mapped)
            )
        candidate = errors / total if total else None
        best = candidate if best is None or (candidate is not None and candidate < best) else best
    return best


@pytest.mark.parametrize("collar_s", [0.0, 0.25])
def test_der_matches_independent_exhaustive_assignment_on_random_timelines(collar_s):
    rng = random.Random(20260930)
    for _ in range(100):
        reference = []
        hypothesis = []
        for timeline, prefix, speakers in ((reference, "R", 2), (hypothesis, "H", 3)):
            for _ in range(6):
                start = rng.uniform(0, 10)
                timeline.append((f"{prefix}{rng.randrange(speakers)}", start, start + rng.uniform(0.01, 2)))
        assert der(reference, hypothesis, collar_s=collar_s).value == pytest.approx(
            _independent_der(reference, hypothesis, collar_s), abs=1e-12
        )


def test_diarization_segment_objects_are_supported():
    segment = DiarizationSegment("A", 0.0, 1.0)
    assert der([segment], [("X", 0.0, 1.0)]).value == 0


def test_inference_json_segment_mappings_are_supported():
    reference = [{"speaker": "A", "start_s": 0.0, "end_s": 1.0}]
    hypothesis = [{"speaker": "X", "start_s": 0.0, "end_s": 1.0, "text": "hello"}]

    assert der(reference, hypothesis).value == 0

    with pytest.raises(ValueError, match="missing fields: end_s"):
        der([{"speaker": "A", "start_s": 0.0}], hypothesis)
