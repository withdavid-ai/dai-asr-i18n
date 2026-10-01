"""Synthetic regression cases with fixed expected results, independent of the test run."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from dai_asr_i18n import (
    der,
    hypothesis_timeline,
    jer,
    normalize,
    reference_from_alignment,
    score_pair,
    score_speakers,
    speaker_count,
)

FIXTURES = json.loads((Path(__file__).parent / "fixtures/scoring-cases.json").read_text())


@pytest.mark.parametrize("case", FIXTURES["normalization"])
def test_normalization_matches_expected_results(case):
    assert normalize(case["text"], case["language"], case["profile"]) == case["expected"]


@pytest.mark.parametrize("case", FIXTURES["lexical"])
def test_lexical_counts_match_expected_results(case):
    score = score_pair(case["reference"], case["hypothesis"], language=case["language"])
    assert asdict(score.counts) == case["counts"]


@pytest.mark.parametrize("case", FIXTURES["permutation"])
def test_permutation_counts_match_expected_results(case):
    score = score_speakers(case["reference"], case["hypothesis"], language=case["language"])
    assert asdict(score.counts) == case["counts"]


@pytest.mark.parametrize("case", FIXTURES["diarization"])
def test_diarization_primitives_match_expected_results(case):
    ref = [tuple(s) for s in case["reference"]]
    hyp = [tuple(s) for s in case["hypothesis"]]
    score = der(ref, hyp, collar_s=case["collar"], evaluation_duration_s=case["duration"])
    assert [score.missed_s, score.false_alarm_s, score.confusion_s, score.denominator] == pytest.approx(
        [case["miss"], case["fa"], case["conf"], case["reference_s"]], abs=1e-12
    )
    j = jer(ref, hyp, evaluation_duration_s=case["duration"])
    assert j.error_sum == pytest.approx(case["jer_errors"], abs=1e-12)
    assert j.reference_speakers == case["jer_speakers"]
    count = speaker_count(ref, hyp, evaluation_duration_s=case["duration"])
    assert count.reference_speakers == case["count"]["reference_speakers"]
    assert count.hypothesis_speakers == case["count"]["hypothesis_speakers"]


@pytest.mark.parametrize("case", FIXTURES["references"])
def test_reference_preparation_matches_expected_results(case):
    reference = reference_from_alignment(case["document"], channel_durations_s=case["durations"])
    assert [(s.speaker, s.start_s, s.end_s) for s in reference.forced_timeline] == [tuple(s) for s in case["expected"]]


@pytest.mark.parametrize("case", FIXTURES["gemini"])
def test_gemini_timeline_matches_expected_results(case):
    result = hypothesis_timeline(case["blob"], provider="gemini_transcribe")
    assert [(s.speaker, s.start_s, s.end_s) for s in result] == [tuple(s) for s in case["expected"]]
