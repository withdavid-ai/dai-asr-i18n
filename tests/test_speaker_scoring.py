"""Speaker-attributed cpWER/cpCER/cpSER scoring."""

from __future__ import annotations

import json

import pytest

from dai_asr_i18n import cpcer, cpser, cpwer, score_speakers


def test_cpwer_recovers_a_perfect_speaker_permutation():
    reference = {"A": "hello world", "B": "good morning"}
    hypothesis = {"X": "good morning", "Y": "hello world"}
    score = cpwer(reference, hypothesis, language="en")
    assert score.value == 0
    assert score.metric == score.source_metric == "cpwer"
    assert score.unit == "word"
    assert score.scorer.startswith("scipy-") and score.scorer.endswith("-linear_sum_assignment")
    assert score.counts.denominator == 4
    assert set(score.speaker_assignment) == {("A", "Y"), ("B", "X")}
    assert json.dumps(score.to_dict(), allow_nan=False)


def test_cpwer_charges_surplus_and_missing_speakers():
    reference = {"A": "red green", "B": "blue yellow"}
    surplus = cpwer(reference, {"X": "extra", "Y": "blue yellow", "Z": "red green"}, language="en")
    missing = cpwer(reference, {"Z": "red green"}, language="en")
    assert (surplus.counts.denominator, surplus.counts.insertions) == (4, 1)
    assert (missing.counts.denominator, missing.counts.deletions) == (4, 2)
    assert any(reference_speaker is None for reference_speaker, _ in surplus.speaker_assignment)
    assert any(hypothesis_speaker is None for _, hypothesis_speaker in missing.speaker_assignment)


@pytest.mark.parametrize("language", ["ja", "ko", "th", "zh"])
def test_automatic_cjkt_score_is_cpcer(language):
    score = score_speakers(
        {"1": "你好世界", "2": "再见"},
        {"A": "再见", "B": "你好世届"},
        language=language,
    )
    assert score.metric == score.source_metric == "cpcer"
    assert score.unit == "character"
    assert (score.counts.numerator, score.counts.denominator) == (1, 6)


def test_cpcer_uses_the_same_nonwhitespace_code_points_as_cer():
    score = cpcer({"A": "한 국 어"}, {"B": "한국어"}, language="ko")
    assert (score.counts.numerator, score.counts.denominator) == (0, 3)


def test_vietnamese_auto_score_is_public_cpwer_with_cpser_provenance():
    score = score_speakers(
        {"1": "xin chào", "2": "bạn"},
        {"A": "ban", "B": "xin chào"},
        language="vi-VN",
    )
    assert score.metric == "cpwer"
    assert score.source_metric == "cpser"
    assert score.unit == "syllable"
    assert (score.counts.numerator, score.counts.denominator) == (1, 3)
    explicit = cpser({"1": "xin chào"}, {"A": "xin chao"})
    assert explicit.metric == explicit.source_metric == "cpser"


def test_speaker_scoring_applies_the_selected_normalizer_to_every_stream():
    score = cpcer({"A": "學習語言"}, {"X": "学习语言"}, language="zh")
    assert score.value == 0
    assert score.reference_normalized == {"A": "学习语言"}
    assert score.normalization_policy == "dai-asr-i18n-normalization-v9"
    assert score.scoring_policy == "dai-asr-i18n-offline-scoring-v2"


def test_speaker_scoring_rejects_invalid_shapes_and_unsupported_word_units():
    with pytest.raises(ValueError, match="speaker labels"):
        cpwer({"": "hello"}, {"A": "hello"}, language="en")
    with pytest.raises(TypeError, match="must be a string"):
        cpwer({"A": 1}, {"A": "hello"}, language="en")  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="word segmenter"):
        cpwer({"A": "你好"}, {"B": "你好"}, language="zh")


def test_empty_speaker_maps_are_a_perfect_empty_score():
    score = cpwer({}, {}, language="en")
    assert score.value == 0
    assert score.speaker_assignment == ()
