"""Offline scoring emits transparent counts and micro-averages."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dai_asr_i18n.policies import METRIC_POLICY_VERSION, SCORING_POLICY_VERSION
from dai_asr_i18n.scoring import aggregate_record_scores, aggregate_scores, score_pair, score_records


def test_auto_metric_uses_cjkt_cer_and_otherwise_wer():
    for language in ("zh", "ja", "ko", "th"):
        assert score_pair("a", "a", language=language).metric == "cer"
    for language in ("en", "hi", "ta", "te", "ar"):
        assert score_pair("a", "a", language=language).metric == "wer"


@pytest.mark.parametrize("language", ["lo", "km", "my", "yue", "xx"])
def test_auto_metric_rejects_languages_outside_the_release_roster(language):
    with pytest.raises(ValueError, match="auto metric is unavailable"):
        score_pair("text", "text", language=language)


def test_unknown_language_can_use_an_explicit_metric():
    assert score_pair("abc", "adc", language="xx", metric="cer").metric == "cer"


def test_word_error_counts_are_exposed():
    score = score_pair("alpha beta gamma", "alpha delta extra", language="en", metric="wer")
    assert (score.counts.substitutions, score.counts.deletions, score.counts.insertions, score.counts.hits) == (
        2,
        0,
        0,
        1,
    )
    assert score.counts.numerator == 2
    assert score.counts.denominator == 3
    assert score.value == pytest.approx(2 / 3)


def test_character_error_ignores_whitespace():
    score = score_pair("你 好", "你好", language="zh", metric="cer")
    assert score.value == 0
    assert score.counts.hits == 2


def test_benchmark_and_whisper_can_be_compared_on_identical_inputs():
    benchmark = score_pair("你好（音乐）世界", "你好世界", language="zh", profile="dai_asr_i18n")
    whisper = score_pair("你好（音乐）世界", "你好世界", language="zh", profile="whisper_baseline")
    assert benchmark.value == 0
    assert whisper.value > 0


@pytest.mark.parametrize("metric", ["wer", "cer"])
@pytest.mark.parametrize(
    "hypothesis",
    [
        "مر\u061cحبا بكم",  # Arabic letter mark inside a word
        "مرحبا\u200bبكم",  # zero-width space in place of a word boundary
        "مرحبا ب\u200cكم",  # zero-width non-joiner inside a word
        "\u202bمرحبا بكم\u202c",  # right-to-left embedding and pop formatting
        "\ufeffمرحبا بكم",  # BOM / zero-width no-break space
    ],
)
def test_arabic_format_controls_do_not_change_wer_or_cer(metric, hypothesis):
    score = score_pair("مرحبا بكم", hypothesis, language="ar", metric=metric)
    assert score.counts.numerator == 0


def test_vietnamese_preserves_public_and_source_metric_names():
    score = score_pair("Việt Nam", "Việt", language="vi")
    assert score.metric == "wer"
    assert score.source_metric == "ser"
    assert score.unit == "syllable"
    assert score.counts.deletions == 1


def test_explicit_vietnamese_wer_still_records_syllable_units():
    score = score_pair("Việt Nam", "Việt", language="vi", metric="wer")
    assert score.metric == "wer"
    assert score.source_metric == "ser"
    assert score.unit == "syllable"


@pytest.mark.parametrize("language", ["zh", "ja", "th"])
def test_wer_requires_a_bundled_word_segmenter(language):
    with pytest.raises(ValueError, match="word segmenter"):
        score_pair("text", "text", language=language, metric="wer")


@pytest.mark.parametrize("language", ["en", "zh", "th"])
def test_ser_is_not_fabricated_from_non_vietnamese_whitespace(language):
    with pytest.raises(ValueError, match="Vietnamese"):
        score_pair("one two", "one", language=language, metric="ser")


def test_score_records_policy_lineage():
    score = score_pair("alpha", "alpha", language="en")
    assert score.normalization_policy == "dai-asr-i18n-normalization-v9"
    assert score.metric_policy == METRIC_POLICY_VERSION
    assert score.scoring_policy == SCORING_POLICY_VERSION


def test_empty_reference_semantics_match_the_harness():
    assert score_pair("", "", language="en").value == 0
    assert score_pair("", "hello", language="en").value == 1


def test_micro_average_sums_counts_before_division():
    scores = [
        score_pair("alpha", "wrong", language="en"),
        score_pair("alpha beta gamma", "alpha beta gamma", language="en"),
    ]
    aggregate = aggregate_scores(scores)[0]
    assert aggregate.items == 2
    assert aggregate.counts.numerator == 1
    assert aggregate.counts.denominator == 4
    assert aggregate.value == 0.25


def test_aggregation_keeps_distinct_normalizer_lineage_separate():
    score = score_pair("alpha", "alpha", language="en")
    aggregates = aggregate_scores([score, replace(score, normalizer="different")])
    assert [aggregate.normalizer for aggregate in aggregates] == ["different", "whisper_english"]


def test_record_validation_names_missing_fields():
    with pytest.raises(ValueError, match="record 1.*hypothesis"):
        score_records([{"language": "en", "reference": "hello"}])


def test_record_aggregation_never_merges_models():
    scored = score_records(
        [
            {"model": "a", "language": "en", "reference": "alpha", "hypothesis": "wrong"},
            {"model": "b", "language": "en", "reference": "alpha", "hypothesis": "alpha"},
        ]
    )
    aggregates = aggregate_record_scores(scored)
    assert [(row.dimensions["model"], row.value) for row in aggregates] == [("a", 1.0), ("b", 0.0)]


def test_custom_grouping_is_additive_to_model_identity():
    scored = score_records(
        [
            {"model": "a", "dataset": "set", "language": "en", "reference": "alpha", "hypothesis": "wrong"},
            {"model": "b", "dataset": "set", "language": "en", "reference": "alpha", "hypothesis": "alpha"},
        ]
    )
    aggregates = aggregate_record_scores(scored, group_by=("dataset",))
    assert [(row.dimensions, row.value) for row in aggregates] == [
        ({"model": "a", "dataset": "set"}, 1.0),
        ({"model": "b", "dataset": "set"}, 0.0),
    ]


def test_custom_grouping_rejects_missing_or_misspelled_fields():
    scored = score_records([{"model": "a", "language": "en", "reference": "alpha", "hypothesis": "alpha"}])
    with pytest.raises(ValueError, match="aggregation fields.*dataset"):
        aggregate_record_scores(scored, group_by=("dataset",))


def test_record_aggregation_rejects_partially_populated_model_identity():
    scored = score_records(
        [
            {"model": "a", "language": "en", "reference": "alpha", "hypothesis": "alpha"},
            {"language": "en", "reference": "alpha", "hypothesis": "alpha"},
        ]
    )
    with pytest.raises(ValueError, match="model identity fields.*model"):
        aggregate_record_scores(scored)


def test_record_aggregation_keeps_boolean_and_numeric_dimensions_separate():
    scored = score_records(
        [
            {"cohort": True, "language": "en", "reference": "alpha", "hypothesis": "alpha"},
            {"cohort": 1, "language": "en", "reference": "alpha", "hypothesis": "alpha"},
        ]
    )
    aggregates = aggregate_record_scores(scored, group_by=("cohort",))
    assert [aggregate.dimensions["cohort"] for aggregate in aggregates] == [True, 1]


def test_record_aggregation_rejects_non_scalar_dimensions():
    scored = score_records([{"cohort": ["a"], "language": "en", "reference": "alpha", "hypothesis": "alpha"}])
    with pytest.raises(ValueError, match="cohort.*finite scalar JSON"):
        aggregate_record_scores(scored, group_by=("cohort",))


def test_record_aggregation_namespaces_colliding_dimensions():
    scored = score_records(
        [
            {"metric": "source-a", "language": "en", "reference": "alpha", "hypothesis": "wrong"},
            {"metric": "source-b", "language": "en", "reference": "alpha", "hypothesis": "alpha"},
        ]
    )
    serialized = [row.to_dict() for row in aggregate_record_scores(scored, group_by=("metric",))]
    assert [row["dimensions"]["metric"] for row in serialized] == ["source-a", "source-b"]
    assert [row["metric"] for row in serialized] == ["wer", "wer"]


def test_record_fields_must_be_strings():
    with pytest.raises(ValueError, match="non-string.*reference"):
        score_records([{"language": "en", "reference": 42, "hypothesis": "forty two"}])
