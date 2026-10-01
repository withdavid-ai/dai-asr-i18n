"""The public metric map is explicit and stable."""

from dai_asr_i18n.languages import (
    BENCHMARK_LANGUAGES,
    PUBLIC_METRIC,
    canonical_language,
    metric_selection_for_language,
)


def test_language_tags_are_canonicalized():
    assert canonical_language("ZH-Hant") == "zh"
    assert canonical_language("pt_BR") == "pt"
    assert canonical_language("  ") is None


def test_public_policy_is_cer_for_cjkt_and_wer_elsewhere():
    assert len(BENCHMARK_LANGUAGES) == 21
    assert {language for language, metric in PUBLIC_METRIC.items() if metric == "cer"} == {"zh", "ja", "ko", "th"}
    assert all(PUBLIC_METRIC[language] == "wer" for language in set(BENCHMARK_LANGUAGES) - {"zh", "ja", "ko", "th"})


def test_vietnamese_public_and_physical_metrics_are_both_explicit():
    selection = metric_selection_for_language("vi-VN")
    assert selection is not None
    assert selection.public_metric == "wer"
    assert selection.source_metric == "ser"
