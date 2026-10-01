"""Normalize and score one reference/hypothesis pair."""

from __future__ import annotations

from typing import cast

from dai_asr_i18n.languages import canonical_language, metric_selection_for_language
from dai_asr_i18n.normalization import canonical_profile, normalize_with_metadata
from dai_asr_i18n.policies import METRIC_POLICY_VERSION, SCORING_POLICY_VERSION
from dai_asr_i18n.scoring.alignment import align_tokens, tokens_for_metric
from dai_asr_i18n.scoring.models import Metric, Score


def _resolve_metric(language: str | None, metric: str | None) -> tuple[str | None, Metric, Metric]:
    """Resolve the public label and physical scoring unit shared by every scorer."""

    code = canonical_language(language)
    selection = metric_selection_for_language(code)
    if metric in {None, "auto"}:
        if selection is None:
            raise ValueError(f"auto metric is unavailable for language {code!r}; choose an explicit metric")
        return code, selection.public_metric, selection.source_metric

    normalized_metric = metric.strip().lower()
    if normalized_metric not in {"wer", "cer", "ser"}:
        raise ValueError(f"unknown metric: {metric!r}")
    selected_metric = cast(Metric, normalized_metric)
    if selected_metric == "wer" and code in {"zh", "ja", "th"}:
        raise ValueError(f"WER requires a word segmenter that is not bundled for language {code!r}")
    if selected_metric == "ser" and code != "vi":
        raise ValueError("SER is supported only for Vietnamese in the standalone scorer")
    source_metric: Metric = "ser" if selected_metric == "wer" and code == "vi" else selected_metric
    return code, selected_metric, source_metric


def score_pair(
    reference: str,
    hypothesis: str,
    *,
    language: str | None,
    profile: str = "dai_asr_i18n",
    metric: str | None = None,
) -> Score:
    """Normalize and score one reference/hypothesis pair.

    With ``metric=None`` (or ``"auto"``), the public benchmark policy is used: CER for CJKT and
    WER for the other release languages. Languages outside the release roster require an explicit
    metric. Vietnamese's whitespace units are recorded as ``source_metric="ser"`` even though the
    requested public label is WER.
    Explicit WER is rejected for release languages that need an unbundled word segmenter, and
    explicit SER is supported only for Vietnamese.
    """

    code, selected_metric, source_metric = _resolve_metric(language, metric)

    reference_result = normalize_with_metadata(reference, code, profile)
    hypothesis_result = normalize_with_metadata(hypothesis, code, profile)
    counts = align_tokens(
        tokens_for_metric(reference_result.text, selected_metric),
        tokens_for_metric(hypothesis_result.text, selected_metric),
    )
    unit = "character" if selected_metric == "cer" else ("syllable" if source_metric == "ser" else "word")
    return Score(
        language=code,
        profile=canonical_profile(profile),
        metric=selected_metric,
        source_metric=source_metric,
        unit=unit,
        normalizer=reference_result.normalizer,
        normalization_policy=reference_result.policy,
        metric_policy=METRIC_POLICY_VERSION,
        scoring_policy=SCORING_POLICY_VERSION,
        reference_normalized=reference_result.text,
        hypothesis_normalized=hypothesis_result.text,
        counts=counts,
    )


__all__ = ["SCORING_POLICY_VERSION", "score_pair"]
