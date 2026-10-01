"""Dataset scoring, validation, and micro-aggregation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from dai_asr_i18n.scoring.core import score_pair
from dai_asr_i18n.scoring.models import AggregateScore, EditCounts, Metric, RecordAggregate, Score

_MODEL_IDENTITY_FIELDS = ("model", "model_version", "channel_config")


def _dimension_key(field: str, value: Any) -> tuple[str, Any]:
    if value is None:
        return "null", None
    if isinstance(value, bool):
        return "boolean", value
    if isinstance(value, str):
        return "string", value
    if isinstance(value, int):
        return "number", value
    if isinstance(value, float) and math.isfinite(value):
        return "number", value
    raise ValueError(f"aggregation dimension {field!r} must be a finite scalar JSON value")


def aggregate_scores(scores: Iterable[Score]) -> list[AggregateScore]:
    """Micro-average scores by language/profile/metric using summed sufficient statistics."""

    grouped: dict[tuple[str | None, str, Metric, Metric, str, str, str, str, str], tuple[int, EditCounts]] = (
        defaultdict(lambda: (0, EditCounts()))
    )
    for score in scores:
        key = (
            score.language,
            score.profile,
            score.metric,
            score.source_metric,
            score.unit,
            score.normalizer,
            score.normalization_policy,
            score.metric_policy,
            score.scoring_policy,
        )
        items, counts = grouped[key]
        grouped[key] = (items + 1, counts + score.counts)
    return [
        AggregateScore(
            language=language,
            profile=profile,
            metric=metric,
            source_metric=source_metric,
            unit=unit,
            normalizer=normalizer,
            normalization_policy=normalization_policy,
            metric_policy=metric_policy,
            scoring_policy=scoring_policy,
            items=items,
            counts=counts,
        )
        for (
            language,
            profile,
            metric,
            source_metric,
            unit,
            normalizer,
            normalization_policy,
            metric_policy,
            scoring_policy,
        ), (items, counts) in sorted(
            grouped.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])
        )
    ]


def aggregate_record_scores(
    scored_records: Iterable[tuple[Mapping[str, Any], Score]],
    *,
    group_by: Sequence[str] = (),
) -> list[RecordAggregate]:
    """Micro-average records, preserving model identity plus requested dimensions."""

    scored = list(scored_records)
    requested = list(dict.fromkeys(group_by))
    missing = [field for field in requested if any(field not in record for record, _ in scored)]
    if missing:
        raise ValueError(f"aggregation fields are missing from one or more records: {', '.join(missing)}")
    partial_identity = [
        field
        for field in _MODEL_IDENTITY_FIELDS
        if any(field in record for record, _ in scored) and any(field not in record for record, _ in scored)
    ]
    if partial_identity:
        raise ValueError(f"model identity fields are missing from one or more records: {', '.join(partial_identity)}")
    identity = [field for field in _MODEL_IDENTITY_FIELDS if scored and all(field in record for record, _ in scored)]
    dimensions = identity + [field for field in requested if field not in identity]
    grouped: dict[tuple[Any, ...], tuple[int, EditCounts]] = defaultdict(lambda: (0, EditCounts()))
    for record, score in scored:
        values = tuple(record.get(field) for field in dimensions)
        dimension_keys = tuple(_dimension_key(field, value) for field, value in zip(dimensions, values, strict=True))
        key = dimension_keys + (
            score.language,
            score.profile,
            score.metric,
            score.source_metric,
            score.unit,
            score.normalizer,
            score.normalization_policy,
            score.metric_policy,
            score.scoring_policy,
        )
        items, counts = grouped[key]
        grouped[key] = (items + 1, counts + score.counts)

    results = []
    for key, (items, counts) in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        values = tuple(value for _, value in key[: len(dimensions)])
        (
            language,
            profile,
            metric,
            source_metric,
            unit,
            normalizer,
            normalization_policy,
            metric_policy,
            scoring_policy,
        ) = key[len(dimensions) :]
        results.append(
            RecordAggregate(
                dict(zip(dimensions, values, strict=True)),
                AggregateScore(
                    language=language,
                    profile=profile,
                    metric=metric,
                    source_metric=source_metric,
                    unit=unit,
                    normalizer=normalizer,
                    normalization_policy=normalization_policy,
                    metric_policy=metric_policy,
                    scoring_policy=scoring_policy,
                    items=items,
                    counts=counts,
                ),
            )
        )
    return results


def score_records(
    records: Iterable[Mapping[str, Any]],
    *,
    profile: str = "dai_asr_i18n",
    metric: str | None = None,
) -> list[tuple[Mapping[str, Any], Score]]:
    """Score records containing ``reference``, ``hypothesis``, and ``language`` fields."""

    results = []
    for index, record in enumerate(records, start=1):
        missing = [key for key in ("reference", "hypothesis", "language") if key not in record]
        if missing:
            raise ValueError(f"record {index} is missing required fields: {', '.join(missing)}")
        invalid = [key for key in ("reference", "hypothesis", "language") if not isinstance(record[key], str)]
        if invalid:
            raise ValueError(f"record {index} has non-string required fields: {', '.join(invalid)}")
        results.append(
            (
                record,
                _score_record(record, index=index, profile=profile, metric=metric),
            )
        )
    return results


def _score_record(record: Mapping[str, Any], *, index: int, profile: str, metric: str | None) -> Score:
    try:
        return score_pair(
            record["reference"],
            record["hypothesis"],
            language=record["language"],
            profile=profile,
            metric=metric,
        )
    except ValueError as error:
        raise ValueError(f"record {index}: {error}") from error


__all__ = ["aggregate_record_scores", "aggregate_scores", "score_records"]
