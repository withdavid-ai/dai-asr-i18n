"""Serializable result and sufficient-statistic models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Metric = Literal["wer", "cer", "ser"]


@dataclass(frozen=True, slots=True)
class EditCounts:
    """Levenshtein substitution/deletion/insertion/hit counts."""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0

    @property
    def numerator(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def denominator(self) -> int:
        return self.hits + self.substitutions + self.deletions

    @property
    def error_rate(self) -> float:
        if self.denominator == 0:
            return 0.0 if self.numerator == 0 else 1.0
        return self.numerator / self.denominator

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            hits=self.hits + other.hits,
        )


@dataclass(frozen=True, slots=True)
class Score:
    """One normalized reference/hypothesis score and its lineage."""

    language: str | None
    profile: str
    metric: Metric
    source_metric: Metric
    unit: str
    normalizer: str
    normalization_policy: str
    metric_policy: str
    scoring_policy: str
    reference_normalized: str
    hypothesis_normalized: str
    counts: EditCounts

    @property
    def value(self) -> float:
        return self.counts.error_rate

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("counts")
        value.update(asdict(self.counts))
        value["numerator"] = self.counts.numerator
        value["denominator"] = self.counts.denominator
        value["value"] = self.value
        return value


@dataclass(frozen=True, slots=True)
class AggregateScore:
    """Micro-average of scores sharing language, profile, metric, and normalizer lineage."""

    language: str | None
    profile: str
    metric: Metric
    source_metric: Metric
    unit: str
    normalizer: str
    normalization_policy: str
    metric_policy: str
    scoring_policy: str
    items: int
    counts: EditCounts

    @property
    def value(self) -> float:
        return self.counts.error_rate

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("counts")
        value.update(asdict(self.counts))
        value["numerator"] = self.counts.numerator
        value["denominator"] = self.counts.denominator
        value["value"] = self.value
        return value


@dataclass(frozen=True, slots=True)
class RecordAggregate:
    """Micro-average with caller-selected dataset dimensions such as model and channel."""

    dimensions: dict[str, Any]
    score: AggregateScore

    @property
    def value(self) -> float:
        return self.score.value

    def to_dict(self) -> dict[str, Any]:
        return {"dimensions": dict(self.dimensions), **self.score.to_dict()}


__all__ = ["AggregateScore", "EditCounts", "Metric", "RecordAggregate", "Score"]
