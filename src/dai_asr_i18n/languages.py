"""Language and headline-metric policy for the public benchmark."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

PublicMetric = Literal["wer", "cer"]
SourceMetric = Literal["wer", "cer", "ser"]


def canonical_language(language: str | None) -> str | None:
    """Return the lowercase primary BCP-47 subtag.

    Examples: ``zh-Hant`` becomes ``zh`` and ``pt_BR`` becomes ``pt``.
    """

    if not language:
        return None
    base = language.strip().lower().replace("_", "-").split("-")[0]
    return base or None


@dataclass(frozen=True, slots=True)
class MetricSelection:
    """Public label and physical scoring row for a benchmark language.

    Vietnamese is deliberately explicit: its requested public label is WER, while whitespace
    delimits orthographic syllables, so the physical units are recorded as SER.
    """

    public_metric: PublicMetric
    source_metric: SourceMetric


BENCHMARK_LANGUAGES: tuple[str, ...] = (
    "ar",
    "bn",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "id",
    "it",
    "ja",
    "ko",
    "mr",
    "pt",
    "ru",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "vi",
    "zh",
)

# Public reporting uses CER for Chinese, Japanese, Korean, and Thai; every other release language
# uses WER. Scoring can still compute an explicitly requested companion metric.
PUBLIC_METRIC: dict[str, PublicMetric] = {
    language: "cer" if language in {"zh", "ja", "ko", "th"} else "wer" for language in BENCHMARK_LANGUAGES
}


def validate_public_metric_map(
    metric_map: Mapping[str, str] = PUBLIC_METRIC,
    languages: Iterable[str] = BENCHMARK_LANGUAGES,
) -> None:
    """Raise with exact coverage or value drift in a candidate metric map."""

    expected = set(languages)
    actual = set(metric_map)
    problems: list[str] = []
    if missing := sorted(expected - actual):
        problems.append(f"missing languages: {missing}")
    if extra := sorted(actual - expected):
        problems.append(f"unexpected languages: {extra}")
    if invalid := sorted((lang, metric) for lang, metric in metric_map.items() if metric not in {"wer", "cer"}):
        problems.append(f"invalid metrics: {invalid}")
    if problems:
        raise ValueError("public metric map is invalid (" + "; ".join(problems) + ")")


def metric_selection_for_language(language: str | None) -> MetricSelection | None:
    """Return the benchmark metric selection, or ``None`` outside the release roster."""

    code = canonical_language(language) or ""
    public = PUBLIC_METRIC.get(code)
    if public is None:
        return None
    source: SourceMetric = "ser" if code == "vi" else public
    return MetricSelection(public_metric=public, source_metric=source)


validate_public_metric_map()
