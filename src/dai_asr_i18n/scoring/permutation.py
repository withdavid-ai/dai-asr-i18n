"""Concatenated minimum-permutation speaker-attributed error rates."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

from dai_asr_i18n.normalization import canonical_profile, normalize_with_metadata
from dai_asr_i18n.policies import METRIC_POLICY_VERSION, SCORING_POLICY_VERSION
from dai_asr_i18n.scoring.alignment import align_tokens, tokens_for_metric
from dai_asr_i18n.scoring.core import _resolve_metric
from dai_asr_i18n.scoring.models import EditCounts

PermutationMetric = Literal["cpwer", "cpcer", "cpser"]
SpeakerAssignment = tuple[tuple[str | None, str | None], ...]


@dataclass(frozen=True, slots=True)
class PermutationScore:
    """One speaker-permutation score with physical-unit and policy provenance."""

    language: str | None
    profile: str
    metric: PermutationMetric
    source_metric: PermutationMetric
    unit: str
    normalizer: str
    normalization_policy: str
    metric_policy: str
    scoring_policy: str
    scorer: str
    reference_normalized: dict[str, str]
    hypothesis_normalized: dict[str, str]
    speaker_assignment: SpeakerAssignment
    counts: EditCounts

    @property
    def value(self) -> float:
        return self.counts.error_rate

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("counts")
        value.update(asdict(self.counts))
        value["speaker_assignment"] = [list(pair) for pair in self.speaker_assignment]
        value["numerator"] = self.counts.numerator
        value["denominator"] = self.counts.denominator
        value["value"] = self.value
        return value


def _validate_speaker_texts(value: Mapping[str, str], side: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{side} must be a mapping of speaker labels to text")
    result: dict[str, str] = {}
    for speaker, text in value.items():
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(f"{side} speaker labels must be nonempty strings")
        if not isinstance(text, str):
            raise TypeError(f"{side} text for speaker {speaker!r} must be a string")
        result[speaker] = text
    return result


def _align_speakers(
    reference: Mapping[str, list[str]],
    hypothesis: Mapping[str, list[str]],
) -> tuple[EditCounts, SpeakerAssignment, str]:
    """Minimize total edit errors over a square speaker assignment."""

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - exercised by installs without the extra
        raise RuntimeError('speaker permutation scoring requires: pip install "dai-asr-i18n[speaker]"') from error

    reference_labels: list[str | None] = sorted(reference)
    hypothesis_labels: list[str | None] = sorted(hypothesis)
    size = max(len(reference_labels), len(hypothesis_labels))
    if size == 0:
        return EditCounts(), (), f"scipy-{importlib.metadata.version('scipy')}-linear_sum_assignment"
    reference_labels.extend([None] * (size - len(reference_labels)))
    hypothesis_labels.extend([None] * (size - len(hypothesis_labels)))
    candidates = [
        [
            align_tokens(
                [] if reference_speaker is None else reference[reference_speaker],
                [] if hypothesis_speaker is None else hypothesis[hypothesis_speaker],
            )
            for hypothesis_speaker in hypothesis_labels
        ]
        for reference_speaker in reference_labels
    ]
    rows, columns = linear_sum_assignment([[score.numerator for score in row] for row in candidates])
    counts = EditCounts()
    assignment: list[tuple[str | None, str | None]] = []
    for row, column in zip(rows, columns, strict=True):
        counts += candidates[int(row)][int(column)]
        assignment.append((reference_labels[int(row)], hypothesis_labels[int(column)]))
    return counts, tuple(assignment), f"scipy-{importlib.metadata.version('scipy')}-linear_sum_assignment"


def score_speakers(
    reference_by_speaker: Mapping[str, str],
    hypothesis_by_speaker: Mapping[str, str],
    *,
    language: str | None,
    profile: str = "dai_asr_i18n",
    metric: str | None = None,
) -> PermutationScore:
    """Normalize and score speaker-attributed text in the selected public or explicit unit.

    Automatic selection reports cpCER for the public CER languages and cpWER elsewhere. Vietnamese
    is externally labeled cpWER while retaining ``source_metric="cpser"`` and ``unit="syllable"``.
    """

    code, selected_metric, source_metric = _resolve_metric(language, metric)
    reference = _validate_speaker_texts(reference_by_speaker, "reference")
    hypothesis = _validate_speaker_texts(hypothesis_by_speaker, "hypothesis")
    probe = normalize_with_metadata("", code, profile)

    def normalized(values: Mapping[str, str]) -> dict[str, str]:
        return {
            speaker: str(normalize_with_metadata(text, code, profile).text) for speaker, text in sorted(values.items())
        }

    normalized_reference = normalized(reference)
    normalized_hypothesis = normalized(hypothesis)
    tokenized_reference = {
        speaker: tokens_for_metric(text, source_metric) for speaker, text in normalized_reference.items()
    }
    tokenized_hypothesis = {
        speaker: tokens_for_metric(text, source_metric) for speaker, text in normalized_hypothesis.items()
    }
    counts, assignment, scorer = _align_speakers(tokenized_reference, tokenized_hypothesis)
    unit = "character" if source_metric == "cer" else ("syllable" if source_metric == "ser" else "word")
    return PermutationScore(
        language=code,
        profile=canonical_profile(profile),
        metric=cast(PermutationMetric, f"cp{selected_metric}"),
        source_metric=cast(PermutationMetric, f"cp{source_metric}"),
        unit=unit,
        normalizer=probe.normalizer,
        normalization_policy=probe.policy,
        metric_policy=METRIC_POLICY_VERSION,
        scoring_policy=SCORING_POLICY_VERSION,
        scorer=scorer,
        reference_normalized=normalized_reference,
        hypothesis_normalized=normalized_hypothesis,
        speaker_assignment=assignment,
        counts=counts,
    )


def cpwer(
    reference_by_speaker: Mapping[str, str],
    hypothesis_by_speaker: Mapping[str, str],
    *,
    language: str | None,
    profile: str = "dai_asr_i18n",
) -> PermutationScore:
    """Score concatenated minimum-permutation WER (or Vietnamese cpSER under its public label)."""

    return score_speakers(
        reference_by_speaker,
        hypothesis_by_speaker,
        language=language,
        profile=profile,
        metric="wer",
    )


def cpcer(
    reference_by_speaker: Mapping[str, str],
    hypothesis_by_speaker: Mapping[str, str],
    *,
    language: str | None,
    profile: str = "dai_asr_i18n",
) -> PermutationScore:
    """Score concatenated minimum-permutation CER over non-whitespace code points."""

    return score_speakers(
        reference_by_speaker,
        hypothesis_by_speaker,
        language=language,
        profile=profile,
        metric="cer",
    )


def cpser(
    reference_by_speaker: Mapping[str, str],
    hypothesis_by_speaker: Mapping[str, str],
    *,
    language: str = "vi",
    profile: str = "dai_asr_i18n",
) -> PermutationScore:
    """Score concatenated minimum-permutation Vietnamese syllable error rate."""

    return score_speakers(
        reference_by_speaker,
        hypothesis_by_speaker,
        language=language,
        profile=profile,
        metric="ser",
    )


__all__ = ["PermutationMetric", "PermutationScore", "cpcer", "cpser", "cpwer", "score_speakers"]
