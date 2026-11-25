"""Normalization profiles and language routing.

``dai_asr_i18n`` is the benchmark normalizer
(:class:`~dai_asr_i18n.normalization.pipeline.DaiAsrI18nNormalizer`), with pair steps that read a hypothesis
against its reference. ``whisper_baseline`` is stock Whisper normalization
for comparison: English for English, Basic for every other language.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from functools import cache
from types import MappingProxyType
from typing import Literal, cast

from dai_asr_i18n.languages import BENCHMARK_LANGUAGES, canonical_language
from dai_asr_i18n.normalization.base import NormalizedText, TextNormalizer
from dai_asr_i18n.normalization.cleanup import CJK_SPACE, AnnotationRemover, ParenthesisRemover
from dai_asr_i18n.normalization.pair import Rewrite
from dai_asr_i18n.normalization.pipeline import (
    BASE_NORMALIZERS,
    DaiAsrI18nNormalizer,
    base_key,
    dai_asr_i18n_normalizer,
)
from dai_asr_i18n.normalization.resolver import normalizer_for_key
from dai_asr_i18n.policies import NORMALIZATION_POLICY_ID

NormalizationProfile = Literal["dai_asr_i18n", "whisper_baseline"]

POLICY_ID = NORMALIZATION_POLICY_ID
DAI_ASR_I18N_NORMALIZERS: Mapping[str, str] = MappingProxyType(dict(BASE_NORMALIZERS))
WHISPER_BASELINE_NORMALIZERS: Mapping[str, str] = MappingProxyType(
    {language: "whisper_english" if language == "en" else "whisper_basic" for language in BENCHMARK_LANGUAGES}
)


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Normalized text with enough provenance to reproduce the operation."""

    text: str
    language: str | None
    profile: NormalizationProfile
    normalizer: str
    policy: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairNormalizationResult:
    """A normalized reference and hypothesis, with the pair-step rewrites applied to the hypothesis."""

    reference: str
    hypothesis: str
    language: str | None
    profile: NormalizationProfile
    normalizer: str
    policy: str
    rewrites: tuple[Rewrite, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["rewrites"] = [asdict(rewrite) for rewrite in self.rewrites]
        return value


def canonical_profile(profile: str) -> NormalizationProfile:
    """Validate and canonicalize a published normalization profile name."""

    aliases = {"dai_asr_i18n": "dai_asr_i18n", "whisper_baseline": "whisper_baseline"}
    try:
        return cast(NormalizationProfile, aliases[profile.strip().lower()])
    except (AttributeError, KeyError) as error:
        raise ValueError(f"unknown normalization profile: {profile!r}") from error


class LanguageNormalizer(TextNormalizer):
    """Shared cleanup around one language transform, with every parenthetical and its contents removed.

    This is the per-language normalization the benchmark pipeline builds on. The transliteration maps name
    their classes with it (:func:`dai_asr_i18n.normalization.transliteration_maps.latin_class`).
    """

    def __init__(
        self,
        inner: TextNormalizer,
        language: str | None,
        *,
        parenthesis_remover: TextNormalizer | None = None,
        annotation_remover: TextNormalizer | None = None,
    ) -> None:
        self.inner = inner
        self.language = language
        self.key = inner.resolved_key
        self.parenthesis_remover = parenthesis_remover or ParenthesisRemover()
        self.annotation_remover = annotation_remover or AnnotationRemover()

    def apply(self, text: str) -> str:
        prepared = self.parenthesis_remover.apply(text)
        if self.language in {"ja", "zh"}:
            prepared = CJK_SPACE.sub("", prepared)
        result = self.inner.apply(self.annotation_remover.apply(prepared))
        if self.language in {"ja", "zh"}:
            result = CJK_SPACE.sub("", result)
        return result


@cache
def base_normalizer(language: str | None) -> LanguageNormalizer:
    """The language transform with shared cleanup and its own punctuation pass."""

    code = canonical_language(language)
    return LanguageNormalizer(normalizer_for_key(base_key(code)), code)


def normalizer_for_language(language: str | None, profile: str = "dai_asr_i18n") -> TextNormalizer:
    """Resolve a reusable class-based normalizer for a language and profile."""

    code = canonical_language(language)
    if canonical_profile(profile) == "whisper_baseline":
        return normalizer_for_key(WHISPER_BASELINE_NORMALIZERS.get(code or "", "whisper_basic"))
    return dai_asr_i18n_normalizer(code)


def _policy(profile: NormalizationProfile) -> str:
    return "stock-whisper" if profile == "whisper_baseline" else POLICY_ID


def normalize_with_metadata(text: str, language: str | None, profile: str = "dai_asr_i18n") -> NormalizationResult:
    """Normalize one string under the `dai_asr_i18n` or `whisper_baseline` profile."""

    code = canonical_language(language)
    selected_profile = canonical_profile(profile)
    normalizer = normalizer_for_language(code, selected_profile)
    return NormalizationResult(
        str(normalizer.apply(text)), code, selected_profile, normalizer.resolved_key, _policy(selected_profile)
    )


def normalize(text: str, language: str | None, profile: str = "dai_asr_i18n") -> str:
    """Return normalized text without the provenance wrapper."""

    return normalize_with_metadata(text, language, profile).text


def normalize_pair(
    reference: str,
    hypothesis: str,
    language: str | None,
    profile: str = "dai_asr_i18n",
) -> PairNormalizationResult:
    """Normalize a reference and a hypothesis together, as scoring reads them.

    The reference is exactly ``normalize(reference)``. Under ``dai_asr_i18n`` the hypothesis may also take the
    reference's spelling where a pair step allows it (transliteration; apostrophes in de, tl and tr); the
    rewrites are returned. Under ``whisper_baseline`` both sides are normalized on their own.
    """

    code = canonical_language(language)
    selected_profile = canonical_profile(profile)
    normalizer = normalizer_for_language(code, selected_profile)
    if isinstance(normalizer, DaiAsrI18nNormalizer):
        pair = normalizer.normalize_pair(reference, hypothesis)
        reference_text, hypothesis_text, rewrites = pair.reference, pair.hypothesis, pair.rewrites
    else:
        reference_text, hypothesis_text, rewrites = normalizer.apply(reference), normalizer.apply(hypothesis), ()
    return PairNormalizationResult(
        reference=str(reference_text),
        hypothesis=str(hypothesis_text),
        language=code,
        profile=selected_profile,
        normalizer=normalizer.resolved_key,
        policy=_policy(selected_profile),
        rewrites=tuple(rewrites),
    )


__all__ = [
    "DAI_ASR_I18N_NORMALIZERS",
    "POLICY_ID",
    "WHISPER_BASELINE_NORMALIZERS",
    "LanguageNormalizer",
    "NormalizationProfile",
    "NormalizationResult",
    "NormalizedText",
    "PairNormalizationResult",
    "base_normalizer",
    "canonical_profile",
    "normalize",
    "normalize_pair",
    "normalize_with_metadata",
    "normalizer_for_language",
]
