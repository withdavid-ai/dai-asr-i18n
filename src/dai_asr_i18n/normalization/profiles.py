"""Benchmark/Whisper profile composition and language routing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Literal, cast

import regex

from dai_asr_i18n.languages import BENCHMARK_LANGUAGES, canonical_language
from dai_asr_i18n.normalization.base import TextNormalizer
from dai_asr_i18n.normalization.cleanup import AnnotationRemover, ParenthesisRemover
from dai_asr_i18n.normalization.resolver import normalizer_for_key
from dai_asr_i18n.policies import NORMALIZATION_POLICY_ID

NormalizationProfile = Literal["dai_asr_i18n", "whisper_baseline"]

POLICY_ID = NORMALIZATION_POLICY_ID
_CJK_SPACE = regex.compile(r"(?<=[\p{Han}\p{Hiragana}\p{Katakana}ー])\s+(?=[\p{Han}\p{Hiragana}\p{Katakana}ー])")

DAI_ASR_I18N_NORMALIZERS: Mapping[str, str] = MappingProxyType(
    {
        "en": "whisper_english",
        **{language: "unicode_marks" for language in ("es", "fr", "de", "it", "pt", "ru", "id", "tl", "vi")},
        "tr": "turkic",
        "ar": "arabic",
        **{language: f"indic_conservative_{language}" for language in ("hi", "mr", "bn", "ta", "te")},
        "zh": "chinese_t2s",
        "ja": "cjk",
        "ko": "cjk",
        "th": "thai",
    }
)

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


class NormalizedText(str):
    """String carrying the benchmark profile that already normalized it."""

    profile: str

    def __new__(cls, text: str, profile: str) -> NormalizedText:
        instance = super().__new__(cls, text)
        instance.profile = profile
        return instance

    def __reduce__(self):
        return type(self), (str(self), self.profile)


def canonical_profile(profile: str) -> NormalizationProfile:
    """Validate and canonicalize a published normalization profile name."""

    aliases = {
        "dai_asr_i18n": "dai_asr_i18n",
        "whisper_baseline": "whisper_baseline",
    }
    try:
        return cast(NormalizationProfile, aliases[profile.strip().lower()])
    except (AttributeError, KeyError) as error:
        raise ValueError(f"unknown normalization profile: {profile!r}") from error


class DaiAsrI18nNormalizer(TextNormalizer):
    """Apply shared benchmark cleanup around one language-specific normalizer."""

    key = "dai_asr_i18n"

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
        self.parenthesis_remover = parenthesis_remover or ParenthesisRemover()
        self.annotation_remover = annotation_remover or AnnotationRemover()
        self.profile = ":".join(
            (
                POLICY_ID,
                "dai_asr_i18n",
                language or "default",
                inner.resolved_key,
                self.parenthesis_remover.resolved_key,
                self.annotation_remover.resolved_key,
            )
        )

    @property
    def resolved_key(self) -> str:
        return self.inner.resolved_key

    def apply(self, text: str) -> str:
        if isinstance(text, NormalizedText):
            if text.profile != self.profile:
                raise ValueError("normalized text belongs to another profile; normalize the original text")
            return text
        prepared = self.parenthesis_remover.apply(text)
        if self.language in {"ja", "zh"}:
            prepared = _CJK_SPACE.sub("", prepared)
        result = self.inner.apply(self.annotation_remover.apply(prepared))
        if self.language in {"ja", "zh"}:
            result = _CJK_SPACE.sub("", result)
        return NormalizedText(result, self.profile)


def normalizer_for_language(language: str | None, profile: str = "dai_asr_i18n") -> TextNormalizer:
    """Resolve a reusable class-based normalizer for a language and profile."""

    code = canonical_language(language)
    selected_profile = canonical_profile(profile)
    if selected_profile == "whisper_baseline":
        key = WHISPER_BASELINE_NORMALIZERS.get(code or "", "whisper_basic")
        return normalizer_for_key(key)
    key = DAI_ASR_I18N_NORMALIZERS.get(code or "", "whisper_basic")
    return DaiAsrI18nNormalizer(normalizer_for_key(key), code)


def normalize_with_metadata(
    text: str,
    language: str | None,
    profile: str = "dai_asr_i18n",
) -> NormalizationResult:
    """Normalize one string under the benchmark or stock-Whisper profile."""

    code = canonical_language(language)
    selected_profile = canonical_profile(profile)
    normalizer = normalizer_for_language(code, selected_profile)
    policy = "stock-whisper" if selected_profile == "whisper_baseline" else POLICY_ID
    return NormalizationResult(normalizer.apply(text), code, selected_profile, normalizer.resolved_key, policy)


def normalize(text: str, language: str | None, profile: str = "dai_asr_i18n") -> str:
    """Return normalized text without the provenance wrapper."""

    return normalize_with_metadata(text, language, profile).text


__all__ = [
    "NormalizationProfile",
    "NormalizationResult",
    "NormalizedText",
    "DAI_ASR_I18N_NORMALIZERS",
    "POLICY_ID",
    "DaiAsrI18nNormalizer",
    "WHISPER_BASELINE_NORMALIZERS",
    "canonical_profile",
    "normalize",
    "normalize_with_metadata",
    "normalizer_for_language",
]
