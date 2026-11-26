"""Public normalization API.

The implementation is split by responsibility while these exports keep imports stable.
"""

from dai_asr_i18n.normalization.base import NormalizedText, TextNormalizer
from dai_asr_i18n.normalization.cleanup import (
    AnnotationRemover,
    ParenthesisRemover,
    RegexTagRemover,
    prepare_scoring_text,
    punctuation_to_spaces,
    strip_annotations,
)
from dai_asr_i18n.normalization.languages import (
    ArabicNormalizer,
    CjkNormalizer,
    ConservativeIndicNormalizer,
    CoreThenPunctuation,
    MarkPreservingNormalizer,
    SimplifiedChineseNormalizer,
    ThaiNormalizer,
    TurkicNormalizer,
    WhisperBasicNormalizer,
    WhisperEnglishNormalizer,
)
from dai_asr_i18n.normalization.manifest import normalization_manifest
from dai_asr_i18n.normalization.pair import PairNormalized, PairStep, Rewrite
from dai_asr_i18n.normalization.pipeline import PAIR_STEPS, STEPS, DaiAsrI18nNormalizer, Step
from dai_asr_i18n.normalization.profiles import (
    DAI_ASR_I18N_NORMALIZERS,
    POLICY_ID,
    WHISPER_BASELINE_NORMALIZERS,
    LanguageNormalizer,
    NormalizationProfile,
    NormalizationResult,
    PairNormalizationResult,
    base_normalizer,
    canonical_profile,
    normalize,
    normalize_pair,
    normalize_with_metadata,
    normalizer_for_language,
)
from dai_asr_i18n.normalization.resolver import normalizer_for_key
from dai_asr_i18n.normalization.tokens import tokenize

__all__ = [
    "AnnotationRemover",
    "ArabicNormalizer",
    "CjkNormalizer",
    "ConservativeIndicNormalizer",
    "CoreThenPunctuation",
    "LanguageNormalizer",
    "DAI_ASR_I18N_NORMALIZERS",
    "MarkPreservingNormalizer",
    "DaiAsrI18nNormalizer",
    "NormalizationProfile",
    "NormalizationResult",
    "NormalizedText",
    "PAIR_STEPS",
    "POLICY_ID",
    "PairNormalizationResult",
    "PairNormalized",
    "PairStep",
    "ParenthesisRemover",
    "RegexTagRemover",
    "Rewrite",
    "STEPS",
    "SimplifiedChineseNormalizer",
    "Step",
    "TextNormalizer",
    "ThaiNormalizer",
    "TurkicNormalizer",
    "WHISPER_BASELINE_NORMALIZERS",
    "WhisperBasicNormalizer",
    "WhisperEnglishNormalizer",
    "base_normalizer",
    "canonical_profile",
    "normalization_manifest",
    "normalize",
    "normalize_pair",
    "normalize_with_metadata",
    "normalizer_for_key",
    "normalizer_for_language",
    "prepare_scoring_text",
    "punctuation_to_spaces",
    "strip_annotations",
    "tokenize",
]
