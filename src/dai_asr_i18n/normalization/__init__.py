"""Public normalization API.

The implementation is split by responsibility while these exports keep imports stable.
"""

from dai_asr_i18n.normalization.base import TextNormalizer
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
    MarkPreservingNormalizer,
    SimplifiedChineseNormalizer,
    ThaiNormalizer,
    TurkicNormalizer,
    WhisperBasicNormalizer,
    WhisperEnglishNormalizer,
)
from dai_asr_i18n.normalization.manifest import normalization_manifest
from dai_asr_i18n.normalization.profiles import (
    DAI_ASR_I18N_NORMALIZERS,
    POLICY_ID,
    WHISPER_BASELINE_NORMALIZERS,
    DaiAsrI18nNormalizer,
    NormalizationProfile,
    NormalizationResult,
    NormalizedText,
    canonical_profile,
    normalize,
    normalize_with_metadata,
    normalizer_for_language,
)
from dai_asr_i18n.normalization.resolver import normalizer_for_key

__all__ = [
    "AnnotationRemover",
    "ArabicNormalizer",
    "CjkNormalizer",
    "ConservativeIndicNormalizer",
    "MarkPreservingNormalizer",
    "NormalizationProfile",
    "NormalizationResult",
    "NormalizedText",
    "DAI_ASR_I18N_NORMALIZERS",
    "POLICY_ID",
    "ParenthesisRemover",
    "DaiAsrI18nNormalizer",
    "RegexTagRemover",
    "SimplifiedChineseNormalizer",
    "TextNormalizer",
    "ThaiNormalizer",
    "TurkicNormalizer",
    "WHISPER_BASELINE_NORMALIZERS",
    "WhisperBasicNormalizer",
    "WhisperEnglishNormalizer",
    "canonical_profile",
    "normalization_manifest",
    "normalize",
    "normalize_with_metadata",
    "normalizer_for_key",
    "normalizer_for_language",
    "prepare_scoring_text",
    "punctuation_to_spaces",
    "strip_annotations",
]
