"""Lazy construction of normalization implementations from stable policy keys."""

from __future__ import annotations

from functools import cache

from dai_asr_i18n.normalization.base import TextNormalizer
from dai_asr_i18n.normalization.languages import (
    INDIC_BLOCKS,
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


def normalizer_for_key(key: str) -> TextNormalizer:
    """Resolve a stable policy key to one lazily cached normalizer."""

    if not isinstance(key, str):
        raise TypeError(f"normalizer key must be a string, got {type(key).__name__}")
    return _normalizer_for_key(key)


@cache
def _normalizer_for_key(key: str) -> TextNormalizer:
    if key == "whisper_basic":
        return WhisperBasicNormalizer()
    if key == "whisper_english":
        return WhisperEnglishNormalizer()
    if key == "unicode_marks":
        return MarkPreservingNormalizer()
    if key == "turkic":
        return TurkicNormalizer()
    if key == "cjk":
        return CjkNormalizer()
    if key == "chinese_t2s":
        return SimplifiedChineseNormalizer()
    if key == "arabic":
        return ArabicNormalizer()
    if key == "thai":
        return ThaiNormalizer()
    if key.startswith("indic_conservative_"):
        language = key.removeprefix("indic_conservative_")
        if language in INDIC_BLOCKS:
            return ConservativeIndicNormalizer(language)
    raise ValueError(f"unknown normalizer key: {key!r}")


__all__ = ["normalizer_for_key"]
