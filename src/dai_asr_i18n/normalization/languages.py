"""Stock Whisper and benchmark language-specific normalization implementations."""

from __future__ import annotations

import re
import unicodedata
from functools import cached_property
from pathlib import Path

from dai_asr_i18n.normalization.base import TextNormalizer
from dai_asr_i18n.normalization.cleanup import punctuation_to_spaces

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)

_ARABIC_HARAKAT = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]")
# ZWNJ is formatting-only under this Arabic policy but can carry orthographic meaning in Persian
# and Urdu. Revisit the joiner rule before routing either language through this normalizer.
_ARABIC_FORMAT_CONTROLS = re.compile("[\u00ad\u061c\u200c-\u200f\u202a-\u202e\u2060\u2066-\u206f\ufeff]")
_ARABIC_ZERO_WIDTH_SPACE = "\u200b"
_ALEF_VARIANTS = re.compile("[آأإٱ]")

INDIC_BLOCKS = {
    "hi": "\u0900-\u097f",
    "mr": "\u0900-\u097f",
    "bn": "\u0980-\u09ff",
    "ta": "\u0b80-\u0bff",
    "te": "\u0c00-\u0c7f",
}
# U+0E2F THAI CHARACTER PAIYANNOI functions as punctuation but Unicode classifies
# it as a letter (Lo), so the shared punctuation/symbol pass cannot remove it.
_THAI_PAIYANNOI = re.compile("ฯ")


class WhisperBasicNormalizer(TextNormalizer):
    """Stock Whisper ``BasicTextNormalizer`` for the multilingual baseline."""

    key = "whisper_basic"

    @cached_property
    def implementation(self):
        from whisper_normalizer.basic import BasicTextNormalizer

        return BasicTextNormalizer()

    def apply(self, text: str) -> str:
        return self.implementation(text)


class WhisperEnglishNormalizer(TextNormalizer):
    """Stock Whisper English normalization, including number and spelling rules."""

    key = "whisper_english"

    @cached_property
    def implementation(self):
        from whisper_normalizer.english import EnglishTextNormalizer

        return EnglishTextNormalizer()

    def apply(self, text: str) -> str:
        return self.implementation(text)


class MarkPreservingNormalizer(TextNormalizer):
    """Whisper-like cleanup that preserves meaningful Unicode marks."""

    key = "unicode_marks"

    def apply(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
        text = re.sub(r"\(([^)]+?)\)", "", text)
        return punctuation_to_spaces(unicodedata.normalize("NFKC", text).lower())


class TurkicNormalizer(TextNormalizer):
    """Locale-correct Turkish casing that preserves dotted/dotless-i distinctions."""

    key = "turkic"
    _CASE_MAP = str.maketrans({"İ": "i", "I": "ı", "Ş": "ş", "Ğ": "ğ", "Ç": "ç", "Ö": "ö", "Ü": "ü"})

    def apply(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text).translate(self._CASE_MAP).lower()
        text = _PUNCT.sub(" ", unicodedata.normalize("NFC", text))
        return _WS.sub(" ", text).strip()


class CjkNormalizer(TextNormalizer):
    """NFKC, lowercase, and punctuation cleanup for CJK text."""

    key = "cjk"

    def _normalize_script(self, text: str) -> str:
        """Apply an optional script transform after Unicode normalization."""

        return text

    def apply(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text).lower()
        return punctuation_to_spaces(self._normalize_script(text))


class SimplifiedChineseNormalizer(CjkNormalizer):
    """Apply CJK cleanup and convert traditional Chinese to simplified Chinese."""

    key = "chinese_t2s"

    @cached_property
    def converter(self):
        import opencc

        config = Path(opencc.__file__).parent / "clib/share/opencc/t2s.json"
        return opencc.OpenCC(str(config))

    def _normalize_script(self, text: str) -> str:
        return self.converter.convert(text)


class ArabicNormalizer(TextNormalizer):
    """Unvocalized Arabic with invisible-control and letter-variant cleanup."""

    key = "arabic"

    def apply(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        # Directional controls, joiners, BOM, word joiner, and soft hyphen are invisible
        # formatting artifacts. Preserve a zero-width space as a real word boundary instead.
        text = text.replace(_ARABIC_ZERO_WIDTH_SPACE, " ")
        text = _ARABIC_FORMAT_CONTROLS.sub("", text)
        text = _ARABIC_HARAKAT.sub("", text)
        text = text.replace("ـ", "")
        text = _ALEF_VARIANTS.sub("ا", text)
        text = text.replace("ى", "ي")
        return punctuation_to_spaces(text.lower())


class ConservativeIndicNormalizer(TextNormalizer):
    """Indic NLP common cleanup restricted to runs in one declared script."""

    def __init__(self, language: str) -> None:
        if language not in INDIC_BLOCKS:
            raise ValueError(f"unsupported conservative Indic language: {language!r}")
        self.language = language
        self.key = f"indic_conservative_{language}"
        block = INDIC_BLOCKS[language]
        self._runs = re.compile(f"[\u200c\u200d]*[{block}][{block}\u200c\u200d]*")

    @cached_property
    def implementation(self):
        from indicnlp.normalize.indic_normalize import BaseNormalizer

        return BaseNormalizer(
            self.language,
            remove_nuktas=False,
            nasals_mode="do_nothing",
            do_normalize_chandras=False,
            do_normalize_vowel_ending=False,
        )

    def apply(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text).lower()
        text = text.replace("\ufeff", "").replace("\u2060", "").replace("\u00ad", "")
        text = text.replace("\u200b", " ")
        text = self._runs.sub(lambda match: self.implementation.normalize(match.group()), text)
        return punctuation_to_spaces(unicodedata.normalize("NFC", text))


class ThaiNormalizer(TextNormalizer):
    """NFC cleanup that preserves Thai vowel, tone, and repetition marks."""

    key = "thai"

    def apply(self, text: str) -> str:
        text = _THAI_PAIYANNOI.sub(" ", unicodedata.normalize("NFC", text).lower())
        # The curated pass handles ฯ (category Lo); the shared Unicode-category pass handles
        # every actual punctuation/symbol character without deleting Thai combining marks.
        return punctuation_to_spaces(text)


__all__ = [
    "ArabicNormalizer",
    "CjkNormalizer",
    "ConservativeIndicNormalizer",
    "MarkPreservingNormalizer",
    "SimplifiedChineseNormalizer",
    "ThaiNormalizer",
    "TurkicNormalizer",
    "WhisperBasicNormalizer",
    "WhisperEnglishNormalizer",
]
