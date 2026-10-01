"""Direct unit tests for every concrete normalization implementation."""

from __future__ import annotations

import pytest

from dai_asr_i18n.normalization import (
    AnnotationRemover,
    ArabicNormalizer,
    CjkNormalizer,
    ConservativeIndicNormalizer,
    DaiAsrI18nNormalizer,
    MarkPreservingNormalizer,
    NormalizedText,
    ParenthesisRemover,
    RegexTagRemover,
    SimplifiedChineseNormalizer,
    ThaiNormalizer,
    TurkicNormalizer,
    WhisperBasicNormalizer,
    WhisperEnglishNormalizer,
)


def test_whisper_basic_normalizer_uses_stock_multilingual_rules():
    normalizer = WhisperBasicNormalizer()

    assert normalizer.apply("Héllo, Straße!") == "héllo straße "
    assert normalizer.resolved_key == "whisper_basic"


def test_whisper_english_normalizer_uses_stock_number_and_currency_rules():
    normalizer = WhisperEnglishNormalizer()

    assert normalizer.apply("Twenty-one dollars and fifty cents.") == "$21.50"
    assert normalizer.resolved_key == "whisper_english"


def test_mark_preserving_normalizer_removes_markup_but_keeps_diacritics():
    normalizer = MarkPreservingNormalizer()

    assert normalizer.apply("ÉTÉ [noise] <laugh> (aside)") == "été"
    assert normalizer.apply("año") != normalizer.apply("ano")
    assert normalizer.resolved_key == "unicode_marks"


def test_turkic_normalizer_applies_locale_aware_casing():
    normalizer = TurkicNormalizer()

    assert normalizer.apply("İSTANBUL ISPARTA, ŞİİR!") == "istanbul ısparta şiir"
    assert normalizer.apply("I") != normalizer.apply("İ")
    assert normalizer.resolved_key == "turkic"


def test_cjk_normalizer_applies_nfkc_without_converting_script():
    normalizer = CjkNormalizer()

    assert normalizer.apply("學習，ｶﾞｸｾｲ！한글") == "學習 ガクセイ 한글"
    assert normalizer.resolved_key == "cjk"


def test_simplified_chinese_normalizer_converts_script():
    normalizer = SimplifiedChineseNormalizer()

    assert normalizer.apply("學習語言，臺灣") == "学习语言 台湾"
    assert normalizer.resolved_key == "chinese_t2s"


def test_arabic_normalizer_removes_harakat_and_folds_selected_variants():
    normalizer = ArabicNormalizer()

    assert normalizer.apply("أُسْتَاذة ـإلى") == "استاذة الي"
    assert normalizer.apply("ٱسم") == "اسم"
    assert normalizer.apply("مدرسة") != normalizer.apply("مدرسه")
    assert normalizer.resolved_key == "arabic"


@pytest.mark.parametrize(
    "control",
    [
        "\u00ad",  # soft hyphen
        "\u061c",  # Arabic letter mark
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\u202a",  # left-to-right embedding
        "\u202e",  # right-to-left override
        "\u2060",  # word joiner
        "\u2066",  # left-to-right isolate
        "\u2069",  # pop directional isolate
        "\ufeff",  # BOM / zero-width no-break space
    ],
)
def test_arabic_normalizer_strips_invisible_format_controls(control):
    assert ArabicNormalizer().apply(f"مر{control}حبا") == "مرحبا"


def test_arabic_normalizer_preserves_zero_width_space_as_a_word_boundary():
    assert ArabicNormalizer().apply("مرحبا\u200bبكم") == "مرحبا بكم"


@pytest.mark.parametrize(
    ("language", "text", "expected"),
    [
        ("hi", "क़िला\u200bयहाँ!", "क़िला यहाँ"),
        ("mr", "मुलां\u200dना!", "मुलांना"),
        ("bn", "ক\u09c7\u09be!", "কো"),
        ("ta", "க\u0bc6\u0bbe!", "கொ"),
        ("te", "మార్కెట్\u200cకి!", "మార్కెట్కి"),
    ],
)
def test_conservative_indic_normalizer_is_language_specific(language, text, expected):
    normalizer = ConservativeIndicNormalizer(language)

    assert normalizer.apply(text) == expected
    assert normalizer.resolved_key == f"indic_conservative_{language}"


def test_conservative_indic_normalizer_rejects_an_unsupported_language():
    with pytest.raises(ValueError, match="unsupported conservative Indic language"):
        ConservativeIndicNormalizer("gu")


def test_thai_normalizer_removes_thai_punctuation_but_preserves_tone_marks():
    normalizer = ThaiNormalizer()

    assert normalizer.apply("ภาษาไทยฯ TV! ป่า") == "ภาษาไทย tv ป่า"
    assert normalizer.apply("ปา") != normalizer.apply("ป่า")
    assert normalizer.apply("เร็วๆ") == "เร็วๆ"
    assert normalizer.resolved_key == "thai"


@pytest.mark.parametrize("mark", ["ฯ", "๏", "๚", "๛", "…"])
def test_thai_normalizer_removes_curated_and_unicode_punctuation(mark):
    assert ThaiNormalizer().apply(f"ภาษาไทย{mark}ทดสอบ") == "ภาษาไทย ทดสอบ"


@pytest.mark.parametrize("symbol", ["€", "£", "¥", "₹", "฿", "₺", "₩", "¢", "°", "§", "©"])
def test_thai_normalizer_strips_unicode_currency_and_miscellaneous_symbols(symbol):
    assert ThaiNormalizer().apply(f"ราคา{symbol}20") == "ราคา 20"


def test_regex_tag_remover_drops_tags_and_preserves_spoken_contents():
    normalizer = RegexTagRemover()

    assert normalizer.apply("one [noise] <laugh>two</laugh> three") == "one two three"
    assert normalizer.resolved_key == "regex_tags"


def test_annotation_remover_handles_braces_and_delegates_tag_cleanup():
    normalizer = AnnotationRemover()

    assert normalizer.apply("one {발음: 하나} {2} [noise] two") == "one 2 two"
    assert normalizer.resolved_key == "annotations"


def test_parenthesis_remover_handles_nested_unicode_parentheses_and_escapes():
    normalizer = ParenthesisRemover()

    assert " ".join(normalizer.apply("a （b (c) d） e").split()) == "a e"
    assert normalizer.apply(r"one\u00A0two") == "one\u00a0two"
    assert normalizer.resolved_key == "parentheses"


def test_benchmark_normalizer_composes_shared_cleanup_and_language_normalization():
    normalizer = DaiAsrI18nNormalizer(SimplifiedChineseNormalizer(), "zh")

    result = normalizer.apply("學 習（noise）{事件: 忽略}")
    assert isinstance(result, NormalizedText)
    assert result == "学习"
    assert normalizer.resolved_key == "chinese_t2s"
