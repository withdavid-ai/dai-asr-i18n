"""Synthetic equivalences and counterexamples for the exported normalizers."""

from __future__ import annotations

import inspect
import unicodedata

import pytest

import dai_asr_i18n
from dai_asr_i18n.normalization import (
    DAI_ASR_I18N_NORMALIZERS,
    WHISPER_BASELINE_NORMALIZERS,
    AnnotationRemover,
    CjkNormalizer,
    DaiAsrI18nNormalizer,
    NormalizedText,
    ParenthesisRemover,
    RegexTagRemover,
    SimplifiedChineseNormalizer,
    TextNormalizer,
    WhisperBasicNormalizer,
    WhisperEnglishNormalizer,
    normalization_manifest,
    normalize,
    normalize_with_metadata,
    normalizer_for_key,
    normalizer_for_language,
    prepare_scoring_text,
    strip_annotations,
)


def test_text_normalizer_is_an_abstract_contract():
    with pytest.raises(TypeError):
        TextNormalizer()


def test_simplified_chinese_normalizer_is_the_public_class_name():
    assert dai_asr_i18n.SimplifiedChineseNormalizer is SimplifiedChineseNormalizer
    assert "SimplifiedChineseNormalizer" in dai_asr_i18n.__all__
    assert not hasattr(dai_asr_i18n, "ChineseNormalizer")
    assert dai_asr_i18n.normalizer_for_key is normalizer_for_key
    assert "normalizer_for_key" in dai_asr_i18n.__all__


def test_profile_resolver_returns_class_based_normalizers():
    assert all(isinstance(key, str) for key in DAI_ASR_I18N_NORMALIZERS.values())
    assert all(isinstance(key, str) for key in WHISPER_BASELINE_NORMALIZERS.values())
    assert normalizer_for_key(DAI_ASR_I18N_NORMALIZERS["es"]) is normalizer_for_key(DAI_ASR_I18N_NORMALIZERS["fr"])
    assert normalizer_for_key(DAI_ASR_I18N_NORMALIZERS["ja"]) is normalizer_for_key(DAI_ASR_I18N_NORMALIZERS["ko"])
    assert normalizer_for_key(WHISPER_BASELINE_NORMALIZERS["es"]) is normalizer_for_key(
        WHISPER_BASELINE_NORMALIZERS["zh"]
    )
    assert isinstance(normalizer_for_language("es", "whisper_baseline"), WhisperBasicNormalizer)
    assert isinstance(normalizer_for_language("en", "whisper_baseline"), WhisperEnglishNormalizer)

    chinese = normalizer_for_language("zh", "dai_asr_i18n")
    assert isinstance(chinese, DaiAsrI18nNormalizer)
    assert isinstance(chinese.inner, SimplifiedChineseNormalizer)
    assert chinese.resolved_key == "chinese_t2s"


def test_policy_maps_are_read_only():
    with pytest.raises(TypeError):
        DAI_ASR_I18N_NORMALIZERS["en"] = "whisper_basic"  # type: ignore[index]
    with pytest.raises(TypeError):
        WHISPER_BASELINE_NORMALIZERS["en"] = "whisper_basic"  # type: ignore[index]


def test_key_resolver_canonicalizes_call_shapes_to_one_cached_instance():
    assert normalizer_for_key("cjk") is normalizer_for_key(key="cjk")
    with pytest.raises(TypeError, match="key must be a string"):
        normalizer_for_key(["cjk"])  # type: ignore[arg-type]


def test_language_key_maps_preserve_the_published_normalizer_routing():
    expected_benchmark = {
        "ar": "arabic",
        "bn": "indic_conservative_bn",
        "de": "unicode_marks",
        "en": "whisper_english",
        "es": "unicode_marks",
        "fr": "unicode_marks",
        "hi": "indic_conservative_hi",
        "id": "unicode_marks",
        "it": "unicode_marks",
        "ja": "cjk",
        "ko": "cjk",
        "mr": "indic_conservative_mr",
        "pt": "unicode_marks",
        "ru": "unicode_marks",
        "ta": "indic_conservative_ta",
        "te": "indic_conservative_te",
        "th": "thai",
        "tl": "unicode_marks",
        "tr": "turkic",
        "vi": "unicode_marks",
        "zh": "chinese_t2s",
    }
    expected_whisper = {
        language: "whisper_english" if language == "en" else "whisper_basic" for language in DAI_ASR_I18N_NORMALIZERS
    }
    assert DAI_ASR_I18N_NORMALIZERS == expected_benchmark
    assert WHISPER_BASELINE_NORMALIZERS == expected_whisper
    for key in {*expected_benchmark.values(), *expected_whisper.values()}:
        assert normalizer_for_key(key).resolved_key == key

    manifest = normalization_manifest(include_opencc_hashes=False)
    assert manifest["dai_asr_i18n_normalizers"] == expected_benchmark
    assert manifest["whisper_baseline_normalizers"] == expected_whisper


@pytest.mark.parametrize("key", ["not-a-normalizer", "indic_conservative_gu"])
def test_unknown_normalizer_key_is_rejected(key: str):
    with pytest.raises(ValueError, match="unknown normalizer key"):
        normalizer_for_key(key)


def test_simplified_chinese_normalizer_always_converts_script():
    simplified = SimplifiedChineseNormalizer()

    assert isinstance(simplified, CjkNormalizer)
    assert simplified.apply("學習語言") == "学习语言"
    assert simplified.resolved_key == "chinese_t2s"
    assert not inspect.signature(SimplifiedChineseNormalizer).parameters


def test_simplified_chinese_inherits_the_cjk_cleanup_pipeline():
    assert SimplifiedChineseNormalizer().apply("Ａ，學！") == "a 学"


def test_normalizers_are_callable_as_well_as_explicitly_applied():
    normalizer = WhisperBasicNormalizer()
    assert normalizer("HELLO") == normalizer.apply("HELLO") == "hello"


def test_benchmark_normalized_text_is_not_normalized_twice():
    normalizer = normalizer_for_language("en")
    once = normalizer.apply("double oh seven")
    assert isinstance(once, NormalizedText)
    assert once == "007"
    assert normalizer.apply(once) is once
    assert normalize(once, "en") is once


def test_benchmark_rejects_normalized_text_from_another_profile():
    normalized = normalizer_for_language("en").apply("double oh seven")
    with pytest.raises(ValueError, match="another profile"):
        normalizer_for_language("es").apply(normalized)


def test_benchmark_marker_distinguishes_custom_cleanup_components():
    standard = normalizer_for_language("en")
    custom = DaiAsrI18nNormalizer(standard.inner, "en", parenthesis_remover=RegexTagRemover())
    normalized = custom.apply("double oh seven")
    with pytest.raises(ValueError, match="another profile"):
        standard.apply(normalized)


def test_regex_tag_remover_keeps_spoken_sgml_contents():
    remover = RegexTagRemover()
    assert remover.apply("one [noise] <laughing>two</laughing> three") == "one two three"
    assert remover.apply("one {PRO: two} three") == "one {PRO: two} three"


def test_annotation_remover_handles_braces_and_delegates_tags():
    remover = AnnotationRemover(RegexTagRemover())
    assert remover.apply("one [noise] {발음: 둘} <em>two</em> three") == "one two three"
    assert remover.apply("one {2} three") == "one 2 three"


def test_parenthesis_remover_handles_nested_unicode_and_allowed_escapes():
    remover = ParenthesisRemover()
    assert " ".join(remover.apply("a （b (c) d） e").split()) == "a e"
    assert remover.apply(r"one\\u00A0two") == "one\u00a0two"
    assert " ".join(remover.apply("a (b c").split()) == "a b c"


@pytest.mark.parametrize(
    "language,left,right",
    [
        ("en", "twenty one", "21"),
        ("es", "NIÑO", "nin\u0303o"),
        ("fr", "ÉTÉ", "E\u0301TE\u0301"),
        ("de", "GRÖẞE", "größE"),
        ("it", "PERCHÉ", "perche\u0301"),
        ("pt", "AÇÃO", "ac\u0327a\u0303o"),
        ("ru", "ЁЛКА", "е\u0308лка"),
        ("id", "SELAMAT", "selamat"),
        ("tl", "KUMUSTA", "kumusta"),
        ("vi", "VIỆT", "VIE\u0323\u0302T"),
        ("tr", "İSTANBUL ISPARTA", "istanbul ısparta"),
        ("ar", "مَدْرَسَة", "مدرسة"),
        ("hi", "क़िला", "क़िला"),
        ("mr", "मुलां\u200dना", "मुलांना"),
        ("bn", "ক\u09c7\u09be", "কো"),
        ("ta", "க\u0bc6\u0bbe", "கொ"),
        ("te", "మార్కెట్\u200cకి", "మార్కెట్కి"),
        ("zh", "學習語言", "学习语言"),
        ("ja", "ｶﾞｸｾｲ", "ガクセイ"),
        ("ko", "한글", "한글"),
        ("th", "ภาษาไทย TV", "ภาษาไทย tv"),
    ],
)
def test_benchmark_intended_equivalence(language, left, right):
    assert normalize(left, language) == normalize(right, language)


@pytest.mark.parametrize(
    "language,left,right",
    [
        ("es", "año", "ano"),
        ("fr", "ou", "où"),
        ("de", "Straße", "Strasse"),
        ("ru", "все", "всё"),
        ("vi", "ma", "má"),
        ("tr", "ı", "i"),
        ("ar", "مدرسة", "مدرسه"),
        ("hi", "क", "क़"),
        ("ta", "க", "கா"),
        ("te", "క", "కా"),
        ("zh", "滑鼠", "鼠标"),
        ("th", "ปา", "ป่า"),
    ],
)
def test_benchmark_meaningful_distinctions_survive(language, left, right):
    assert normalize(left, language) != normalize(right, language)


def test_shared_policy_removes_annotations_and_unicode_parentheses():
    assert normalize("hello [chuckle] {PRO: five} <laughing>world</laughing>", "es") == "hello world"
    assert normalize("مرحبا ﴿تعليق﴾ عالم", "ar") == "مرحبا عالم"
    assert normalize("你好（音乐）世界", "zh") == "你好世界"
    assert "music" not in normalize("hello (short music) world", "en")


def test_nested_and_unmatched_parentheses_are_deterministic():
    assert " ".join(prepare_scoring_text("a (b (c) d) e").split()) == "a e"
    assert " ".join(prepare_scoring_text("a (b c").split()) == "a b c"


def test_annotation_labels_work_in_any_script_and_numeric_braces_are_unwrapped():
    assert strip_annotations("one {발음: 하나} two") == "one two"
    assert strip_annotations("one {1} two") == "one 1 two"


def test_mark_preserving_normalizer_accepts_canonical_equivalents():
    raw = "q\u0301 क़ ภาษาไทย"
    assert normalize(raw, "es") == unicodedata.normalize("NFKC", raw)


def test_whisper_baseline_has_explicit_lineage_and_skips_benchmark_wrapper():
    benchmark = normalize_with_metadata("hello {PRO: five} world", "en", "dai_asr_i18n")
    whisper = normalize_with_metadata("hello {PRO: five} world", "en", "whisper_baseline")
    assert benchmark.profile == "dai_asr_i18n"
    assert whisper.profile == "whisper_baseline"
    assert benchmark.policy != whisper.policy
    assert "pro" not in benchmark.text
    assert "pro" in whisper.text


def test_unknown_profile_is_rejected_instead_of_silently_falling_back():
    with pytest.raises(ValueError, match="unknown normalization profile"):
        normalize("學習", "zh", "unknown")


def test_manifest_versions_normalization_scoring_and_implementation():
    manifest = normalization_manifest(include_opencc_hashes=False)
    assert manifest["policy"] == "dai-asr-i18n-normalization-v9"
    assert manifest["scoring_policy"] == "dai-asr-i18n-offline-scoring-v3"
    assert manifest["metric_policy"]["version"] == "benchmark-primary-v1"
    assert manifest["metric_policy"]["public_metric"]["zh"] == "cer"
    assert manifest["diarization_policy"] == {
        "der": "dai-asr-i18n-der-v9",
        "jer": "dai-asr-i18n-jer-v3",
        "speaker_count": "dai-asr-i18n-speaker-count-v4",
    }
    assert len(manifest["dai_asr_i18n"]["policy_implementation_sha256"]) == 64
    assert manifest["packages"]["more-itertools"] == "11.1.0"
    assert manifest["packages"]["rapidfuzz"] == "3.14.6"
    assert manifest["resource_hashes_included"] is False
