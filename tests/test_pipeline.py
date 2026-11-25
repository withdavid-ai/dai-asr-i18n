"""The benchmark pipeline: one example per step, idempotence, and step order within a stage."""

from __future__ import annotations

from itertools import permutations

import pytest

from dai_asr_i18n.languages import BENCHMARK_LANGUAGES
from dai_asr_i18n.normalization import base_normalizer, normalize
from dai_asr_i18n.normalization.pipeline import BASE, LATE, PAIR_STEPS, STEPS, DaiAsrI18nNormalizer

# (language, text, normalized)
EXAMPLES = [
    # event parentheses
    ("en", "(laughs) Call (02) 8012 3456", "call 02 8012 3456"),
    ("es", "sí (risas) claro", "sí claro"),
    ("ja", "はい（笑）そうです", "はいそうです"),
    ("fr", "il a dit (quelque chose) hier", "il a dit quelque chose hier"),
    # apostrophes and punctuation
    ("fr", "L’homme qu’il a vu", "l'homme qu'il a vu"),
    ("it", "un'altra volta", "un altra volta"),
    ("en", "Rock-’n’-roll, isn’t it?", "rock n roll isn't it"),
    ("de", "Wie geht's? Gut, danke!", "wie geht's gut danke"),
    ("ko", "만족감'이라고도 했어요", "만족감이라고도 했어요"),
    ("te", "రెడ్డి'లో phone'లో", "రెడ్డి లో phone లో"),
    ("id", "benar-benar wi-fi", "benar benar wi fi"),
    # spaced clitics
    ("en", "I 'm sure it 's fine", "i'm sure it's fine"),
    ("fr", "c 'est la vie", "c'est la vie"),
    # Russian ё
    ("ru", "Ещё всё", "еще все"),
    # Thai marks
    ("th", "เเปลก", "แปลก"),
    ("th", "ก่ี", "กี่"),
    # Japanese characters and long vowels
    ("ja", "コーヒー〜♪", "コーヒー"),
    ("ja", "コーーヒー", "コーヒー"),
    ("ja", "〒100", "〒100"),
    # native digits
    ("hi", "३ लोग", "3 लोग"),
    ("ar", "٣ أيام", "3 ايام"),
    ("th", "๓ คน", "3 คน"),
    # English: contractions and fillers stay, leading zeros stay, numbers are read as Whisper reads them
    ("en", "Don't you wanna go? Gonna be fun.", "don't you wanna go gonna be fun"),
    ("en", "Uh, mm-hmm, um", "uh mhm um"),
    ("en", "My code is 0556, and 007.", "my code is 0556 and 007"),
    ("en", "twenty-one dollars and fifty cents", "$21.50"),
    ("en", "five oh five", "5 oh 5"),
    ("en", "five pounds", "5 pounds"),
    ("en", "an N95 mask", "an n95 mask"),
    ("en", "The colour of the theatre", "the color of the theater"),
]


@pytest.mark.parametrize("language,text,expected", EXAMPLES)
def test_step_examples(language, text, expected):
    assert normalize(text, language) == expected


# Inputs that touch every step in some language.
CORPUS = [
    "(laughs) Call (02) 8012, it’s “fine” — I 'm sure! 0556 rock-’n’-roll 50% $3.50",
    "c 'est l’homme (rires) qu'il a vu, un'altra volta; wa- este- este",
    "Ещё ЁЛКА — ё̈ Ё",
    "เเปลก ก่ี ถ ิ ่น ํา ๓ ฯ เ'เ",
    "コーーヒー〜♪ ｶﾞｸｾｲ（笑）ー{ー 'テスト'です 〒100",
    "३ लोग फ़ोन'वर रेड्डी'लो (हँसी) ٣ أيام",
    "İSTANBUL'DA türkçe'yi ŞİİR, ısparta!",
    "學 習 語 言（音乐）, 你好！",
    "만족감'이라고도 했어요 (웃음) mcdonald's",
]


@pytest.mark.parametrize("language", BENCHMARK_LANGUAGES)
def test_normalizing_an_output_again_changes_nothing(language):
    for text in CORPUS:
        once = normalize(text, language)
        assert normalize(once, language) == once


@pytest.mark.parametrize("language", BENCHMARK_LANGUAGES)
def test_steps_of_one_stage_run_in_any_order(language):
    normalizer = DaiAsrI18nNormalizer(language)
    groups = normalizer.stage_groups()
    for index, group in enumerate(groups):
        if len(group) < 2:
            continue
        for order in permutations(group):
            full = [name for g in groups[:index] for name in g] + list(order)
            full += [name for g in groups[index + 1 :] for name in g]
            for text in CORPUS:
                assert normalizer.normalize(text, full) == normalizer.normalize(text)


def test_stage_groups_place_the_base_and_english_late_stage():
    english = DaiAsrI18nNormalizer("en").stage_groups()
    assert english[0] == ("apostrophes", "spaced-clitics")
    assert english[1] == (BASE,)
    assert english[-1] == (LATE,)
    assert all(LATE not in group for group in DaiAsrI18nNormalizer("fr").stage_groups())
    assert ("japanese-long-vowels",) in DaiAsrI18nNormalizer("ja").stage_groups()


def test_order_must_name_every_stage_once():
    with pytest.raises(ValueError, match="order must name"):
        DaiAsrI18nNormalizer("en").normalize("hello", ["base"])


# Text no step's trigger matches, in every script of the benchmark.
PLAIN = [
    "Hello world, this is fine.",
    "Bonjour [bruit] tout le monde {PRO: un} <rire>là</rire>.",
    "Merhaba DÜNYA, İyi günler!",
    "مَرْحَبًا بِكُمْ، أهلاً",
    "नमस्ते, आप कैसे हैं।",
    "নমস্কার, কেমন আছেন।",
    "வணக்கம், எப்படி இருக்கிறீர்கள்.",
    "నమస్కారం, ఎలా ఉన్నారు.",
    "學習語言，你好！",
    "こんにちは、元気です。",
    "안녕하세요, 반갑습니다!",
    "สวัสดีครับ, ภาษาไทย!",
    "Xin chào, các bạn!",
    "Привет, мир!",
]


@pytest.mark.parametrize("language", [language for language in BENCHMARK_LANGUAGES if language != "en"])
def test_benchmark_equals_the_base_normalizer_where_no_trigger_matches(language):
    steps = DaiAsrI18nNormalizer(language).steps
    for text in PLAIN:
        assert not any(step.trigger.search(text) for step in steps), text
        assert normalize(text, language) == " ".join(base_normalizer(language).apply(text).split())


def test_steps_and_pair_steps_have_unique_ids_and_descriptions():
    ids = [step.id for step in (*STEPS, *PAIR_STEPS)]
    assert len(ids) == len(set(ids))
    assert all(step.name and step.description for step in (*STEPS, *PAIR_STEPS))
