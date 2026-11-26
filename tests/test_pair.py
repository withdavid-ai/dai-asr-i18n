"""Pair steps: the hypothesis takes the reference's spelling, the reference never changes, and no error is added."""

from __future__ import annotations

import pytest

from dai_asr_i18n.normalization import normalize, normalize_pair
from dai_asr_i18n.normalization.pair import PairStep, Rewrite, reconcile
from dai_asr_i18n.normalization.tokens import align, word_tokens

# (language, reference, hypothesis, hypothesis as scored)
EXAMPLES = [
    ("hi", "मेरा कंप्यूटर खराब है", "मेरा computer खराब है", "मेरा कंप्यूटर खराब है"),
    ("hi", "मेरा computer खराब है", "मेरा कंप्यूटर खराब है", "मेरा computer खराब है"),
    ("bn", "এট লিস্ট একবার", "at least একবার", "এট লিস্ট একবার"),
    ("ja", "コンピューターが壊れた", "computerが壊れた", "コンピューターが壊れた"),
    ("ko", "오케이 좋아요", "okay 좋아요", "오케이 좋아요"),
    ("ru", "это окей", "это okay", "это окей"),
    # two spellings in one script are a spelling difference, not a script choice
    ("hi", "मेरा कंप्यूटर", "मेरा कम्प्यूटर", "मेरा कम्प्यूटर"),
    # a reference that never writes the word in the other script leaves the model's word alone
    ("hi", "मेरा फ़ोन", "मेरा computer", "मेरा computer"),
    # apostrophes: German, Tagalog and Turkish
    ("de", "aus'm Haus", "ausm Haus", "aus'm haus"),
    ("de", "son Tag", "so'n Tag", "son tag"),
    ("tl", "sa'yo na", "sayo na", "sa'yo na"),
    ("tr", "Türkçe'yi biliyorum", "Türkçeyi biliyorum", "türkçe'yi biliyorum"),
    # only the stretch that differs by an apostrophe takes the reference's spelling; the spacing stays
    ("tl", "ka agad sayo", "kaagad sa'yo", "kaagad sayo"),
    # no pair step in this language
    ("es", "la computadora", "la computer", "la computer"),
]


@pytest.mark.parametrize("language,reference,hypothesis,expected", EXAMPLES)
def test_pair_examples(language, reference, hypothesis, expected):
    pair = normalize_pair(reference, hypothesis, language)
    assert pair.reference == normalize(reference, language)
    assert pair.hypothesis == expected
    assert bool(pair.rewrites) == (expected != normalize(hypothesis, language))


def _errors(reference: str, hypothesis: str, language: str) -> tuple[int, int]:
    ref, hyp = word_tokens(reference, language), word_tokens(hypothesis, language)
    chars = lambda tokens: [char for token in tokens for char in token]  # noqa: E731
    return align(ref, hyp)[1], align(chars(ref), chars(hyp))[1]


@pytest.mark.parametrize("language,reference,hypothesis,expected", EXAMPLES)
def test_pair_steps_never_add_errors_and_are_idempotent(language, reference, hypothesis, expected):
    pair = normalize_pair(reference, hypothesis, language)
    before = _errors(pair.reference, normalize(hypothesis, language), language)
    after = _errors(pair.reference, pair.hypothesis, language)
    assert after[0] <= before[0] and after[1] <= before[1]
    assert normalize_pair(reference, pair.hypothesis, language).hypothesis == pair.hypothesis


@pytest.mark.parametrize(
    "language,reference,hypothesis",
    [
        ("hi", "मेरा कंप्यूटर खराब है", "मेरा computer खराब है"),
        ("ja", "ラインのスタンプを買った", "LINEのスタンプを買った"),
        ("th", "เขาจะมีบริการ ฟรี", "เขาจะมีบริการ free"),
        ("de", "aus'm Haus", "ausm Haus"),
    ],
)
def test_a_rewrite_can_be_read_on_its_own(language, reference, hypothesis):
    pair = normalize_pair(reference, hypothesis, language)
    assert pair.rewrites
    for rewrite in pair.rewrites:
        assert rewrite.language == language
        assert rewrite.reference_text == pair.reference
        assert rewrite.reference_tokens()[rewrite.ref_start : rewrite.ref_end] == list(rewrite.reference)
        assert rewrite.hypothesis_tokens()[rewrite.hyp_start : rewrite.hyp_end] == list(rewrite.hypothesis)
    first = pair.rewrites[0]
    assert first.hypothesis_text == normalize(hypothesis, language)
    assert first.tokenization == ("cjk_character" if language == "ja" else "word")


def test_a_rewrite_that_would_add_an_error_is_declined():
    def swap_everything(ref, hyp, ops, language):
        return [Rewrite("swap", "test", 0, len(ref), 0, len(hyp), tuple(ref), tuple(hyp))] if ref and hyp else []

    step = PairStep("swap", swap_everything, name="Swap", description="Copies the whole reference.", anywhere=True)
    # copying the reference never adds an error, so it is kept
    kept = reconcile("a b c", "a x c", "en", [step])
    assert kept.hypothesis == "a b c"
    assert [rewrite.step for rewrite in kept.rewrites] == ["swap"]

    def copy_the_last_word_first(ref, hyp, ops, language):
        return [Rewrite("noise", "test", len(ref) - 1, len(ref), 0, 1, (ref[-1],), (hyp[0],))]

    noisy = PairStep("noise", copy_the_last_word_first, name="Noise", description="Copies a word.", anywhere=True)
    declined = reconcile("a b", "a b", "en", [noisy])
    assert declined.hypothesis == "a b"
    assert declined.rewrites == ()
    assert [rewrite.step for rewrite in declined.declined] == ["noise"]


def test_a_rewrite_outside_the_alignment_errors_is_rejected():
    def rewrite_a_hit(ref, hyp, ops, language):
        return [Rewrite("bad", "test", 0, 1, 0, 1, (ref[0],), (hyp[0],))]

    step = PairStep("bad", rewrite_a_hit, name="Bad", description="Rewrites a correct word.")
    with pytest.raises(ValueError, match="outside the alignment"):
        reconcile("a b", "a c", "en", [step])
