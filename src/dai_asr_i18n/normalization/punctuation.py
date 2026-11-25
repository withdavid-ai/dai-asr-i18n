"""Punctuation: strip punctuation, keeping word-internal apostrophes; every hyphen becomes a space.

"hello, world!" becomes "hello world", "l'homme" stays, and "benar-benar" becomes "benar benar". This
step replaces each language transform's own punctuation pass, which removes every apostrophe and with
it the French and Italian elision.

- An apostrophe (' or any look-alike the apostrophe step unifies) stays when a letter is on both sides.
  In Korean, Japanese, Chinese and Thai it is deleted unless a Latin letter is on both sides: these
  scripts have no word-internal apostrophe, so it is a quote mark ("만족감'이라고도" becomes "만족감이라고도",
  "'テスト'です" becomes "テストです"), while a Latin word keeps it ("let's", "mcdonald's", as the
  transliteration maps spell it). CER, the metric there, ignores spaces.
- A clitic written with a space before its apostrophe is attached first, by its own step before the base
  stage (:func:`attach_spaced_clitics`): in English a contraction ending ("I 'm" becomes "I'm"), in
  French and Italian the apostrophe after an elision particle ("c 'est" becomes "c'est").
- In Hindi, Marathi, Bengali, Tamil and Telugu an apostrophe next to a letter of the language's script is
  a space: it marks a suffix boundary ("రెడ్డి'లో" becomes "రెడ్డి లో", "phone'లో" becomes "phone లో"), which
  transcripts write either way. Inside a Latin word it stays ("mcdonald's").
- Italian "un'" before a vowel is "un " ("un'altro" becomes "un altro"): the article is written either
  way, so the apostrophe is not a word-internal one there.
- Every hyphen (-, or the variants ‐ ‑ － ﹣) becomes a space: inside a word ("benar-benar", "wi-fi"), at
  a cut-off ("wa-", "este- este"), in a trailing-off marker ("It's--"), and between digits ("covid-19",
  "3-4"), in every language.
- Every other character the language's base punctuation pass removes becomes a space: Unicode P* and S*
  for most languages, plus ฯ for Thai, and for Turkish everything but letters, digits, underscores and
  spaces.
- Characters another step owns are left to it: parentheses (event parentheses), and in Japanese the
  characters the Japanese step deletes or keeps (the hyphen look-alikes ‐ ‑ among them). Disjoint
  ownership is what lets the steps of one stage run in any order.
- English punctuation other than apostrophes and hyphens stays with Whisper's rules, which keep the
  symbols they read as numbers ($21, 50%, 3.5) and remove the rest.

A letter is any Unicode letter or mark (category L or M), so a vowel sign ends a word as its base letter
does. Looking for the letters on either side of an apostrophe, the step looks through characters another
step deletes (in Japanese ★ and 〜, in Thai zero-width spaces): "コ★'ン" keeps its apostrophe whether the
★ is deleted yet or not.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable

import regex

from dai_asr_i18n.normalization.characters import APOSTROPHES, JAPANESE_DELETED, JAPANESE_KEPT, PARENTHESES

APOSTROPHE_LIKE = "'" + APOSTROPHES
HYPHENS = "-‐‑－﹣"
_APOSTROPHE_CLASS = f"[{regex.escape(APOSTROPHE_LIKE)}]"
_LETTER = r"[\p{L}\p{M}]"
# A clitic written with a space before its apostrophe ("I 'm", "c 'est"): the space goes.
_SPACED_CLITIC = {
    "en": regex.compile(
        rf"(?<={_LETTER})\s+(?={_APOSTROPHE_CLASS}(?:s|m|re|ve|ll|d|t)(?!{_LETTER}))", regex.IGNORECASE
    ),
    **dict.fromkeys(
        ("fr", "it"),
        regex.compile(
            rf"(?<!{_LETTER})(c|d|j|l|m|n|s|t|qu|jusqu|lorsqu|puisqu)\s+(?={_APOSTROPHE_CLASS}{_LETTER})",
            regex.IGNORECASE,
        ),
    ),
}
# Italian "un'" before a vowel: the article, written with or without the apostrophe.
_ITALIAN_UN = regex.compile(rf"(?<!{_LETTER})(un){_APOSTROPHE_CLASS}(?=[aeiouàèéìíîòóùú])", regex.IGNORECASE)
CLITIC_LANGUAGES = frozenset(_SPACED_CLITIC)
CLITIC_TRIGGER = regex.compile(rf"\s{_APOSTROPHE_CLASS}")


def attach_spaced_clitics(text: str, language: str | None = None) -> str:
    """Attach a clitic written with a space before its apostrophe: "I 'm" becomes "I'm", "c 'est" "c'est".

    Runs before the base stage, with the apostrophe step: there no step has yet turned a character into
    the space before an apostrophe ("c⁾'est" is not "c 'est", whether or not the parenthesis is removed).
    """

    clitic = _SPACED_CLITIC.get(language)
    return text if clitic is None else clitic.sub(lambda match: match.group(1) if match.groups() else "", text)


# Scripts with no word-internal apostrophe: there it is a quote mark, and is deleted, except inside a
# Latin word ("mcdonald's", which the transliteration maps spell with it).
APOSTROPHE_DELETED = frozenset({"ko", "ja", "zh", "th"})
# An apostrophe next to a letter of the language's script marks a suffix boundary: a space.
APOSTROPHE_SPACED = frozenset({"hi", "mr", "bn", "ta", "te"})


def _latin(char: str) -> bool:
    return unicodedata.name(char, "").startswith("LATIN")


# Left to the step that owns them.
_OWNED = {"ja": PARENTHESES + JAPANESE_DELETED + JAPANESE_KEPT}
# Deleted by another step of the language, so invisible when looking for a neighboring letter.
_TRANSPARENT = {"ja": JAPANESE_DELETED, "th": "​‌"}
# Where this step and the base punctuation pass can differ: an apostrophe after a character (kept between
# letters outside APOSTROPHE_SPACED; deleted in APOSTROPHE_DELETED, where the base pass leaves a space),
# and 〒 in Japanese. A hyphen is a space in both.
TRIGGER = regex.compile(f"\\S[{regex.escape(APOSTROPHE_LIKE)}]|〒")


def _punctuation_or_symbol(char: str) -> bool:
    return unicodedata.category(char)[0] in "PS"


# Each language's base punctuation pass, as a test on one character (see the language transforms'
# ``punctuation``); P* and S* unless listed.
_BASE_REMOVES: dict[str | None, Callable[[str], bool]] = {
    "tr": lambda char: not (char.isalnum() or char == "_" or char.isspace()),  # TurkicNormalizer: [^\w\s]
    "th": lambda char: char == "ฯ" or _punctuation_or_symbol(char),  # ThaiNormalizer: ฯ, then P/S
}


def base_removes(language: str | None) -> Callable[[str], bool]:
    """What the language's base punctuation pass removes, one character at a time."""

    return _BASE_REMOVES.get(language, _punctuation_or_symbol)


def _never(char: str) -> bool:
    return False


def _is_letter(char: str) -> bool:
    return bool(char) and char not in APOSTROPHE_LIKE and unicodedata.category(char)[0] in "LM"


def _neighbor(text: str, index: int, step: int, transparent: str) -> str:
    index += step
    while 0 <= index < len(text) and text[index] in transparent:
        index += step
    return text[index] if 0 <= index < len(text) else ""


def strip_punctuation(text: str, language: str | None = None) -> str:
    """Remove punctuation as the module docstring describes."""

    owned = _OWNED.get(language, PARENTHESES)
    transparent = _TRANSPARENT.get(language, "")
    # English leaves everything but apostrophes and hyphens to Whisper's rules, which keep $ % and the
    # decimal point for numbers and remove the rest.
    strips = _never if language == "en" else base_removes(language)
    if language == "it":
        text = _ITALIAN_UN.sub(r"\1 ", text)
    if language == "tr":
        # Composed first, as TurkicNormalizer composes before its punctuation pass: that pass removes
        # leftover combining marks, and the dot of a decomposed İ must not be one of them.
        text = unicodedata.normalize("NFC", text)

    def kept_letter(char: str) -> bool:
        # A letter this pass removes (Thai ฯ, a Turkish mark) is no neighbor: a second pass would see the
        # space it leaves, and the step must be idempotent.
        return _is_letter(char) and not strips(char)

    out = []
    for index, char in enumerate(text):
        before, after = _neighbor(text, index, -1, transparent), _neighbor(text, index, 1, transparent)
        if char in owned:
            out.append(char)
        elif char in APOSTROPHE_LIKE:
            keep = kept_letter(before) and kept_letter(after)
            if language in APOSTROPHE_DELETED:
                if keep and _latin(before) and _latin(after):  # inside a Latin word: "mcdonald's"
                    out.append(char)
                continue
            if keep and language in APOSTROPHE_SPACED:
                keep = _latin(before) and _latin(after)  # a suffix boundary: a native letter on a side
            out.append(char if keep else " ")
        elif char in HYPHENS or strips(char):
            out.append(" ")
        else:
            out.append(char)
    return "".join(out)


__all__ = ["attach_spaced_clitics", "base_removes", "strip_punctuation"]
