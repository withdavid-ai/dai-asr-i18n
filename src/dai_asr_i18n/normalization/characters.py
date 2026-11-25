"""Character steps: apostrophes, Russian ё, Thai mark order, Japanese characters, and native digits."""

from __future__ import annotations

import re
import unicodedata

import regex

from dai_asr_i18n.normalization.cleanup import PARENTHESIS_TRANSLATION

# Owned by the event-parentheses step: ASCII parentheses and every Unicode variant folded to them.
PARENTHESES = "()" + "".join(chr(code) for code in PARENTHESIS_TRANSLATION)

# Apostrophe look-alikes, plus the fullwidth apostrophe, which NFKC maps to ' anyway.
APOSTROPHES = "’‘ʼ´`′＇"
APOSTROPHE_TRIGGER = re.compile(f"[{APOSTROPHES}]")
_TO_APOSTROPHE = str.maketrans(dict.fromkeys(APOSTROPHES, "'"))


def unify_apostrophes(text: str, language: str | None = None) -> str:
    """Map ’ ‘ ʼ ´ ` ′ to the straight apostrophe: "l’homme" becomes "l'homme"."""

    return text.translate(_TO_APOSTROPHE)


class ChangedBy:
    """A trigger for a rule no pattern describes: it fires where the rule changes the text."""

    def __init__(self, rule) -> None:
        self.rule = rule

    def search(self, text: str) -> bool:
        return self.rule(text) != text


# Russian: ё becomes е, composed or decomposed, in either case.
_COMBINING_DIAERESIS = "\u0308"
RUSSIAN_TRIGGER = re.compile(f"[ёЁ]|[еЕ]{_COMBINING_DIAERESIS}")
# Any further diaeresis goes too: "ё" + U+0308 would otherwise leave "е" + U+0308 for a second pass.
_RUSSIAN_YO = re.compile(f"ё{_COMBINING_DIAERESIS}*|е{_COMBINING_DIAERESIS}+")
_RUSSIAN_CAPITAL_YO = re.compile(f"Ё{_COMBINING_DIAERESIS}*|Е{_COMBINING_DIAERESIS}+")


def fold_russian_yo(text: str, language: str | None = None) -> str:
    """Write ё as е: "ещё" becomes "еще"."""

    return _RUSSIAN_CAPITAL_YO.sub("Е", _RUSSIAN_YO.sub("е", text))


# Thai: one order for vowel and tone marks, and nothing else. Every mark and every repeated vowel stays
# ("่าาาาาา", "นานาาา", and a mark at the start of a word, "๋อครับ", are what was written). The rules are the
# reordering rules of PyThaiNLP's ``reorder_vowels``, without its follow-vowel/tone-mark swap (which moves a
# mark across a vowel: "อ่า่"), its ๅ -> า, and the rest of its ``normalize``, whose dangling-mark and
# repeated-vowel removal loses marks ("ถ ิ ่น" becomes "ถิ น").
_THAI_COMBINING = "\u0e31\u0e34-\u0e3a\u0e47-\u0e4e"  # vowels above and below, tone marks, signs
_THAI_TONES = "\u0e48-\u0e4b"
_THAI_ZERO_WIDTH = "\u200b\u200c"  # deleted; the punctuation step looks through them
_SPACE_BEFORE_MARK = re.compile(f"\\s+(?=[{_THAI_COMBINING}])")
_THAI_REORDER = (
    (re.compile("\u0e40\u0e40"), "\u0e41"),  # sara e + sara e -> sara ae: "เเปลก" -> "แปลก"
    # tone mark or thanthakhat before a vowel above or below -> after it: "ก่ี" -> "กี่"
    (re.compile(f"([{_THAI_TONES}\u0e4c]+)([\u0e31\u0e34-\u0e39\u0e47\u0e4d]+)"), r"\2\1"),
    (re.compile(f"\u0e4d([{_THAI_TONES}]*)\u0e32"), "\\1\u0e33"),  # nikhahit (+ tone) + sara aa -> (tone +) sara am
)
_DELETE_THAI_ZERO_WIDTH = str.maketrans(dict.fromkeys(_THAI_ZERO_WIDTH))


def reorder_thai_marks(text: str, language: str | None = None) -> str:
    """Put Thai marks in one order: "เเปลก" (two sara e) becomes "แปลก", "ํา" becomes "ำ", "ถ ิ ่น" becomes "ถิ่น".

    A space before a combining mark is removed, so the mark joins the character before it. Marks and
    repeated vowels are never removed.
    """

    text = _SPACE_BEFORE_MARK.sub("", text.translate(_DELETE_THAI_ZERO_WIDTH))
    while True:  # to the fixed point: one swap can expose another ("่ิ่ิ")
        before = text
        for pattern, replacement in _THAI_REORDER:
            text = pattern.sub(replacement, text)
        if text == before:
            return text


THAI_TRIGGER = ChangedBy(reorder_thai_marks)

# Japanese. The deleted set is closed under NFKC: NFKC maps ～ to ~ and ‑ to ‐, so both images are
# deleted too, and the result is the same before or after the base stage's own NFKC.
JAPANESE_DELETED = "〜～~‐‑♪☆★○●◎♥"
JAPANESE_KEPT = "〒"
JAPANESE_TRIGGER = re.compile(f"[{re.escape(JAPANESE_DELETED)}]")
# Two ー with nothing but punctuation, symbols or spaces between: the punctuation step can join them.
LONG_VOWEL_TRIGGER = regex.compile(r"[ーｰ][^\p{L}\p{N}]*[ーｰ]")
_JAPANESE_DELETE = str.maketrans(dict.fromkeys(JAPANESE_DELETED))
# An apostrophe without a Latin letter on both sides is a quote mark in Japanese, which the punctuation
# step deletes. Deleting it before NFKC here as well lets NFKC compose a letter with a mark the apostrophe
# separated ("l'" + U+0301), whichever of the two steps runs first. "mcdonald's" keeps its apostrophe.
_QUOTE_APOSTROPHE = regex.compile(r"(?<!\p{Latin})'|'(?!\p{Latin})")
# Across whitespace too: the ja/zh finalize removes spaces between kana, which would rejoin "ー ー" into
# a run that a second pass would then collapse.
_LONG_VOWEL_RUN = re.compile(r"ー(?:\s*ー)+")


def japanese_characters(text: str, language: str | None = None) -> str:
    """Apply NFKC, and delete hyphen and tilde look-alikes and decorative symbols.

    "コーヒー〜♪" becomes "コーヒー"; 〒 is kept. Apostrophes are unified first, because NFKC destroys ´
    (see :func:`unify_apostrophes`). Runs of ー are :func:`long_vowel_runs`, a later stage.
    """

    # Delete before NFKC as well as after: "a～" + U+0308 must compose to "ä" in this pass, not in the next
    # one (NFKC leaves the accent apart while the ～ is still between them).
    text = unicodedata.normalize("NFKC", _QUOTE_APOSTROPHE.sub("", unify_apostrophes(text).translate(_JAPANESE_DELETE)))
    return _QUOTE_APOSTROPHE.sub("", text.translate(_JAPANESE_DELETE))


def long_vowel_runs(text: str, language: str | None = None) -> str:
    """Keep one ー per run: "コーーヒー" becomes "コーヒー".

    Runs after the punctuation step and the deletions, which can join two ー ("ー{ー", "ー★ー").
    """

    return _LONG_VOWEL_RUN.sub("ー", text.replace("ｰ", "ー"))


# Native digits. Category and value both come from ``unicodedata``, so they cannot disagree on the
# Unicode version. Superscripts and other No digits (², ௰) are not Nd and stay.
def _is_native_digit(char: str) -> bool:
    return char > "\x7f" and unicodedata.category(char) == "Nd"


def fold_digits(text: str, language: str | None = None) -> str:
    """Write every Unicode decimal digit as its ASCII digit: "๓", "٣" and "३" become "3"."""

    if not any(_is_native_digit(char) for char in text):
        return text
    return "".join(str(unicodedata.digit(char)) if _is_native_digit(char) else char for char in text)


DIGIT_TRIGGER = ChangedBy(fold_digits)

__all__ = [
    "APOSTROPHES",
    "JAPANESE_DELETED",
    "JAPANESE_KEPT",
    "PARENTHESES",
    "fold_digits",
    "fold_russian_yo",
    "japanese_characters",
    "long_vowel_runs",
    "reorder_thai_marks",
    "unify_apostrophes",
]
