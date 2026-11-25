"""English: Whisper's ``EnglishTextNormalizer``, without contraction expansion, split around the steps.

Whisper's English normalizer runs as two stages here. :class:`EnglishBase` is the base stage, before the
per-text steps: NFKC, lowercase, bracket spans, titles ("mr" becomes "mister"), commas between digits,
accent folding and British-to-American spelling. :func:`english_late` runs after the steps: symbol
removal, then number reading.

Four differences from Whisper:

- Contractions stay as written ("don't", "gonna", "wanna"): the contraction replacers are skipped
  (:data:`CONTRACTION_PATTERNS`). "'cause" and "cause" are still "because".
- The ``\\s+'`` join is skipped: with apostrophes kept, it would glue a quoted word to the word before it.
- The apostrophe survives symbol removal; the punctuation step removes every one without a letter on both
  sides, so quotation marks still go.
- Fillers are kept on both sides (Whisper deletes hmm, mm, mhm, mmm, uh and um), so a filler on one side
  only is an error, as in every other language. "mm hmm" is written "mhm".

Numbers are read as Whisper reads them, with the exceptions in :class:`EnglishNumbers`.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from whisper_normalizer.basic import ADDITIONAL_DIACRITICS
from whisper_normalizer.english import EnglishNumberNormalizer, EnglishTextNormalizer

# Every contraction-expansion pattern in whisper-normalizer 0.1.x's EnglishTextNormalizer: its "common
# contractions", "perfect tenses" and "general contractions" groups. Its fourth group, titles (mr ->
# mister), expands abbreviations rather than contractions and stays, as does ``\bcause\b`` ("because").
CONTRACTION_PATTERNS: frozenset[str] = frozenset(
    {
        r"\bwon't\b",
        r"\bcan't\b",
        r"\blet's\b",
        r"\bain't\b",
        r"\by'all\b",
        r"\bwanna\b",
        r"\bkinda\b",
        r"\bsorta\b",
        r"\bdunno\b",
        r"\bgotta\b",
        r"\bgonna\b",
        r"\bi'ma\b",
        r"\bimma\b",
        r"\bwoulda\b",
        r"\bcoulda\b",
        r"\bshoulda\b",
        r"\bma'am\b",
        r"'d been\b",
        r"'s been\b",
        r"'d gone\b",
        r"'s gone\b",
        r"'d done\b",
        r"'s got\b",
        r"n't\b",
        r"'re\b",
        r"'s\b",
        r"'d\b",
        r"'ll\b",
        r"'t\b",
        r"'ve\b",
        r"'m\b",
    }
)

# Fillers Whisper's spelling map rewrites as "hmm" that stay as said.
_FILLERS_KEPT = frozenset({"mhm", "mmm"})
_KEEP_LATE = ".%$¢€£'"
_INTEGER = re.compile(r"[0-9]+")
_LONE_ONE = re.compile(r"(?<![0-9][ ])(?<![0-9.€£$¢])\b1(s?)\b(?![.,][0-9])")
# "mm-hmm" and "mm hmm" are the filler written "mhm".
_MM_HMM = re.compile(r"\bmm hmm\b")
_WS = re.compile(r"\s+")


class EnglishBase:
    """The English base stage: Whisper's English normalizer up to, but not including, symbol removal.

    No contraction expansion, filler deletion or "(...)" deletion (the event-parentheses step decides);
    no symbol removal or number reading (:func:`english_late`). Accents are folded here, before the
    steps, so the punctuation step sees the letters it keeps an apostrophe between.
    """

    def __init__(self) -> None:
        whisper = EnglishTextNormalizer()
        self.replacers = {
            pattern: replacement
            for pattern, replacement in whisper.replacers.items()
            if pattern not in CONTRACTION_PATTERNS
        }
        mapping = whisper.standardize_spellings.mapping
        self.spellings = {key: value for key, value in mapping.items() if key not in _FILLERS_KEPT}

    def __call__(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text).lower()
        text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)  # remove words between brackets
        for pattern, replacement in self.replacers.items():  # titles only: contractions stay
            text = re.sub(pattern, replacement, text)
        text = re.sub(r"(\d),(\d)", r"\1\2", text)  # remove commas between digits
        text = _fold_accents(text)
        text = re.sub(r"[a-z]+", lambda match: self.spellings.get(match.group(), match.group()), text)
        return _WS.sub(" ", text).strip()


def _fold_accents(text: str) -> str:
    """Whisper's diacritic folding without its symbol removal.

    NFKD, combining marks dropped, Whisper's manual mappings (æ -> ae, ø -> o, ß -> ss), then NFC, so
    nothing is left decomposed for the later steps.
    """

    folded = "".join(
        ADDITIONAL_DIACRITICS.get(char, "" if unicodedata.category(char) == "Mn" else char)
        for char in unicodedata.normalize("NFKD", text)
    )
    return unicodedata.normalize("NFC", folded)


class EnglishNumbers(EnglishNumberNormalizer):
    """Whisper's English number reader, with four exceptions.

    - "oh" and "o" are never numbers: "oh my god" stays, and "five oh five" is "5 oh 5".
    - Pounds are not currency: "five pounds" is "5 pounds", not "£5" (it is usually a weight).
    - A letter/digit code stays one token: "n95" and "f6y9k3" are not split.
    - A digit group that starts with 0 keeps its zeros: the leading-zeros step has marked it as letters
      before this runs, so the reader never sees it.

    Everything else is Whisper's reading: "second" is "2nd", "seven and a half" is 7.5, "1.8 billion" is
    1800000000, and a lone 1 is written "one". Three changes keep the result a fixed point, where
    Whisper's is not: "1" after a number is not written "one" ("2 1" stays "2 1"; Whisper's "2 one"
    reads as 21 next time), nor inside a decimal or after a currency sign ("1.5", "$1"); and a digit
    group before "and a half" takes the half like a number word ("1 and a half" is 1.5).
    """

    def __init__(self) -> None:
        super().__init__()
        self.zeros = self.zeros - {"o", "oh"}
        self.following_prefixers = {
            key: value for key, value in self.following_prefixers.items() if key not in {"pound", "pounds"}
        }
        self.decimals = {*self.ones, *self.tens, *self.zeros}
        self.words = (
            set(self.zeros)
            | set(self.ones)
            | set(self.ones_suffixed)
            | set(self.tens)
            | set(self.tens_suffixed)
            | set(self.multipliers)
            | set(self.multipliers_suffixed)
            | set(self.preceding_prefixers)
            | set(self.following_prefixers)
            | set(self.suffixers)
            | set(self.specials)
        )

    def preprocess(self, s: str) -> str:
        # Whisper's "<number> and a half" -> "<number> point five", for a digit group too; and no space
        # is put at a letter/digit boundary (codes stay joined).
        segments = re.split(r"\band\s+a\s+half\b", s)
        results = []
        for index, segment in enumerate(segments):
            if len(segment.strip()) == 0:
                continue
            results.append(segment)
            if index < len(segments) - 1:
                last_word = segment.rsplit(maxsplit=2)[-1]
                is_number = last_word in self.decimals or last_word in self.multipliers or _INTEGER.fullmatch(last_word)
                results.append("point five" if is_number else "and a half")
        s = " ".join(results)
        return re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", s)  # but remove spaces which could be a suffix

    def postprocess(self, s: str) -> str:
        s = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", _combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", lambda match: f"¢{int(match.group(1))}", s)
        # "one(s)" instead of "1(s)", as Whisper writes it, except after a number, in a decimal or after a
        # currency sign
        return _LONE_ONE.sub(r"one\1", s)


def _combine_cents(match: re.Match[str]) -> str:
    return f"{match.group(1)}{match.group(2)}.{int(match.group(3)):02d}"


def _stray_symbols(text: str) -> str:
    """A $ . ¢ € £ not before a digit, or a % not after one, becomes a space (Whisper's last rules)."""

    text = re.sub(r"[.$¢€£]([^0-9]|$)", r" \1", text)
    return re.sub(r"(^|[^0-9])%", r"\1 ", text)


@lru_cache(maxsize=1)
def _numbers() -> EnglishNumbers:
    return EnglishNumbers()


def english_late(text: str) -> str:
    """The English late stage, after the steps: what Whisper does after its replacers, in its order.

    A period not before a digit, and every remaining punctuation and symbol character except . % $ ¢ € £
    and the apostrophe, become spaces; then the number reader (:class:`EnglishNumbers`); then a $ . ¢ € £
    not before a digit or a % not after one become spaces. Last, "mm hmm" is written "mhm".
    """

    text = re.sub(r"\.([^0-9]|$)", r" \1", text)
    text = "".join(
        char if char in _KEEP_LATE else (" " if unicodedata.category(char)[0] in "PS" else char) for char in text
    )
    # The stray symbols go before the reader too, so their removal afterwards creates nothing it would read
    # differently ("3$ s" -> "3 s", which a second pass would join as "3s"); and after it, as in Whisper.
    text = _numbers()(_WS.sub(" ", _stray_symbols(text)).strip())
    text = _stray_symbols(text)
    return _MM_HMM.sub("mhm", _WS.sub(" ", text).strip())


__all__ = ["CONTRACTION_PATTERNS", "EnglishBase", "EnglishNumbers", "english_late"]
