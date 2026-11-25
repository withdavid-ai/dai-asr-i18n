"""Event parentheses: remove a parenthetical only when it names an event; otherwise keep its contents.

"(laughs) call (02) 8012 3456" becomes "call 02 8012 3456". Removing every parenthetical with its
contents would erase spoken words such as an area code.

Every Unicode parenthesis pair is folded to ASCII first, nesting resolves inside out, and an unmatched
delimiter is dropped. The step owns the parentheses: no other step touches them.

An event is contents that, after :func:`event_key`, contain no digit, have at most ``MAX_EVENT_WORDS``
words, and match the event list (``resources/events.tsv``) in any language, since a model can write an
English "(laughs)" into a Spanish transcript. :func:`event_key` folds everything the language transforms
and the other steps can change in a word (case, compatibility forms, Arabic harakat and letter variants,
Turkish dotless i, ё, traditional Han (with the same OpenCC t2s conversion Chinese uses), ー runs,
punctuation, and for English the diacritics Whisper drops), so the decision is the same whether the step
runs first or last.
"""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from whisper_normalizer.basic import remove_symbols_and_diacritics

from dai_asr_i18n.normalization.characters import (
    PARENTHESES,
    japanese_characters,
    reorder_thai_marks,
    unify_apostrophes,
)
from dai_asr_i18n.normalization.cleanup import PARENTHESIS_TRANSLATION
from dai_asr_i18n.normalization.languages import ArabicNormalizer, SimplifiedChineseNormalizer
from dai_asr_i18n.normalization.punctuation import base_removes

EVENTS_PATH = Path(__file__).parent / "resources" / "events.tsv"
MAX_EVENT_WORDS = 6
# Substring terms (Han, kana, Thai) only count in short contents, so a long gloss that happens to
# contain 笑 is not taken for an event.
MAX_SUBSTRING_CONTENT = 12
TRIGGER = re.compile(f"[{re.escape(PARENTHESES)}]")
_INNERMOST = re.compile(r"\(([^()]*)\)")
_KEY_FOLDS = str.maketrans({"ı": "i", "ё": "е", "ʼ": " "})
_LONG_VOWEL_RUN = re.compile("ー+")
# The character steps that change more than single letters (deleting, reordering); ё is in _KEY_FOLDS.
_LANGUAGE_CHARACTERS = {"ja": japanese_characters, "th": reorder_thai_marks}
_ARABIC = ArabicNormalizer()
_CHINESE = SimplifiedChineseNormalizer()


def event_key(text: str, language: str | None) -> str:
    """Contents as every language transform and step would agree on them, for matching only."""

    # The transcript language's own character step first, so a deleted ～ joins words here exactly as it
    # does in the text (a key that spaced it instead would split "x～laughs" into two words).
    text = _LANGUAGE_CHARACTERS.get(language, unify_apostrophes)(text)
    # Apostrophes first: NFKC would turn ´ into a space and a combining accent glued to the word.
    text = unicodedata.normalize("NFKC", unify_apostrophes(text)).casefold().replace("i̇", "i")
    text = _ARABIC.core(text.translate(_KEY_FOLDS))
    if language == "en":
        text = remove_symbols_and_diacritics(text)
    text = _LONG_VOWEL_RUN.sub("ー", _CHINESE.converter.convert(text))
    # What the language's punctuation pass removes (Thai ฯ, Turkish marks), so the decision is the same
    # whether that pass has run inside the parentheses yet or not.
    removes = base_removes(language)
    return " ".join("".join(" " if removes(char) else char for char in text).split())


@dataclass(frozen=True)
class EventList:
    """The event list as event keys, by how each term matches."""

    words: frozenset[str]  # whole words or phrases
    prefixes: tuple[str, ...]
    substrings: tuple[str, ...]
    sha256: str


@cache
def event_list() -> EventList:
    """Read ``resources/events.tsv`` once."""

    raw = EVENTS_PATH.read_bytes()
    lines = [line for line in raw.decode("utf-8").splitlines() if not line.startswith("#")]
    terms: dict[str, set[str]] = {"word": set(), "prefix": set(), "substring": set()}
    for row in csv.DictReader(lines, delimiter="\t"):
        # Keyed both ways so an accented term also matches English contents, whose key drops accents.
        terms[row["match"]].update(key for language in (None, "en") if (key := event_key(row["term"], language)))
    return EventList(
        words=frozenset(terms["word"]),
        prefixes=tuple(sorted(terms["prefix"])),
        substrings=tuple(sorted(terms["substring"])),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def is_event(contents: str, language: str | None) -> bool:
    """Whether a parenthetical's contents name an event."""

    key = event_key(contents, language)
    words = key.split()
    if not words:
        return True  # "()", "(...)": nothing spoken to keep
    if len(words) > MAX_EVENT_WORDS or any(unicodedata.category(char) == "Nd" for char in key):
        return False
    events = event_list()
    padded = f" {key} "
    if any(f" {term} " in padded for term in events.words):
        return True
    if any(word.startswith(prefix) for word in words for prefix in events.prefixes):
        return True
    compact = "".join(words)
    return len(compact) <= MAX_SUBSTRING_CONTENT and any(term in compact for term in events.substrings)


def resolve_parentheses(text: str, language: str | None) -> str:
    """Remove event parentheticals with their contents; unwrap the rest; drop unmatched delimiters."""

    text = text.translate(PARENTHESIS_TRANSLATION)
    while _INNERMOST.search(text):
        text = _INNERMOST.sub(lambda match: " " if is_event(match.group(1), language) else f" {match.group(1)} ", text)
    return text.replace("(", " ").replace(")", " ")


__all__ = ["EVENTS_PATH", "EventList", "event_key", "event_list", "is_event", "resolve_parentheses"]
