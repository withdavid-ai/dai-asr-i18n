"""Transliteration maps: the spellings of borrowed words in a host script and in Latin script.

Speakers borrow constantly, English above all (mobile, doctor, actually), but also French in Arabic or
Spanish and Japanese words in Korean and Thai. A reference may write such a word in the host script
(कंप्यूटर, إيميل, アイデア) while a model writes it as the source language does (computer, email, idea),
or the reverse. Only borrowings whose source language uses a different script from the host are mapped.

One TSV file per language (``resources/transliteration/<language>.tsv``) has these columns:

- ``native``: a spelling in the host script;
- ``target``: the word's Latin form;
- ``decision``: ``map`` for a loanword; ``native_block`` for a native word that a loanword's spelling
  variant must never capture (ज़ी, "Zee", must not turn the honorific जी into another word); or
  ``preferred`` for a spelling pair that only marks preferred spellings (see ``preferred``);
- ``occurrences``: how often the native spelling occurs in the benchmark references (``map`` rows);
- ``preferred``: ``yes`` when the row's spelling is preferred where a transcript writes one word in several
  spellings of one script. For a ``map`` row the native spelling is preferred; for a ``preferred`` row both
  spellings are. Lookup and classes ignore ``preferred`` rows.

All native spellings of one Latin form are a class, represented by its most frequent native spelling.
Lookup is exact, then by a spelling-variant key (:func:`variant_key`); a key that two classes share, or
that a native word folds to, is dropped rather than guessed. Only bare loanwords are mapped: a loanword
with a native prefix or ending (Marathi फोनवर, "on the phone") stays exactly as written.
"""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cache
from itertools import groupby
from pathlib import Path

from dai_asr_i18n.normalization.tokens import thai_words

SPACE_DELIMITED = frozenset({"hi", "mr", "bn", "ta", "te", "ar", "ko", "ru"})
LANGUAGES = SPACE_DELIMITED | {"ja", "th"}
MAPS_DIR = Path(__file__).parent / "resources" / "transliteration"
# sha256 over every map file (name and bytes, in name order); tests/test_transliteration.py checks it.
MAPS_SHA256 = "66e334ec48463dcfb5e56fc5caccd02feaa8b908b1307f18cab96bb3fd178326"

# Lookup units.
LATIN_ONLY = re.compile(r"^[a-z0-9'’\-]+$")
KATAKANA_RUN = re.compile("[ァ-ヺー-ヾ]+")
THAI_RUN = re.compile("[฀-๿]+")
_HAS_LETTER = re.compile("[a-z]")
_LATIN_RUN = r"(?P<latin>[a-z0-9'’\-]+(?: [a-z0-9'’\-]+)*)"
# Unspaced scripts: a match is either a run of space-separated Latin tokens or a native run.
_UNSPACED_UNITS = {
    "ja": re.compile(f"{_LATIN_RUN}|{KATAKANA_RUN.pattern}"),
    "th": re.compile(f"{_LATIN_RUN}|{THAI_RUN.pattern}"),
}

# Spelling-variant folding, for lookup only.
_NUKTA = {"hi": "़", "mr": "़", "bn": "়"}
_DEVA_HALF_NASAL = re.compile("[ङञणनम]्(?=[क-ह])")
_BENG_HALF_NASAL = re.compile("[ঙঞণনম]্(?=[ক-হ])")
_DEVA_CANDRA = str.maketrans({"ॉ": "ो", "ॅ": "े", "ऑ": "ओ", "ऍ": "ए"})
_ARABIC = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي", "ـ": None})
_ARABIC_HARAKAT = re.compile("[ً-ْٰ]")
_SMALL_KANA = str.maketrans({"ァ": "ア", "ィ": "イ", "ゥ": "ウ", "ェ": "エ", "ォ": "オ", "ー": None})


def variant_key(language: str, word: str) -> str:
    """Fold optional spelling differences, so कम्प्यूटर / कंप्यूटर and コンピュータ / コンピューター share a key.

    Nukta, half-nasal and the candra vowels in Devanagari; nukta and half-nasal in Bengali; alef forms,
    ta marbuta, alef maqsura, harakat and tatweel in Arabic; ё in Russian; the long-vowel mark and small
    vowel kana in Japanese.
    """

    word = unicodedata.normalize("NFC", word)
    if language in _NUKTA:
        word = unicodedata.normalize("NFC", unicodedata.normalize("NFD", word).replace(_NUKTA[language], ""))
    if language in {"hi", "mr"}:
        word = _DEVA_HALF_NASAL.sub("ं", word).translate(_DEVA_CANDRA)
    elif language == "bn":
        word = _BENG_HALF_NASAL.sub("ং", word)
    elif language == "ar":
        word = _ARABIC_HARAKAT.sub("", word).translate(_ARABIC)
    elif language == "ru":
        word = word.replace("ё", "е")
    elif language == "ja":
        word = word.translate(_SMALL_KANA)
    return word


def latin_class(language: str, target: str) -> str:
    """A map row's Latin form as the language's base normalizer writes it: a class's name.

    The base normalizer is the language transform with its own punctuation pass, so "it's" is named
    "it s".
    """

    from dai_asr_i18n.normalization.profiles import base_normalizer  # profiles imports this module's users

    return " ".join(base_normalizer(language).apply(target.lower()).split())


@dataclass(frozen=True)
class Entry:
    """A native spelling's class."""

    target: str  # the class: the Latin form, as :func:`latin_class` writes it
    canonical: str  # the class's representative (its most frequent native spelling)

    @property
    def parts(self) -> list[str]:
        return self.canonical.split()


@dataclass(frozen=True)
class TransliterationMap:
    """One language's map, indexed for lookup."""

    exact: dict[str, Entry]  # native lookup unit -> entry
    by_variant: dict[str, Entry]  # spelling-variant key -> entry
    latin: dict[tuple[str, ...], str]  # Latin form as tokens -> representative
    longest_latin: int


_EMPTY_MAP = TransliterationMap({}, {}, {}, 0)


def map_path(language: str) -> Path:
    return MAPS_DIR / f"{language}.tsv"


def read_map_file(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    """(rows decided ``map``, native words decided ``native_block``), in file order."""

    mapped: list[dict[str, str]] = []
    native_blocks: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["decision"] == "map":
                mapped.append(row)
            elif row["decision"] == "native_block":
                native_blocks.add(row["native"])
    return mapped, native_blocks


def preferred_spellings(language: str) -> frozenset[str]:
    """The spellings marked preferred in the language's map file (see the module docstring)."""

    path = map_path(language)
    if not path.exists():
        return frozenset()
    out: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["preferred"] != "yes":
                continue
            out.add(row["native"])
            if row["decision"] == "preferred":
                out.add(row["target"])
    return frozenset(out)


def _representatives(language: str, rows: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    """(class -> representative, native spelling -> class).

    The representative is the native spelling the references use most; ties go to code-point order.
    """

    weights: dict[str, Counter] = defaultdict(Counter)
    class_of: dict[str, str] = {}
    for row in rows:
        target, native = latin_class(language, row["target"]), row["native"]
        if not target or not native:
            continue
        weights[target][native] += int(row.get("occurrences") or 1)
        class_of[native] = target
    canonical = {target: min(counts, key=lambda s, w=counts: (-w[s], s)) for target, counts in weights.items()}
    return canonical, class_of


def _spell_groups_word_by_word(canonical: dict[str, str]) -> None:
    """Represent a multi-word class word by word when each of its words has a class of its own.

    Then every segmentation converges: ఓకెకే ("okay okay"), "okay okay" and ఓకే ఓకే all become ఓకే ఓకే.
    Not when a piece is a single letter: "it's" is named "it s", and spelling it word by word would turn
    இட்ஸ் into ஐடி எஸ் ("IT S"); the group keeps its own representative instead.
    """

    for target in [t for t in canonical if " " in t]:
        words = target.split()
        if all(word in canonical and len(word) > 1 for word in words):
            canonical[target] = " ".join(canonical[word] for word in words)


def _variant_index(language: str, exact: dict[str, Entry], native_blocks: set[str]) -> dict[str, Entry]:
    """Spelling-variant key -> entry, without the keys that are ambiguous or belong to a native word."""

    by_variant: dict[str, Entry] = {}
    dropped: set[str] = set()
    for native, entry in exact.items():
        key = variant_key(language, native)
        if by_variant.get(key, entry) != entry:
            dropped.add(key)
        by_variant[key] = entry
    dropped |= {variant_key(language, native) for native in native_blocks}
    for key in dropped:
        by_variant.pop(key, None)
    return by_variant


def _latin_index(canonical: dict[str, str]) -> dict[tuple[str, ...], str]:
    """Latin form (as tokens) -> representative, for classes whose form is Latin text with a letter."""

    return {
        tuple(target.split()): representative
        for target, representative in canonical.items()
        if all(LATIN_ONLY.match(token) for token in target.split()) and _HAS_LETTER.search(target)
    }


@cache
def load_map(language: str) -> TransliterationMap:
    """The language's map; empty when the language has no map file."""

    if language not in LANGUAGES:
        return _EMPTY_MAP
    path = map_path(language)
    if not path.exists():
        return _EMPTY_MAP
    rows, native_blocks = read_map_file(path)
    canonical, class_of = _representatives(language, rows)
    _spell_groups_word_by_word(canonical)
    exact = {native: Entry(target, canonical[target]) for native, target in class_of.items()}
    latin = _latin_index(canonical)
    return TransliterationMap(
        exact=exact,
        by_variant=_variant_index(language, exact, native_blocks),
        latin=latin,
        longest_latin=max((len(key) for key in latin), default=0),
    )


def lookup(language: str, unit: str) -> Entry | None:
    """The entry for a native lookup unit: an exact match first, then its spelling-variant key."""

    loaded = load_map(language)
    return loaded.exact.get(unit) or loaded.by_variant.get(variant_key(language, unit))


def _latin_phrase_length(loaded: TransliterationMap, tokens: list[str], index: int) -> int:
    """Length of the longest Latin form starting at ``tokens[index]``, or 0."""

    for size in range(min(loaded.longest_latin, len(tokens) - index), 0, -1):
        if tuple(tokens[index : index + size]) in loaded.latin:
            return size
    return 0


def _fold_latin_tokens(loaded: TransliterationMap, tokens: list[str]) -> list[str]:
    """Latin tokens with every Latin form (longest first) replaced by its representative."""

    out: list[str] = []
    index = 0
    while index < len(tokens):
        size = _latin_phrase_length(loaded, tokens, index)
        out.append(loaded.latin[tuple(tokens[index : index + size])] if size else tokens[index])
        index += size or 1
    return out


def _fold_space_delimited(language: str, loaded: TransliterationMap, text: str) -> str:
    """Fold each whitespace token; consecutive Latin tokens are matched as phrases."""

    out: list[str] = []
    for is_latin, run in groupby(text.split(), key=lambda token: bool(LATIN_ONLY.match(token))):
        if is_latin:
            for folded in _fold_latin_tokens(loaded, list(run)):
                out.extend(folded.split())
        else:
            for token in run:
                entry = lookup(language, token)
                out.extend(entry.parts if entry else [token])
    return " ".join(out)


def _fold_unspaced(language: str, loaded: TransliterationMap, text: str) -> str:
    """Fold Latin runs and native units inside text without word spaces (Japanese, Thai).

    A replaced unit is padded with spaces so it becomes its own token; text that matches nothing stays
    exactly as it was. A Japanese unit is a katakana run; a Thai unit is a PyThaiNLP word.
    """

    def fold_native(unit: str) -> str:
        entry = lookup(language, unit)
        if entry is None or entry.parts == [unit]:
            return unit
        return f" {' '.join(entry.parts)} "

    def replace(match: re.Match[str]) -> str:
        run = match.group()
        if match.group("latin"):
            folded = " ".join(_fold_latin_tokens(loaded, run.split()))
            return folded if folded == run else f" {folded} "
        if language == "ja":
            return fold_native(run)
        return "".join(fold_native(word) for word in thai_words(run))

    return _UNSPACED_UNITS[language].sub(replace, text)


def fold(language: str, text: str) -> str:
    """Rewrite every member of a mapped class (native spelling or Latin form) to its representative."""

    loaded = load_map(language)
    if language not in LANGUAGES or not loaded.exact:
        return text
    if language in SPACE_DELIMITED:
        return _fold_space_delimited(language, loaded, text)
    return _fold_unspaced(language, loaded, text)


def maps_digest(directory: Path = MAPS_DIR) -> str:
    """One sha256 over every map file (name and bytes) in a directory, in name order."""

    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.tsv")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


__all__ = [
    "LANGUAGES",
    "MAPS_DIR",
    "MAPS_SHA256",
    "Entry",
    "TransliterationMap",
    "fold",
    "latin_class",
    "load_map",
    "lookup",
    "maps_digest",
    "preferred_spellings",
    "read_map_file",
    "variant_key",
]
