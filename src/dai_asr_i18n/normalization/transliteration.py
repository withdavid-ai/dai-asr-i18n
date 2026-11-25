"""Transliteration: a borrowed word written in its source language's script and in the host script matches.

A Hindi reference writes "ओके" and the model "okay"; a Japanese reference "コンピューター" and the model
"computer". Both sides wrote the same spoken word. The transliteration maps
(:mod:`dai_asr_i18n.normalization.transliteration_maps`) put the native spellings of a loanword and its Latin form
into one class, and this pair step rewrites the hypothesis only, into a spelling the reference uses:

1. Each side's class occurrences are found on their own. Latin: in a run of Latin tokens, the longest Latin
   form of the map first. Native: the longest span of up to ``MAX_TOKENS`` native tokens (Japanese: a whole
   katakana run, and its parts) that folds, spaces removed, to a class's representative; so the reference's
   "এট লিস্ট" is the class "at least" although the map stores only "এটলিস্ট".
2. The reference's spellings of each class are cached in two groups by script, Latin and the language's
   own. In each group a spelling the maps mark preferred wins; then the one written most often; then the
   one written first.
3. Every hypothesis occurrence of a class takes the winning reference spelling from the other script
   group: a Latin word the reference's native spelling, a native word the reference's Latin one. When the
   reference never writes the class in the other script, the word stays. Two spellings in one script are
   never matched (Telugu కెరియర్ / కేరియర్, Hindi कंप्यूटर / कम्प्यूटर): that is a spelling difference, not a
   script choice.
4. The rewritten hypothesis is aligned against the reference. Inside each error run, a class's rewritten
   words pair in order with the reference's occurrences of that class, so each copy of a repeated loanword
   takes the spelling at its own place: the reference's spelling when it is in the other script from the
   model's own word, and the model's word when it is in the same script.

:func:`dai_asr_i18n.normalization.pair.reconcile` declines a rewrite that would add an error. Only bare loanwords
in hi, mr, bn, ta, te, ar, ko, ru, ja and th are mapped; one with a native affix (फोनवर) is not. Each class
is also indexed under its Latin form as the pipeline writes it ("it's"), besides the name the maps give it
("it s").
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache, lru_cache

from dai_asr_i18n.normalization import transliteration_maps
from dai_asr_i18n.normalization.pair import UNSPACED, Rewrite, error_runs, is_mark
from dai_asr_i18n.normalization.tokens import AlignmentOp, align

STEP = "transliteration"
LANGUAGES = transliteration_maps.LANGUAGES
MAX_TOKENS = 3  # native tokens per span: a loanword the reference split ("এট লিস্ট" for "at least")
MAX_CHARS = 12  # Japanese tokens per span inside a katakana run ("ライン" in "ラインスタンプ")
_JOINED = UNSPACED | {"th"}
# A katakana loanword never starts with these, nor stops right before one: they belong to the sound before.
_NOT_A_START = frozenset("ーｰァィゥェォッャュョヮヵヶｧｨｩｪｫｬｭｮｯ")
_KATAKANA = re.compile("[ァ-ヺー-ヾｦ-ﾟ]+")


def _join(tokens: Sequence[str], language: str | None) -> str:
    """Text as the language writes it: ja, zh and th tokens run together, except beside a Latin word."""
    if language not in _JOINED:
        return " ".join(tokens)
    out = ""
    for t in tokens:
        if out and (out[-1].isascii() and out[-1].isalnum() or t[:1].isascii() and t[:1].isalnum()):
            out += " "
        out += t
    return out


def tokenization(language: str | None) -> str:
    """How this step reads a text: Japanese by character (a Latin word stays one token), so a loanword
    inside a compound such as ウェブサイト can be rewritten; every other language by its word tokens."""

    return "cjk_character" if language == "ja" else "word"


def _is_latin(token: str) -> bool:
    return bool(transliteration_maps.LATIN_ONLY.match(token))


@cache
def _latin_index(language: str) -> dict[tuple[str, ...], str]:
    """Latin form (tokens) -> representative: the maps' index, plus each class whose Latin form has an
    apostrophe, under that form as the pipeline writes it. The maps name a class by the base normalizer's
    form ("it s"), which a pipeline output never contains, since the punctuation step keeps an apostrophe
    between letters."""

    from dai_asr_i18n.normalization.pipeline import DaiAsrI18nNormalizer  # the pipeline imports this module

    m = transliteration_maps.load_map(language)
    index = dict(m.latin)
    path = transliteration_maps.map_path(language)
    if not path.exists():
        return index
    rows, _ = transliteration_maps.read_map_file(path)
    normalizer = DaiAsrI18nNormalizer(language)
    for row in rows:
        if "'" not in row["target"] and "’" not in row["target"]:
            continue
        named = tuple(transliteration_maps.latin_class(language, row["target"]).split())
        written = tuple(normalizer.normalize(row["target"]).split())
        if named in m.latin and written and all(_is_latin(t) for t in written):
            index.setdefault(written, m.latin[named])
    return index


@cache
def _classes(language: str) -> dict[str, str]:
    """A representative without spaces -> the representative (a class's id)."""
    return {"".join(rep.split()): rep for rep in set(_latin_index(language).values())}


@lru_cache(maxsize=65536)
def _class_of(language: str, text: str) -> str | None:
    """The class a native span folds to, spaces removed, or None."""
    return _classes(language).get("".join(transliteration_maps.fold(language, text).split()))


@lru_cache(maxsize=65536)
def _exact_class(language: str, text: str) -> str | None:
    """The class of a native spelling the maps list exactly (or a class's representative), or None."""
    entry = transliteration_maps.load_map(language).exact.get(text)
    rep = entry.canonical if entry is not None else text
    return _classes(language).get("".join(rep.split()))


def _squeeze_all(text: str) -> str:
    return "".join(text.split())


@cache
def _preferred(language: str) -> frozenset[str]:
    """The spellings, spaces removed, the language's map file marks preferred."""

    return frozenset(_squeeze_all(spelling) for spelling in transliteration_maps.preferred_spellings(language))


@dataclass(frozen=True)
class _Occurrence:
    start: int
    end: int
    cls: str  # the class's representative
    latin: bool  # written in Latin script; else in the language's own


def _katakana_runs(toks: Sequence[str]) -> dict[int, int]:
    """Every position inside a katakana run -> the run's end. A run is a stretch of tokens written wholly in
    katakana. It never starts with a token that starts with a long-vowel mark or a small kana ("えー、ライン":
    the filler keeps its ー)."""
    out: dict[int, int] = {}
    i, n = 0, len(toks)
    while i < n:
        if not _KATAKANA.fullmatch(toks[i]) or toks[i][0] in _NOT_A_START:
            i += 1
            continue
        j = i + 1
        while j < n and _KATAKANA.fullmatch(toks[j]):
            j += 1
        out.update(dict.fromkeys(range(i, j), j))
        i = j
    return out


@dataclass(frozen=True)
class _Side:
    """One side's tokens and where its katakana runs are (Japanese)."""

    toks: Sequence[str]
    language: str
    runs: dict[int, int]

    @classmethod
    def of(cls, toks: Sequence[str], language: str) -> _Side:
        return cls(toks, language, _katakana_runs(toks) if language == "ja" else {})

    def scan(self, start: int, end: int, longest_latin: int, longest_native: int) -> list[_Occurrence]:
        """Occurrences in toks[start:end], left to right, longest first, never overlapping."""
        toks, language, latin = self.toks, self.language, _latin_index(self.language)
        out: list[_Occurrence] = []
        i = start
        while i < end:
            hit = None
            if is_mark(toks[i]):
                pass
            elif _is_latin(toks[i]):
                for k in range(min(longest_latin, end - i), 0, -1):
                    key = tuple(toks[i : i + k])
                    if key in latin and all(_is_latin(t) for t in key):
                        hit = _Occurrence(i, i + k, latin[key], True)
                        break
            elif language == "ja":
                # inside a katakana run; a part never starts or stops at a long-vowel mark or a small kana
                run_end = self.runs.get(i)
                if run_end is not None and toks[i][0] not in _NOT_A_START:
                    run_start = i > 0 and self.runs.get(i - 1) == run_end
                    for j in range(min(run_end, end, i + longest_native), i, -1):
                        if j < run_end and toks[j][0] in _NOT_A_START:
                            continue
                        # a whole run is looked up as the maps look it up, and so is a part of 3 characters
                        # or more (ファイア, a variant of ファイアー); a shorter part only by its exact spelling,
                        # since the variant key drops ー and small kana (ル would match ルー, ルプ ループ)
                        whole = not run_start and j == run_end
                        span = "".join(toks[i:j])
                        loose = whole or len(span) >= 3
                        cls = _class_of(language, span) if loose else _exact_class(language, span)
                        if cls is not None:
                            hit = _Occurrence(i, j, cls, False)
                            break
            else:
                for k in range(min(longest_native, end - i), 0, -1):
                    span = toks[i : i + k]
                    if any(_is_latin(t) or is_mark(t) for t in span):
                        continue
                    cls = _class_of(language, _join(span, language))
                    if cls is not None:
                        hit = _Occurrence(i, i + k, cls, False)
                        break
            if hit is None:
                i += 1
            else:
                out.append(hit)
                i = hit.end
        return out

    def occurrences(self) -> list[_Occurrence]:
        longest_latin = max((len(k) for k in _latin_index(self.language)), default=0)
        longest_native = MAX_CHARS if self.language == "ja" else MAX_TOKENS
        return self.scan(0, len(self.toks), longest_latin, longest_native)

    def parts(self, o: _Occurrence) -> list[_Occurrence]:
        """The shorter occurrences inside a phrase or compound ("amazon prime": "amazon", "prime"; ラインスタンプ:
        ライン, スタンプ), each of them with its own parts after it."""
        size = o.end - o.start
        if size < 2:
            return []
        out = []
        for p in self.scan(o.start, o.end, size - 1, size - 1):
            out += [p, *self.parts(p)]
        return out


def _cache(
    ref: _Side, occurrences: Sequence[_Occurrence]
) -> tuple[dict[tuple[str, bool], _Occurrence], dict[tuple[str, bool], set[tuple[str, ...]]]]:
    """(class, Latin?) -> the reference occurrence whose spelling is preferred; and (class, Latin?) -> every
    spelling the reference writes. Preferred: a spelling the maps mark preferred (`_preferred`), then the
    one written most often, then the one written first. The parts of a phrase or compound count too: "amazon
    prime" also spells "prime"."""
    preferred = _preferred(ref.language)
    spellings: dict[tuple[str, bool], dict[tuple[str, ...], list[_Occurrence]]] = {}
    for o in [x for o in occurrences for x in (o, *ref.parts(o))]:
        spellings.setdefault((o.cls, o.latin), {}).setdefault(tuple(ref.toks[o.start : o.end]), []).append(o)

    def rank(item: tuple[tuple[str, ...], list[_Occurrence]]) -> tuple[bool, int, int]:
        spelling, found = item
        return (_squeeze_all("".join(spelling)) not in preferred, -len(found), found[0].start)

    best = {key: min(by.items(), key=rank)[1][0] for key, by in spellings.items()}
    return best, {key: set(by) for key, by in spellings.items()}


def transliteration_rewrites(
    ref: Sequence[str], hyp: Sequence[str], ops: Sequence[AlignmentOp], language: str | None
) -> list[Rewrite]:
    """The hypothesis's loanwords in the reference's spelling (see the module docstring). `ops` is not read:
    the spellings come from the whole reference, and pass 2 aligns the rewritten hypothesis itself."""
    if language not in LANGUAGES:
        return []
    ref_side, hyp_side = _Side.of(ref, language), _Side.of(hyp, language)
    ref_top = ref_side.occurrences()
    if not ref_top:
        return []
    cache, written = _cache(ref_side, ref_top)

    def planned(o: _Occurrence) -> list[tuple[_Occurrence, _Occurrence | None]]:
        """(hypothesis occurrence, reference source): the whole when the reference spells it in the other
        script, else its parts that it does ("ウェブサイト" against "webサイト": ウェブ takes "web"). The source is
        None when the reference also writes the model's own spelling: only pass 2 may move that word."""
        src = cache.get((o.cls, not o.latin))
        if src is not None:
            own = tuple(hyp[o.start : o.end]) in written.get((o.cls, o.latin), ())
            return [(o, None if own else src)]
        out = []
        for p in hyp_side.scan(o.start, o.end, o.end - o.start - 1, o.end - o.start - 1):
            out += planned(p)
        return out

    pairs = [x for o in hyp_side.occurrences() for x in planned(o)]
    if not pairs:
        return []
    hyp_occ = [o for o, _ in pairs]
    plan: dict[int, _Occurrence | None] = dict(enumerate(src for _, src in pairs))
    ref_occ = [x for o in ref_top for x in (o, *ref_side.parts(o))]

    # The hypothesis with every planned spelling, and where each planned occurrence lands in it.
    rewritten: list[str] = []
    where: dict[int, tuple[int, int]] = {}
    pos = 0
    for k, o in enumerate(hyp_occ):
        rewritten += hyp[pos : o.start]
        start = len(rewritten)
        src = plan[k]
        rewritten += ref[src.start : src.end] if src is not None else hyp[o.start : o.end]
        where[k] = (start, len(rewritten))
        pos = o.end
    rewritten += hyp[pos:]

    # Pass 2: a planned word aligned with a reference occurrence of its class follows that occurrence.
    # Inside each error run, a class's planned words pair in order with the reference's occurrences of that
    # class in the run: not with the alignment's own pairing, which on a tie can face a loanword with the
    # word next to its partner ("ran online" / "แลนด์ ออนไลน์" aligned as ran/ออนไลน์).
    for ref_run, hyp_run in error_runs(align(list(ref), rewritten)[0]):
        in_ref, in_hyp = set(ref_run), set(hyp_run)
        for cls in {hyp_occ[k].cls for k, (s, e) in where.items() if in_hyp & set(range(s, e))}:
            mine = sorted(k for k, (s, e) in where.items() if hyp_occ[k].cls == cls and in_hyp & set(range(s, e)))
            theirs = sorted(
                {p for p in ref_occ if p.cls == cls and in_ref & set(range(p.start, p.end))},
                key=lambda p: (p.start, -p.end),
            )
            theirs = [p for i, p in enumerate(theirs) if not any(q.start <= p.start < q.end for q in theirs[:i])]
            for k, p in zip(mine, theirs):
                if p.latin != hyp_occ[k].latin:
                    plan[k] = p
                else:
                    plan[k] = None  # the reference writes it here in the model's own script: the word stays

    out = []
    for k, src in sorted(plan.items()):
        o = hyp_occ[k]
        if src is None or tuple(ref[src.start : src.end]) == tuple(hyp[o.start : o.end]):
            continue
        out.append(
            Rewrite(
                STEP,
                f"loanword {src.cls}",
                src.start,
                src.end,
                o.start,
                o.end,
                tuple(ref[src.start : src.end]),
                tuple(hyp[o.start : o.end]),
            )
        )
    return out


__all__ = ["LANGUAGES", "STEP", "tokenization", "transliteration_rewrites"]
