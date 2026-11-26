"""Apostrophes against the reference (de, tl, tr): a difference of only apostrophes is not an error.

German writes an optional apostrophe for a clitic ("aus'm" / "ausm", "über's" / "übers", "so'n" / "so n"),
Tagalog for a contraction ("sa'yo" / "sayo"), Turkish before a suffix on a name or a foreign word
("türkçe'yi" / "türkçeyi", "devlet'ten" / "devletten"). Either spelling is right, and two transcripts of
one moment often choose differently. The punctuation step keeps an apostrophe between letters, so on its
own each choice would be an error.

Inside an error run, a stretch of up to ``MAX_TOKENS`` tokens per side whose letters are equal once
apostrophes and spaces are removed, with an apostrophe on at least one side, takes the reference's
spelling on the hypothesis side. The longest hypothesis stretch is tried first, left to right, and for it
the first reference stretch (the longest at each start) that matches. A stretch with no apostrophe is not
touched: "so n" / "son" is a spacing difference, which this step does not decide. Nor is a space against
nothing anywhere in a stretch, even one that has an apostrophe elsewhere: the stretches may differ only
where one side writes an apostrophe ("ka agad sayo" / "kaagad sa'yo" stays an error).
"""

from __future__ import annotations

from collections.abc import Sequence

from dai_asr_i18n.normalization.pair import Rewrite, error_runs, is_mark
from dai_asr_i18n.normalization.tokens import AlignmentOp

STEP = "apostrophe-spacing"
LANGUAGES = frozenset({"de", "tl", "tr"})
MAX_TOKENS = 4  # tokens per side
APOSTROPHE = "'"


def _key(tokens: Sequence[str]) -> str:
    return "".join(tokens).replace(APOSTROPHE, "")


def _marks(tokens: Sequence[str]) -> dict[int, str]:
    """Letter position -> the mark written before it: an apostrophe (with or without a space) or a space."""
    out: dict[int, str] = {}
    letters = 0
    for ch in " ".join(tokens):
        if ch in (" ", APOSTROPHE):
            out[letters] = APOSTROPHE if (ch == APOSTROPHE or out.get(letters) == APOSTROPHE) else " "
        else:
            letters += 1
    return out


def _only_at_apostrophes(ref: Sequence[str], hyp: Sequence[str]) -> bool:
    """Every place where the two stretches mark a letter boundary differently has an apostrophe on one side.

    A space against nothing is a spacing difference this step does not decide, even next to an apostrophe
    elsewhere in the stretch: "ka agad sayo" / "kaagad sa'yo" is left alone, not respaced to the reference."""
    ref_marks, hyp_marks = _marks(ref), _marks(hyp)
    return all(
        APOSTROPHE in (ref_marks.get(p), hyp_marks.get(p))
        for p in ref_marks.keys() | hyp_marks.keys()
        if ref_marks.get(p) != hyp_marks.get(p)
    )


def _matches(ref: Sequence[str], hyp: Sequence[str]) -> bool:
    return (
        list(ref) != list(hyp)
        and _key(ref) == _key(hyp)
        and any(APOSTROPHE in t for t in (*ref, *hyp))
        and not any(is_mark(t) for t in (*ref, *hyp))
        and _only_at_apostrophes(ref, hyp)
    )


def apostrophe_rewrites(
    ref: Sequence[str], hyp: Sequence[str], ops: Sequence[AlignmentOp], language: str | None
) -> list[Rewrite]:
    """The hypothesis stretches inside error runs that differ from the reference only by apostrophes."""

    if language not in LANGUAGES:
        return []
    out: list[Rewrite] = []
    for ref_run, hyp_run in error_runs(ops):
        h = [ref[k] for k in ref_run]
        m = [hyp[k] for k in hyp_run]
        i = j = 0
        while j < len(m):
            hit = None
            for mj in range(min(len(m), j + MAX_TOKENS), j, -1):
                for hi in range(i, len(h)):
                    for hk in range(min(len(h), hi + MAX_TOKENS), hi, -1):
                        if _matches(h[hi:hk], m[j:mj]):
                            hit = (hi, hk, mj)
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit is None:
                j += 1
                continue
            hi, hk, mj = hit
            out.append(
                Rewrite(
                    STEP,
                    "apostrophes and spaces only",
                    ref_run[hi],
                    ref_run[hk - 1] + 1,
                    hyp_run[j],
                    hyp_run[mj - 1] + 1,
                    tuple(h[hi:hk]),
                    tuple(m[j:mj]),
                )
            )
            i, j = hk, mj
    return out


__all__ = ["LANGUAGES", "STEP", "apostrophe_rewrites"]
