"""Pair steps: rules that settle a difference by reading the hypothesis against its reference.

A per-text step sees one transcript. Some differences cannot be settled that way: a loanword written in
Latin script by the model and in the host script by the reference is the same word only because the other
transcript writes it. A pair step reads both normalized transcripts, aligned, and proposes
:class:`Rewrite` objects: replace a hypothesis span with a reference span.

:func:`reconcile` holds these guarantees for every pair step:

- The reference is never changed, so the denominator is exactly the per-text output.
- A rewrite lies inside one error run, or covers one stretch of the alignment that holds an error, and
  makes the hypothesis side equal to the reference side. A step marked ``anywhere`` (transliteration)
  instead copies a spelling the reference writes elsewhere in the text.
- Never more errors, in words or in characters: a step's rewrites are kept together when neither edit
  distance goes up; otherwise they are taken left to right, and one that would raise either is declined
  and recorded. Each step sees the text the earlier steps left.
- Every rewrite is returned with its step and rule.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace

from dai_asr_i18n.languages import canonical_language
from dai_asr_i18n.normalization.tokens import TOKENIZATIONS, AlignmentOp, align, tokenize, word_tokens

# Scripts written without spaces: their tokens are joined back without spaces.
UNSPACED = frozenset({"ja", "zh"})
# A mark a per-text step leaves until its ``finish`` (leading zeros, "zqz...q").
_MARK = re.compile(r"^zq[stz][a-z]+q$")


def is_mark(token: str) -> bool:
    """Marks are read by the step that made them; every pair step looks past them."""

    return bool(_MARK.match(token))


@dataclass(frozen=True)
class Rewrite:
    """One hypothesis span replaced by a reference span, with everything needed to read it on its own.

    ``reference_text`` is the normalized reference and ``hypothesis_text`` the hypothesis as the step read
    it, before this step's rewrites (after any earlier step's). Both are split with ``tokenization`` (see
    :func:`dai_asr_i18n.normalization.tokens.tokenize`); the positions are end-exclusive indices into those token
    lists, and ``reference`` and ``hypothesis`` are the tokens at those positions. The guard counts errors
    on the word tokens whatever tokenization the step reads.
    """

    step: str
    rule: str  # what allowed it, e.g. "loanword computer"
    ref_start: int
    ref_end: int
    hyp_start: int
    hyp_end: int
    reference: tuple[str, ...]  # the reference tokens copied in
    hypothesis: tuple[str, ...]  # the hypothesis tokens replaced
    language: str | None = field(default=None, kw_only=True)
    tokenization: str = field(default="word", kw_only=True)
    reference_text: str = field(default="", kw_only=True)
    hypothesis_text: str = field(default="", kw_only=True)

    def reference_tokens(self) -> list[str]:
        """``reference_text`` as the step read it; ``reference`` is its ``ref_start:ref_end`` slice."""

        return tokenize(self.reference_text, self.language, self.tokenization)

    def hypothesis_tokens(self) -> list[str]:
        """``hypothesis_text`` as the step read it; ``hypothesis`` is its ``hyp_start:hyp_end`` slice."""

        return tokenize(self.hypothesis_text, self.language, self.tokenization)


PairRule = Callable[[Sequence[str], Sequence[str], Sequence[AlignmentOp], str | None], list[Rewrite]]


@dataclass(frozen=True)
class PairStep:
    """One rule that needs both transcripts. ``languages`` None means every language."""

    id: str  # stable key: the ``step`` of each Rewrite it makes, and its manifest entry
    rule: PairRule
    languages: frozenset[str] | None = None
    name: str = field(kw_only=True)
    description: str = field(kw_only=True)
    # True: a rewrite copies a spelling the reference uses anywhere in the text, so it need not lie inside
    # an error run, and several rewrites may copy the same reference span.
    anywhere: bool = field(default=False, kw_only=True)
    # The name of the tokenization the step reads and rewrites in, by language (``TOKENIZATIONS``):
    # transliteration reads Japanese by character. The guard always counts errors on the word tokens.
    tokenization: Callable[[str | None], str] = field(default=lambda language: "word", kw_only=True)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.description.strip():
            raise ValueError(f"pair step {self.id!r} needs a name and a description")

    def applies_to(self, language: str | None) -> bool:
        return self.languages is None or language in self.languages


@dataclass(frozen=True)
class PairNormalized:
    """A normalized reference and hypothesis, with the rewrites the pair steps made and declined."""

    reference: str
    hypothesis: str
    rewrites: tuple[Rewrite, ...]
    declined: tuple[Rewrite, ...] = ()  # proposed, but would have raised the edit distance


def _spans(text: str, tokens: Sequence[str]) -> list[tuple[int, int]] | None:
    """Each token's (start, end) in ``text``, or None when a token is not a substring in order."""

    out, position = [], 0
    for token in tokens:
        start = text.find(token, position)
        if start < 0:
            return None
        out.append((start, start + len(token)))
        position = start + len(token)
    return out


def detokenize(tokens: Sequence[str], language: str | None) -> str:
    return ("" if canonical_language(language) in UNSPACED else " ").join(tokens)


def error_runs(ops: Sequence[AlignmentOp]) -> Iterator[tuple[list[int], list[int]]]:
    """(reference positions, hypothesis positions) of each maximal run of non-equal ops."""

    index = 0
    while index < len(ops):
        if ops[index].op == "equal":
            index += 1
            continue
        end = index
        while end < len(ops) and ops[end].op != "equal":
            end += 1
        run = ops[index:end]
        yield (
            [op.ref_index for op in run if op.ref_token is not None],
            [op.hyp_index for op in run if op.hyp_token is not None],
        )
        index = end


def _check(
    rewrite: Rewrite,
    ref: Sequence[str],
    hyp: Sequence[str],
    ops: Sequence[AlignmentOp],
    anywhere: bool = False,
) -> None:
    """A rewrite must lie inside one error run, or be one stretch of the alignment holding an error (unless
    its step copies spellings from anywhere in the reference), and must name the tokens it replaces."""

    if not anywhere:
        ref_positions = set(range(rewrite.ref_start, rewrite.ref_end))
        hyp_positions = set(range(rewrite.hyp_start, rewrite.hyp_end))
        in_run = any(ref_positions <= set(r) and hyp_positions <= set(h) for r, h in error_runs(ops))
        touched = [
            k
            for k, op in enumerate(ops)
            if (op.ref_token is not None and op.ref_index in ref_positions)
            or (op.hyp_token is not None and op.hyp_index in hyp_positions)
        ]
        stretch = ops[touched[0] : touched[-1] + 1] if touched else []
        is_stretch = (
            {op.ref_index for op in stretch if op.ref_token is not None} == ref_positions
            and {op.hyp_index for op in stretch if op.hyp_token is not None} == hyp_positions
            and any(op.op != "equal" for op in stretch)
        )
        if not (in_run or is_stretch):
            raise ValueError(f"{rewrite.step}: rewrite is outside the alignment's errors: {rewrite}")
    if tuple(ref[rewrite.ref_start : rewrite.ref_end]) != rewrite.reference:
        raise ValueError(f"{rewrite.step}: reference span does not match: {rewrite}")
    if tuple(hyp[rewrite.hyp_start : rewrite.hyp_end]) != rewrite.hypothesis:
        raise ValueError(f"{rewrite.step}: hypothesis span does not match: {rewrite}")


class _Text:
    """A transcript as text and as tokens.

    A rewrite replaces the hypothesis's own characters for the span with the reference's own characters
    for its span, so everything else stays exactly as written (Thai tokens are not whitespace-delimited).
    """

    def __init__(
        self,
        text: str,
        language: str | None,
        tokenize: Callable[[str, str | None], list[str]] | None = None,
    ) -> None:
        self.text, self.language, self._tokenize = text, language, tokenize or word_tokens
        self.toks = self._tokenize(text, language)
        self.spans = _spans(text, self.toks)

    def piece(self, start: int, end: int) -> str:
        if self.spans is None:
            return detokenize(self.toks[start:end], self.language)
        return self.text[self.spans[start][0] : self.spans[end - 1][1]]

    def apply(self, ref: _Text, rewrites: Sequence[Rewrite]) -> _Text:
        if self.spans is None:  # a tokenizer that changes characters: fall back to joining tokens
            out = list(self.toks)
            for rewrite in sorted(rewrites, key=lambda r: r.hyp_start, reverse=True):
                out[rewrite.hyp_start : rewrite.hyp_end] = rewrite.reference
            return _Text(detokenize(out, self.language), self.language, self._tokenize)
        text = self.text
        for rewrite in sorted(rewrites, key=lambda r: r.hyp_start, reverse=True):
            start, end = self.spans[rewrite.hyp_start][0], self.spans[rewrite.hyp_end - 1][1]
            piece = ref.piece(rewrite.ref_start, rewrite.ref_end)
            # In text written without spaces, a Latin word copied next to another Latin word gets a space,
            # so the two stay two words ("amazon" + "prime", not "amazonprime").
            if start > 0 and _latin_edge(text[start - 1]) and _latin_edge(piece[:1]):
                piece = " " + piece
            if end < len(text) and _latin_edge(text[end]) and _latin_edge(piece[-1:]):
                piece = piece + " "
            text = text[:start] + piece + text[end:]
        return _Text(text, self.language, self._tokenize)


def _latin_edge(char: str) -> bool:
    return char.isascii() and char.isalnum()


def _chars(tokens: Sequence[str]) -> list[str]:
    return [char for token in tokens for char in token if not char.isspace()]


def _errors(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int]:
    """Word and character edit distance (CER counts non-whitespace code points)."""

    return align(ref, hyp)[1], align(_chars(ref), _chars(hyp))[1]


def _no_worse(after: tuple[int, int], before: tuple[int, int]) -> bool:
    return after[0] <= before[0] and after[1] <= before[1]


def reconcile(
    reference: str,
    hypothesis: str,
    language: str | None,
    steps: Sequence[PairStep],
    finish: Callable[[str], str] | None = None,
) -> PairNormalized:
    """Run the pair steps, in order, on two normalized transcripts. The reference comes back unchanged.

    ``finish`` maps a marked text to the text scoring reads (leading-zero marks); the guard counts errors
    on finished text, since a rewrite that looks harmless next to a mark can still cost an error once the
    mark is gone.
    """

    language = canonical_language(language)
    active = [step for step in steps if step.applies_to(language)]
    if not active:
        return PairNormalized(reference, hypothesis, ())
    hyp = _Text(hypothesis, language)
    as_scored = finish or (lambda text: text)
    ref_scored = word_tokens(as_scored(reference), language)

    def scored(text: _Text) -> tuple[int, int]:
        """Errors on the word tokens, whatever tokens the step works on."""

        return _errors(ref_scored, word_tokens(as_scored(text.text), language))

    done: list[Rewrite] = []
    declined: list[Rewrite] = []
    for step in active:
        tokenization = step.tokenization(language)
        tokenizer = TOKENIZATIONS[tokenization]
        ref, hyp = _Text(reference, language, tokenizer), _Text(hyp.text, language, tokenizer)
        ops, _ = align(ref.toks, hyp.toks)
        found = sorted(
            (
                replace(
                    rewrite,
                    language=language,
                    tokenization=tokenization,
                    reference_text=ref.text,
                    hypothesis_text=hyp.text,
                )
                for rewrite in step.rule(ref.toks, hyp.toks, ops, language)
            ),
            key=lambda r: r.hyp_start,
        )
        for first, second in zip(found, found[1:]):
            if second.hyp_start < first.hyp_end or (not step.anywhere and second.ref_start < first.ref_end):
                raise ValueError(f"{step.id}: overlapping rewrites {first} and {second}")
        for rewrite in found:
            _check(rewrite, ref.toks, hyp.toks, ops, step.anywhere)
        if not found:
            continue
        before = scored(hyp)
        whole = hyp.apply(ref, found)
        if _no_worse(scored(whole), before):
            kept, hyp = list(found), whole
        else:
            kept, errors, base = [], before, hyp
            for rewrite in found:
                candidate = base.apply(ref, [*kept, rewrite])
                after = scored(candidate)
                if _no_worse(after, errors):
                    kept.append(rewrite)
                    errors, hyp = after, candidate
                else:
                    declined.append(rewrite)
        done.extend(kept)
    return PairNormalized(reference, hyp.text, tuple(done), tuple(declined))


__all__ = ["PairNormalized", "PairStep", "Rewrite", "detokenize", "error_runs", "is_mark", "reconcile"]
