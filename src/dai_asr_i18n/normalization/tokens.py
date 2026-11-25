"""Word tokens for the pair steps, and the edit alignment they read."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache

import jiwer

# Han, kana and Hangul: one token per character in the CJK-aware tokenization.
_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # Extension A
    (0x20000, 0x2A6DF),  # Extension B
    (0x2A700, 0x2B73F),  # Extension C
    (0x2B740, 0x2B81F),  # Extension D
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0xAC00, 0xD7AF),  # Hangul Syllables
)


def is_cjk_char(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in _CJK_RANGES)


def cjk_aware_tokens(text: str) -> list[str]:
    """Every Han, kana or Hangul character is a token; other text is split on whitespace."""

    tokens: list[str] = []
    buffer: list[str] = []
    for char in text:
        if is_cjk_char(char):
            if buffer:
                tokens.extend("".join(buffer).split())
                buffer = []
            tokens.append(char)
        else:
            buffer.append(char)
    if buffer:
        tokens.extend("".join(buffer).split())
    return tokens


@cache
def _thai_word_tokenize():
    from pythainlp.tokenize import word_tokenize

    return word_tokenize


def thai_words(text: str) -> list[str]:
    """Thai words from PyThaiNLP's default word tokenizer, whitespace tokens included."""

    return _thai_word_tokenize()(text)


def word_tokens(text: str, language: str | None) -> list[str]:
    """See :func:`tokenize` with ``tokenization="word"``."""

    return tokenize(text, language, "word")


def _word_tokens(text: str, language: str | None) -> list[str]:
    """The word tokens the pair steps read and their guard counts word errors on.

    Whitespace for spaced languages; PyThaiNLP words for Thai; the CJK-aware tokenization for Japanese
    and Chinese, which are written without spaces.
    """

    if language == "th":
        return [token for token in thai_words(text) if token.strip()]
    if language in {"ja", "zh"}:
        return cjk_aware_tokens(text)
    return text.split()


# The tokenizations a pair step can read a text in, by name. A ``Rewrite`` records the name, so its token
# positions can be reproduced from its texts with :func:`tokenize`.
TOKENIZATIONS: dict[str, Callable[[str, str | None], list[str]]] = {
    "word": _word_tokens,
    "cjk_character": lambda text, language: cjk_aware_tokens(text),
}


def tokenize(text: str, language: str | None, tokenization: str = "word") -> list[str]:
    """Split a normalized text the way a pair step reads it.

    ``"word"``: whitespace for spaced languages, PyThaiNLP words for Thai, and one token per Han, kana or
    Hangul character for Japanese and Chinese (other text split on whitespace). ``"cjk_character"``: one
    token per Han, kana or Hangul character in any language, other text split on whitespace; the
    transliteration step reads Japanese this way.
    """

    try:
        return TOKENIZATIONS[tokenization](text, language)
    except KeyError:
        raise ValueError(f"unknown tokenization: {tokenization!r}") from None


@dataclass(frozen=True, slots=True)
class AlignmentOp:
    """One edit operation between reference and hypothesis tokens."""

    op: str  # "equal", "substitute", "delete" or "insert"
    ref_token: str | None
    hyp_token: str | None
    ref_index: int | None = None
    hyp_index: int | None = None


class _PreserveTokens(jiwer.AbstractTransform):
    def process_list(self, inp: list[str]) -> list[list[str]]:
        return [inp]


_PRESERVE_TOKENS = _PreserveTokens()


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> tuple[list[AlignmentOp], int]:
    """The edit operations aligning two token sequences, and their error count (S + D + I)."""

    if not reference and not hypothesis:
        return [], 0
    if not hypothesis:
        return [AlignmentOp("delete", token, None, index, None) for index, token in enumerate(reference)], len(
            reference
        )
    if not reference:
        return [AlignmentOp("insert", None, token, None, index) for index, token in enumerate(hypothesis)], len(
            hypothesis
        )
    output = jiwer.process_words(
        list(reference),
        list(hypothesis),
        reference_transform=_PRESERVE_TOKENS,
        hypothesis_transform=_PRESERVE_TOKENS,
    )
    ref_seq, hyp_seq = output.references[0], output.hypotheses[0]
    ops: list[AlignmentOp] = []
    for chunk in output.alignments[0]:
        if chunk.type in {"equal", "substitute"}:
            for offset in range(chunk.ref_end_idx - chunk.ref_start_idx):
                ref_index, hyp_index = chunk.ref_start_idx + offset, chunk.hyp_start_idx + offset
                ops.append(AlignmentOp(chunk.type, ref_seq[ref_index], hyp_seq[hyp_index], ref_index, hyp_index))
        elif chunk.type == "delete":
            for ref_index in range(chunk.ref_start_idx, chunk.ref_end_idx):
                ops.append(AlignmentOp("delete", ref_seq[ref_index], None, ref_index, None))
        elif chunk.type == "insert":
            for hyp_index in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                ops.append(AlignmentOp("insert", None, hyp_seq[hyp_index], None, hyp_index))
    return ops, output.substitutions + output.deletions + output.insertions


__all__ = ["TOKENIZATIONS", "AlignmentOp", "align", "cjk_aware_tokens", "thai_words", "tokenize", "word_tokens"]
