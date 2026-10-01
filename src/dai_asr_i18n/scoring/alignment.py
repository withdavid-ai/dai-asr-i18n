"""Tokenization and Levenshtein alignment primitives."""

from __future__ import annotations

from collections.abc import Sequence

import jiwer

from dai_asr_i18n.scoring.models import EditCounts, Metric


class _PreserveTokens(jiwer.AbstractTransform):
    def process_list(self, inp: list[str]) -> list[list[str]]:
        return [inp]


_PRESERVE_TOKENS = _PreserveTokens()


def tokens_for_metric(text: str, metric: Metric) -> list[str]:
    """Split normalized text into the declared metric unit."""

    if metric == "cer":
        return [char for char in text if not char.isspace()]
    return text.split()


def align_tokens(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Return deterministic S/D/I/H counts for two pre-tokenized sequences."""

    if not reference and not hypothesis:
        return EditCounts()
    if not hypothesis:
        return EditCounts(deletions=len(reference))
    if not reference:
        return EditCounts(insertions=len(hypothesis))
    output = jiwer.process_words(
        list(reference),
        list(hypothesis),
        reference_transform=_PRESERVE_TOKENS,
        hypothesis_transform=_PRESERVE_TOKENS,
    )
    return EditCounts(
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
    )


__all__ = ["align_tokens", "tokens_for_metric"]
