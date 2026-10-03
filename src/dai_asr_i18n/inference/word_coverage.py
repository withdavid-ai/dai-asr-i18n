"""Validate native text coverage without rewriting text or word annotations."""

import unicodedata
from collections import defaultdict
from collections.abc import Iterable


def word_text_coverage_complete(phrases: Iterable[tuple[str, str]], words: Iterable[tuple[str, str]]) -> bool:
    """Compare each speaker's ordered text with explicit formatting equivalences.

    NFC, casefold, whitespace and Unicode punctuation are validation-only.
    Preserve marks, symbols and digits; do not apply NFKC, a scoring normalizer,
    number spelling, script conversion, or any timestamp/speaker transformation.
    Phrase boundaries may differ from the native word-list boundaries.
    """

    def streams(parts):
        grouped = defaultdict(list)
        for speaker, text in parts:
            grouped[str(speaker)].append(text)
        result = {}
        for speaker, texts in grouped.items():
            value = unicodedata.normalize("NFC", " ".join(texts)).casefold()
            value = unicodedata.normalize(
                "NFC", "".join(c for c in value if not c.isspace() and unicodedata.category(c)[0] != "P")
            )
            if value:
                result[speaker] = value
        return result

    return streams(phrases) == streams(words)
