"""Leading zeros (English): a digit group that starts with 0 keeps its zeros.

Whisper's English number rules read a digit group as a number: "0556" becomes "556", "02134" becomes
"2134", "00" becomes "0". A phone number, zip code or code then loses digits on both sides, so a model
that dropped a real 0 is not charged for it.

The step marks each such group before the base stage, as a letters-only token the base leaves alone
("0556" becomes "zqzaffgq"), and :func:`finish` writes it back as it was. The mark is padded with
spaces, so the punctuation step treats the characters next to it the same whichever runs first.

Not marked: a lone "0" (Whisper keeps it); a group joined to a letter ("05th"); a decimal or grouped
number ("0.5", "0,5"). Clock times are marked like any group: "9:00" is "9 00" and "09:30 am" is
"09 30 am".
"""

from __future__ import annotations

import re
import unicodedata

LANGUAGES = frozenset({"en"})
_LETTERS = "abcdefghij"
# Any script's digits (``\d`` is Unicode): the step must not depend on whether native digits are folded yet.
TRIGGER = re.compile(r"(?<!\d)\d\d")
_GROUP = re.compile(r"(?<![\w.,])\d\d+(?!\w|[.,]\d)")
_MARK = re.compile(r"(?<![a-z])zqz([a-j]+)q(?![a-z])")


def _mark(match: re.Match[str]) -> str:
    group = match.group(0)
    if unicodedata.digit(group[0]) != 0:
        return group
    return " zqz" + "".join(_LETTERS[unicodedata.digit(char)] for char in group) + "q "


def mark_zeros(text: str, language: str | None) -> str:
    """Replace each digit group that starts with 0 with its mark."""

    return _GROUP.sub(_mark, text)


def finish(text: str, language: str | None) -> str:
    """Write each mark back as its digits."""

    return _MARK.sub(lambda match: "".join(str(_LETTERS.index(char)) for char in match.group(1)), text)


__all__ = ["LANGUAGES", "TRIGGER", "finish", "mark_zeros"]
