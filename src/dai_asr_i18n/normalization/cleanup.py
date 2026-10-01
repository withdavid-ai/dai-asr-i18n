"""Shared tag, annotation, parenthesis, and punctuation cleanup."""

from __future__ import annotations

import re
import unicodedata

from dai_asr_i18n.normalization.base import TextNormalizer

_WS = re.compile(r"\s+")

# A deliberately closed list of serialization artifacts observed in the benchmark source format.
_FORMATTING_ESCAPE = re.compile(r"\\+u(200[cCdD]|00[aA]0|3000)")
_PAREN_OPENERS = "⁽₍❨❪⟮⦅⸨⹙⹛﴿︵﹙（｟"
_PAREN_CLOSERS = "⁾₎❩❫⟯⦆⸩⹚⹜﴾︶﹚）｠"
_PAREN_TRANSLATION = {ord(c): ord("(") for c in _PAREN_OPENERS} | {ord(c): ord(")") for c in _PAREN_CLOSERS}
_PARENTHESES = re.compile(r"\([^()]*\)")

_ANNOTATION_TAG = re.compile(r"\[[^\[\]]{0,80}\]")
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_ANNOTATION_BRACE = re.compile(r"\{([^{}]{0,200})\}")
_ANNOTATION_SGML = re.compile(r"</?[^<>]{0,120}>")


def punctuation_to_spaces(text: str) -> str:
    """Replace Unicode punctuation and symbols while preserving combining marks."""

    return " ".join("".join(" " if unicodedata.category(char)[0] in "PS" else char for char in text).split())


def _unwrap_or_drop(match: re.Match[str]) -> str:
    body = match.group(1)
    return " " if _HAS_LETTER.search(body) else f" {body} "


class RegexTagRemover(TextNormalizer):
    """Remove square-bracket and SGML-style tags while preserving spoken SGML contents."""

    key = "regex_tags"

    def apply(self, text: str) -> str:
        text = _ANNOTATION_TAG.sub(" ", text)
        text = _ANNOTATION_SGML.sub(" ", text)
        return _WS.sub(" ", text).strip()


class AnnotationRemover(TextNormalizer):
    """Remove corpus event/gloss markup without inferring what was spoken."""

    key = "annotations"

    def __init__(self, tag_remover: TextNormalizer | None = None) -> None:
        self.tag_remover = tag_remover or RegexTagRemover()

    def apply(self, text: str) -> str:
        if not text:
            return text
        text = _ANNOTATION_BRACE.sub(_unwrap_or_drop, text)
        return self.tag_remover.apply(text)


class ParenthesisRemover(TextNormalizer):
    """Decode allowed escapes and remove nested Unicode parenthetical annotations."""

    key = "parentheses"

    def apply(self, text: str) -> str:
        text = _FORMATTING_ESCAPE.sub(lambda match: chr(int(match.group(1), 16)), text)
        text = text.translate(_PAREN_TRANSLATION)
        while _PARENTHESES.search(text):
            text = _PARENTHESES.sub(" ", text)
        return text.replace("(", " ").replace(")", " ")


_ANNOTATION_REMOVER = AnnotationRemover()
_PARENTHESIS_REMOVER = ParenthesisRemover()


def prepare_scoring_text(text: str) -> str:
    """Compatibility function for :class:`ParenthesisRemover`."""

    return _PARENTHESIS_REMOVER.apply(text)


def strip_annotations(text: str) -> str:
    """Compatibility function for :class:`AnnotationRemover`."""

    return _ANNOTATION_REMOVER.apply(text)


__all__ = [
    "AnnotationRemover",
    "ParenthesisRemover",
    "RegexTagRemover",
    "prepare_scoring_text",
    "punctuation_to_spaces",
    "strip_annotations",
]
