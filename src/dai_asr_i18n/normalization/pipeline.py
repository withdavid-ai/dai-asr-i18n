"""The benchmark normalizer: each language's base transform plus one small, independent step per rule.

The pipeline for one text:

    prepare        decode serialization escapes, remove ja/zh spaces between Han and kana, strip annotations
    stage -1       the steps that must see the text before the base stage (apostrophes, spaced clitics)
    base           the language transform without its punctuation pass, which the punctuation step
                   replaces, and without "(...)" deletion, which the event-parentheses step decides
                   (English: :class:`~dai_asr_i18n.normalization.english.EnglishBase`)
    stage 0, 1     the other :data:`STEPS`; a stage-1 step reads what a stage-0 step can create
    english-late   English only: Whisper's symbol removal and number reading
    finalize       ja/zh spacing again, collapse whitespace
    finish         each step's ``finish``: write out the marks a step left (leading zeros)

Each step is a pure ``str -> str``. The steps of one stage touch disjoint characters, so they run in any
order. Normalizing an output again changes nothing.

Scoring then reads each hypothesis against its reference with the :data:`PAIR_STEPS` (see
:mod:`dai_asr_i18n.normalization.pair`): rules that need both transcripts. Transliteration rewrites a loanword
into the reference's spelling of it in the other script, and in German, Tagalog and Turkish a difference
of only apostrophes is settled. They rewrite only the hypothesis, only into a spelling the reference
writes, and never add an error. See :meth:`DaiAsrI18nNormalizer.normalize_pair`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache, cached_property, lru_cache
from typing import Protocol

from dai_asr_i18n.languages import canonical_language
from dai_asr_i18n.normalization import apostrophe_pairs, characters, parentheses, punctuation, transliteration, zeros
from dai_asr_i18n.normalization.base import NormalizedText, TextNormalizer
from dai_asr_i18n.normalization.cleanup import CJK_SPACE, decode_formatting_escapes, strip_annotations
from dai_asr_i18n.normalization.english import EnglishBase, english_late
from dai_asr_i18n.normalization.languages import CoreThenPunctuation, MarkPreservingNormalizer
from dai_asr_i18n.normalization.pair import PairNormalized, PairStep, reconcile
from dai_asr_i18n.normalization.resolver import normalizer_for_key
from dai_asr_i18n.policies import NORMALIZATION_POLICY_ID

PROFILE = "dai_asr_i18n"
BASE = "base"
LATE = "english-late"
CJK_SPACED = frozenset({"ja", "zh"})
_WS = re.compile(r"\s+")
_BRACKETS = re.compile(r"[<\[][^>\]]*[>\]]")

# The language transform each benchmark language's base stage comes from; Whisper Basic elsewhere.
BASE_NORMALIZERS: dict[str, str] = {
    "en": "whisper_english",
    **{language: "unicode_marks" for language in ("es", "fr", "de", "it", "pt", "ru", "id", "tl", "vi")},
    "tr": "turkic",
    "ar": "arabic",
    **{language: f"indic_conservative_{language}" for language in ("hi", "mr", "bn", "ta", "te")},
    "zh": "chinese_t2s",
    "ja": "cjk",
    "ko": "cjk",
    "th": "thai",
}


class Trigger(Protocol):
    def search(self, text: str) -> object: ...


@dataclass(frozen=True)
class Step:
    """One rule: a pure, idempotent ``str -> str``, applied to reference and hypothesis alike.

    ``languages`` limits the step to those languages (None means every language). ``trigger`` matches every
    input the step can change. A step that leaves marks (leading zeros) has a ``finish`` that writes them
    out, after the pair steps. ``stage`` orders the steps: -1 before the base stage, then 0, then 1.
    """

    id: str
    rule: Callable[[str, str | None], str]
    trigger: Trigger
    languages: frozenset[str] | None = None
    finish: Callable[[str, str | None], str] | None = None
    name: str = field(kw_only=True)
    description: str = field(kw_only=True)
    stage: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.description.strip():
            raise ValueError(f"step {self.id!r} needs a name and a description")

    def applies_to(self, language: str | None) -> bool:
        return self.languages is None or language in self.languages

    def __call__(self, text: str, language: str | None) -> str:
        return self.rule(text, language) if self.applies_to(language) else text


STEPS: tuple[Step, ...] = (
    Step(
        "event-parentheses",
        parentheses.resolve_parentheses,
        parentheses.TRIGGER,
        name="Event parentheses",
        description=(
            'Removes a parenthetical only when it names a sound or event from the event list, like "(laughs)"; '
            'other parentheses are dropped and their contents kept, so "(02) 8012" keeps its area code.'
        ),
    ),
    # Before the base stage: NFKC turns ´ into a space and a combining accent, and Whisper's symbol
    # removal turns ’ into a space.
    Step(
        "apostrophes",
        characters.unify_apostrophes,
        characters.APOSTROPHE_TRIGGER,
        stage=-1,
        name="Apostrophes",
        description="Writes every apostrophe look-alike (’ ‘ ʼ ´ ` ′) as the straight apostrophe.",
    ),
    Step(
        "spaced-clitics",
        punctuation.attach_spaced_clitics,
        punctuation.CLITIC_TRIGGER,
        punctuation.CLITIC_LANGUAGES,
        stage=-1,
        name="Spaced clitics",
        description=(
            'Attaches a clitic written with a space before its apostrophe: English "I \'m" becomes "I\'m", '
            'French and Italian "c \'est" becomes "c\'est".'
        ),
    ),
    # Turkish needs no step: its base transform already applies NFC, then İ -> i and I -> ı, before
    # lowercasing.
    Step(
        "russian-yo",
        characters.fold_russian_yo,
        characters.RUSSIAN_TRIGGER,
        frozenset({"ru"}),
        name="Russian ё",
        description='Writes ё as е, so "ещё" and "еще" are the same word.',
    ),
    # Stage 1: the punctuation step can bring two marks together ("เ'เ", where the apostrophe is deleted).
    Step(
        "thai-marks",
        characters.reorder_thai_marks,
        characters.THAI_TRIGGER,
        frozenset({"th"}),
        stage=1,
        name="Thai mark order",
        description=(
            "Puts Thai vowel and tone marks in one order (เเ to แ, ํา to ำ, a space before a mark removed), so a "
            "syllable typed in another order still matches. No mark or repeated vowel is removed."
        ),
    ),
    Step(
        "japanese-characters",
        characters.japanese_characters,
        characters.JAPANESE_TRIGGER,
        frozenset({"ja"}),
        name="Japanese characters",
        description=(
            "Applies NFKC (full- and half-width forms become one) and deletes hyphen and "
            'tilde look-alikes and decorative symbols: "コーヒー〜♪" becomes "コーヒー". 〒 is kept.'
        ),
    ),
    Step(
        "japanese-long-vowels",
        characters.long_vowel_runs,
        characters.LONG_VOWEL_TRIGGER,
        frozenset({"ja"}),
        stage=1,
        name="Japanese long vowels",
        description=(
            'Keeps one ー per run: "コーーヒー" becomes "コーヒー". It runs after punctuation and the deletions, '
            'which can bring two ー together ("ー{ー").'
        ),
    ),
    Step(
        "native-digits",
        characters.fold_digits,
        characters.DIGIT_TRIGGER,
        name="Native digits",
        description='Writes every Unicode decimal digit as its ASCII digit: "๓", "٣" and "३" become "3".',
    ),
    Step(
        "punctuation",
        punctuation.strip_punctuation,
        punctuation.TRIGGER,
        name="Punctuation",
        description=(
            "Removes punctuation. An apostrophe or hyphen stays only with a letter on both sides "
            '("l\'homme", "run-of-the-mill"); a cut-off word loses its hyphen ("wa-" becomes "wa").'
        ),
    ),
    Step(
        "leading-zeros",
        zeros.mark_zeros,
        zeros.TRIGGER,
        zeros.LANGUAGES,
        finish=zeros.finish,
        name="Leading zeros",
        description=(
            'Keeps the zeros of an English digit group that starts with 0 ("0556", "02134", "007"), which '
            "Whisper's number rules would otherwise read as a number and drop."
        ),
    ),
)

# Rules that read the hypothesis against its reference, in this order, after both are normalized. Each
# step reads what the earlier ones left.
PAIR_STEPS: tuple[PairStep, ...] = (
    PairStep(
        transliteration.STEP,
        transliteration.transliteration_rewrites,
        transliteration.LANGUAGES,
        name="Transliteration",
        description=(
            'A model loanword takes the reference\'s spelling of it in the other script ("computer" -> "कंप्यूटर", '
            '"জিবিটি" -> "gpt"), from the spellings the reference uses anywhere in the text and the '
            "transliteration maps."
        ),
        anywhere=True,
        tokenization=transliteration.tokenization,
    ),
    PairStep(
        apostrophe_pairs.STEP,
        apostrophe_pairs.apostrophe_rewrites,
        apostrophe_pairs.LANGUAGES,
        name="Apostrophes against the reference",
        description=(
            "In German, Tagalog and Turkish, a model stretch that differs from the reference only by apostrophes "
            'and spaces ("aus\'m" / "ausm", "sa\'yo" / "sayo", "türkçe\'yi" / "türkçeyi") takes the '
            "reference's spelling."
        ),
    ),
)


def base_key(language: str | None) -> str:
    """The language transform the base stage comes from."""

    return BASE_NORMALIZERS.get(canonical_language(language) or "", "whisper_basic")


def _mark_preserving_core(text: str) -> str:
    """:class:`MarkPreservingNormalizer` core without "(...)" deletion: lowercase, bracket spans, NFKC."""

    return unicodedata.normalize("NFKC", _BRACKETS.sub("", text.lower())).lower()


class DaiAsrI18nNormalizer(TextNormalizer):
    """The benchmark normalizer for one language. ``apply`` returns :class:`NormalizedText`."""

    def __init__(self, language: str | None) -> None:
        self.language = canonical_language(language)
        self.profile = f"{NORMALIZATION_POLICY_ID}:{PROFILE}:{self.language or 'default'}"
        self.key = f"{PROFILE}:{base_key(self.language)}"

    @cached_property
    def _core(self) -> Callable[[str], str]:
        """The language's transform without its punctuation pass, which the punctuation step replaces."""

        if self.language == "en":
            return EnglishBase()
        inner = normalizer_for_key(base_key(self.language))
        if isinstance(inner, MarkPreservingNormalizer):
            return _mark_preserving_core
        # Whisper Basic, the fallback outside the benchmark languages, has no separate punctuation pass, so
        # it keeps removing every apostrophe and hyphen.
        return inner.core if isinstance(inner, CoreThenPunctuation) else inner.apply

    @property
    def steps(self) -> tuple[Step, ...]:
        """The steps that do anything for this language."""

        return tuple(step for step in STEPS if step.applies_to(self.language))

    def stage_groups(self) -> tuple[tuple[str, ...], ...]:
        """The default order in groups whose members run in any order among themselves."""

        before = [step for step in self.steps if step.stage < 0]
        after = [step for step in self.steps if step.stage >= 0]
        groups = [
            *(tuple(s.id for s in before if s.stage == stage) for stage in sorted({s.stage for s in before})),
            (BASE,),
            *(tuple(s.id for s in after if s.stage == stage) for stage in sorted({s.stage for s in after})),
            (LATE,) if self.language == "en" else (),
        ]
        return tuple(group for group in groups if group)

    def default_order(self) -> tuple[str, ...]:
        return tuple(name for group in self.stage_groups() for name in group)

    def apply(self, text: str) -> NormalizedText:
        if isinstance(text, NormalizedText):
            if text.profile != self.profile:
                raise ValueError("normalized text belongs to another profile; normalize the original text")
            return text
        return NormalizedText(self.normalize(text), self.profile)

    def normalize(self, text: str, order: Sequence[str] | None = None) -> str:
        """Normalize ``text``. ``order`` permutes the base stage and the steps (for order tests)."""

        return self.finish(self.normalize_marked(text, order))

    def normalize_marked(self, text: str, order: Sequence[str] | None = None) -> str:
        """:meth:`normalize` before :meth:`finish`: the marks some steps leave are still there."""

        if order is None:
            return _normalize_cached(self.language, text)
        return self._normalize(text, order)

    def finish(self, text: str) -> str:
        """Write out the marks the steps left (each marking step's ``finish``), then :meth:`finalize`."""

        for step in self.steps:
            if step.finish is not None:
                text = step.finish(text, self.language)
        return self.finalize(text)

    def _normalize(self, text: str, order: Sequence[str] | None) -> str:
        stages: dict[str, Callable[[str], str]] = {BASE: self._core}
        if self.language == "en":
            stages[LATE] = english_late
        stages.update({step.id: (lambda value, step=step: step(value, self.language)) for step in self.steps})
        order = self.default_order() if order is None else tuple(order)
        if sorted(order) != sorted(stages):
            raise ValueError(f"order must name each of {sorted(stages)} once, got {list(order)}")
        text = self.prepare(text)
        for name in order:
            text = stages[name](text)
        return self.finalize(text)

    def normalize_pair(self, reference: str, hypothesis: str) -> PairNormalized:
        """Normalize both transcripts, then read the hypothesis against the reference (:data:`PAIR_STEPS`).

        The reference is exactly ``normalize(reference)``; the hypothesis differs from
        ``normalize(hypothesis)`` only by the returned rewrites.
        """

        out = reconcile(
            self.normalize_marked(reference),
            self.normalize_marked(hypothesis),
            self.language,
            PAIR_STEPS,
            finish=self.finish,
        )
        return PairNormalized(self.finish(out.reference), self.finish(out.hypothesis), out.rewrites, out.declined)

    def prepare(self, text: str) -> str:
        """Decode escapes, remove ja/zh spaces between Han and kana, and strip annotations.

        Parentheses stay: the event-parentheses step decides what to remove.
        """

        text = decode_formatting_escapes(text)
        if self.language in CJK_SPACED:
            text = CJK_SPACE.sub("", text)
        return strip_annotations(text)

    def finalize(self, text: str) -> str:
        if self.language in CJK_SPACED:
            text = CJK_SPACE.sub("", text)
        return _WS.sub(" ", text).strip()


@cache
def dai_asr_i18n_normalizer(language: str | None) -> DaiAsrI18nNormalizer:
    """One cached :class:`DaiAsrI18nNormalizer` per language."""

    return DaiAsrI18nNormalizer(language)


@lru_cache(maxsize=16384)
def _normalize_cached(language: str | None, text: str) -> str:
    """One normalization per (language, text): scoring reads a reference once per hypothesis, and the pair
    steps read both texts again."""

    return dai_asr_i18n_normalizer(language)._normalize(text, None)


def step_manifest() -> dict[str, object]:
    """The steps and pair steps, for provenance."""

    def entry(step: Step | PairStep) -> dict[str, object]:
        return {
            "id": step.id,
            "name": step.name,
            "languages": sorted(step.languages or []),
            "description": step.description,
        }

    return {
        "steps": [entry(step) for step in STEPS],
        "pair_steps": [entry(step) for step in PAIR_STEPS],
        "base_normalizers": dict(sorted(BASE_NORMALIZERS.items())),
    }


__all__ = [
    "BASE_NORMALIZERS",
    "PAIR_STEPS",
    "PROFILE",
    "STEPS",
    "DaiAsrI18nNormalizer",
    "PairNormalized",
    "Step",
    "base_key",
    "dai_asr_i18n_normalizer",
    "step_manifest",
]
