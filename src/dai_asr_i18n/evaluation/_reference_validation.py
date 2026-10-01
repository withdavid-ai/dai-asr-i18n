"""Validate aligned references and construct speaker activity regions.

Internal output times are milliseconds; public timeline adapters use seconds.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any

import regex


def _lexical_characters(text: str, language: str = "") -> str:
    # A diagnostic comparison, not an ASR scoring normalizer: disregard only punctuation,
    # whitespace, case and compatibility Unicode representation. Aligner preprocessing uses
    # NFKC (notably Thai SARA AM -> NIKHAHIT + SARA AA); this is not a lost lexical unit.
    # Keep accents and digits so an aligner dropping numerals remains a failure.
    text = unicodedata.normalize("NFKC", text).casefold()
    if language == "fr":
        text = text.translate(str.maketrans({"œ": "oe", "æ": "ae"}))
    return "".join(regex.findall(r"[\p{L}\p{M}\p{N}]", text))


def audit_alignment_document(document: dict, payload: dict, durations: dict[str, float]) -> dict[str, Any]:
    """Report identity, annotation coverage, interval validity, and potential lost reference text.

    No words are discarded or repaired here. A zero-duration or missing word remains an explicit
    quality issue for the caller to resolve before adopting a forced-timing scoring protocol.
    """
    expected = {}
    for speaker in payload["speakers"]:
        for chunk in speaker["chunks"]:
            expected[chunk["annotationId"]] = (str(speaker["sid"]), chunk)
    annotations = document.get("annotations") or []
    ids = [a.get("annotationId") for a in annotations]
    issues = []
    word_count = zero_duration = no_words = text_mismatches = invalid_intervals = 0
    if document.get("jobId") != payload["jobId"] or document.get("lang") != payload["lang"]:
        issues.append({"kind": "document_identity_mismatch"})
    if len(ids) != len(set(ids)):
        issues.append({"kind": "duplicate_annotation_ids"})
    missing = sorted(set(expected) - set(ids))
    extra = sorted(str(x) for x in set(ids) - set(expected))
    if missing or extra:
        issues.append({"kind": "annotation_coverage", "missing": missing, "extra": extra})
    for annotation in annotations:
        aid = annotation.get("annotationId")
        if aid not in expected:
            continue
        speaker_id, original = expected[aid]
        if str(annotation.get("speakerId")) != speaker_id or annotation.get("silverText") != original["silverText"]:
            issues.append({"kind": "annotation_identity_mismatch", "annotation": aid})
        if any(annotation.get(k) != original[k] for k in ("audioStart", "audioEnd")):
            issues.append({"kind": "annotation_window_mismatch", "annotation": aid})
        words = annotation.get("words") or []
        if not words and _lexical_characters(original["silverText"], payload["lang"]):
            no_words += 1
            issues.append({"kind": "no_words", "annotation": aid})
        returned_text = " ".join(str(w.get("text") or "") for w in words)
        if _lexical_characters(original["silverText"], payload["lang"]) != _lexical_characters(
            returned_text, payload["lang"]
        ):
            text_mismatches += 1
            issues.append({"kind": "lexical_text_mismatch", "annotation": aid})
        previous_start = -math.inf
        for index, word in enumerate(words):
            word_count += 1
            start, end = word.get("start"), word.get("end")
            valid_numbers = all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (start, end)
            )
            if not valid_numbers or not 0 <= start <= end <= durations[speaker_id] + 0.05:
                invalid_intervals += 1
                issues.append({"kind": "invalid_word_interval", "annotation": aid, "word": index})
                continue
            if start == end:
                zero_duration += 1
            if start < original["audioStart"] - 0.05 or end > original["audioEnd"] + 0.05:
                issues.append({"kind": "word_outside_annotation", "annotation": aid, "word": index})
            if start < previous_start:
                issues.append({"kind": "nonmonotonic_words", "annotation": aid, "word": index})
            previous_start = start
    return {
        "job_id": payload["jobId"],
        "language": payload["lang"],
        "aligner": document.get("aligner"),
        "expected_annotations": len(expected),
        "returned_annotations": len(annotations),
        "words": word_count,
        "zero_duration_words": zero_duration,
        "annotations_without_words": no_words,
        "lexical_text_mismatches": text_mismatches,
        "invalid_word_intervals": invalid_intervals,
        "issues": issues,
        "reported_failure_stats": document.get("failureStats"),
    }


def merge_speech_intervals(intervals: list[tuple[float, float]], max_gap_s: float = 0.2) -> list[tuple[float, float]]:
    """Union overlapping same-speaker words and join gaps of at most the declared threshold."""
    if not math.isfinite(max_gap_s) or max_gap_s < 0:
        raise ValueError("merge gap must be finite and nonnegative")
    merged = []
    for start, end in sorted(intervals):
        if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start < end:
            raise ValueError("speech intervals must be finite, nonnegative, and have positive duration")
        if merged and start <= merged[-1][1] + max_gap_s + 1e-12:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def forced_speech_regions(document: dict, payload: dict, durations: dict[str, float]):
    """Strict reference speech regions: complete text, positive intervals, merge gaps <= .20 s.

    A timestamp point supplies no measured speech duration. Refuse an artifact containing such
    words for DER rather than deleting them or fabricating a duration. Timed permutation scoring
    can independently accept points because its matching definition explicitly supports them.
    """
    audit = audit_alignment_document(document, payload, durations)
    if audit["issues"] or audit["zero_duration_words"]:
        raise ValueError("forced speech timing fails coverage, text, or positive-duration validation")
    intervals = {}
    for annotation in document["annotations"]:
        speaker = str(annotation["speakerId"])
        for word in annotation["words"]:
            if word["end"] > durations[speaker]:
                raise ValueError("forced speech interval exceeds audio duration")
            intervals.setdefault(speaker, []).append((word["start"], word["end"]))
    return [
        (speaker, start * 1000, end * 1000)
        for speaker, words in sorted(intervals.items())
        for start, end in merge_speech_intervals(words, max_gap_s=0.2)
    ]
