"""Validate prepared connector predictions without changing their text or timestamps."""

from __future__ import annotations

from collections.abc import Mapping

from dai_asr_i18n.evaluation._hypothesis_timing import PROTOCOL, _hyp_by_speaker, _hyp_timeline, gemini_timeline
from dai_asr_i18n.scoring.diarization import DiarizationSegment

GEMINI_TIMING_POLICY = PROTOCOL


def validate_hypothesis_blob(blob: Mapping) -> None:
    """Reject an unknown schema instead of treating it as successful empty output."""
    if not isinstance(blob, Mapping):
        raise ValueError("hypothesis must be a prepared prediction artifact, or {} for a successful empty output")
    if not blob:
        return
    verbatim = blob.get("verbatim")
    if not isinstance(verbatim, Mapping) or not isinstance(verbatim.get("segments"), list):
        raise ValueError(
            "expected verbatim.segments; convert local inference records with hypothesis_from_local_record"
        )
    for segment in verbatim["segments"]:
        if not isinstance(segment, Mapping) or not isinstance(segment.get("text", ""), str):
            raise ValueError("hypothesis segments must be objects with string text")


def hypothesis_from_local_record(record: Mapping) -> dict:
    """Adapt an ASR-only `run` JSONL record for lexical scoring, without inventing timing.

    The local ASR interface has no speaker attribution. Its complete text is kept
    at the physical channel; this adapter must not be used to claim diarization.
    """
    channel = record.get("channel")
    transcript = record.get("transcript")
    if channel not in {"ch1", "ch2", "mono"} or not isinstance(transcript, Mapping):
        raise ValueError("expected a successful local inference record with channel and transcript")
    if not isinstance(transcript.get("text"), str):
        raise ValueError("local transcript.text must be a string")
    return {
        "verbatim": {
            "segments": [
                {
                    "channel_index": 2 if channel == "ch2" else 1,
                    "text": transcript["text"],
                    "start_ms": 0.0,
                    "end_ms": 0.0,
                }
            ]
        }
    }


def hypothesis_timeline(blob: Mapping, *, provider: str | None = None) -> tuple[DiarizationSegment, ...]:
    """Convert prediction intervals from milliseconds to seconds.

    Explicit ``provider='gemini_transcribe'`` enables the published reversed-word
    omission policy for mixed audio only. It requires the original provider response
    to match the prepared prediction. All other providers use strict timing validation.
    The result is a speaker-activity timeline for diarization scoring.
    """
    validate_hypothesis_blob(blob)
    if provider == "gemini_transcribe":
        if blob.get("physical_channel_blobs"):
            raise ValueError("Gemini omission policy applies only to mixed audio")
        timeline = gemini_timeline(blob)
    else:
        timeline = _hyp_timeline(blob)
    return tuple(DiarizationSegment(speaker, start / 1000, end / 1000) for speaker, start, end in timeline)


def hypothesis_text_by_speaker(blob: Mapping) -> dict[str, str]:
    """Join prepared transcript segments in their original order, independent of timing validity."""
    validate_hypothesis_blob(blob)
    return _hyp_by_speaker(blob)
