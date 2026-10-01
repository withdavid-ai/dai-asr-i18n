"""Timing validation for speaker-attributed connector predictions.

This is not a model client. The Gemini exception requires the original response
and validates it against the stored timing/speaker boundaries before omission.
All internal timing tuples are milliseconds; public APIs return seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

PROTOCOL = "gemini-invalid-words-missing-v1"
MODEL = "gemini_transcribe"


@dataclass
class TimedText:
    speaker: str
    start_s: float
    end_s: float
    text: str

    def __post_init__(self):
        if (
            any(
                isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
                for t in (self.start_s, self.end_s)
            )
            or not 0 <= self.start_s <= self.end_s
        ):
            raise ValueError("timed text requires finite, nonnegative, ordered timestamps")


def _speaker_segments_from_words(words, ch_idx):
    segments = []
    for word in words:
        speaker = f"{ch_idx}:{word.get('speaker')}"
        start, end = word.get("start", 0.0) or 0.0, word.get("end", 0.0) or 0.0
        if segments and segments[-1].speaker_external_id == speaker:
            segments[-1].start_ms = min(segments[-1].start_ms, start * 1000.0)
            segments[-1].end_ms = max(segments[-1].end_ms, end * 1000.0)
        else:
            segments.append(SimpleNamespace(speaker_external_id=speaker, start_ms=start * 1000.0, end_ms=end * 1000.0))
    return segments


def _speaker_label(segment: dict) -> str:
    for key in ("speaker_external_id", "channel_index"):
        if segment.get(key) is not None:
            return str(segment[key])
    return "1"


def _hyp_by_speaker(blob: dict) -> dict[str, str]:
    """Group segments by speaker labels for permutation scoring of diarized mono output."""
    segs = ((blob or {}).get("verbatim") or {}).get("segments") or []
    out: dict[str, str] = {}
    for s in segs:
        spk = _speaker_label(s)
        out[spk] = (out.get(spk, "") + " " + (s.get("text") or "")).strip()
    return out


def _validate_hypothesis_word_timing(blob: dict) -> None:
    """Do not let grouping hide an invalid provider word inside a valid speaker turn."""

    for channel_blob in blob.get("physical_channel_blobs", []):
        _validate_hypothesis_word_timing(channel_blob)
    for metadata in (blob.get("provider_meta") or {}).values():
        if isinstance(metadata, dict) and metadata.get("word_alignment_complete") is False:
            raise ValueError("provider word alignment incomplete; lexical transcript retained")
    for word in blob.get("words") or []:
        TimedText(str(word.get("channel_index", "")), word.get("start_ms"), word.get("end_ms"), "")

    # Legacy text-only responses were represented as a transcript at [0, 0].
    # That is a storage fallback, not evidence that every word occurred at time zero.
    # Check each physical channel so a valid partner cannot hide missing timing in dual.
    if not blob.get("words"):
        by_channel: dict[str, list[dict]] = {}
        for segment in (blob.get("verbatim") or {}).get("segments", []):
            by_channel.setdefault(str(segment.get("channel_index", "")), []).append(segment)
        for segments in by_channel.values():
            if any(s.get("text", "").strip() for s in segments) and all(
                s.get("start_ms") == 0 and s.get("end_ms") == 0 for s in segments
            ):
                raise ValueError("text-only hypothesis has no usable timestamp interval")


def _hyp_timeline(blob: dict) -> list[tuple[str, float, float]]:

    _validate_hypothesis_word_timing(blob)
    segs = ((blob or {}).get("verbatim") or {}).get("segments") or []
    intervals = [TimedText(_speaker_label(s), s.get("start_ms"), s.get("end_ms"), "") for s in segs]
    return [(i.speaker, i.start_s, i.end_s) for i in intervals]


def seconds(value: str | float) -> float:
    result = float(value.removesuffix("s") if isinstance(value, str) else value)
    if not 0 <= result < float("inf"):
        raise ValueError("Gemini returned invalid time offset")
    return result


def parse_response(raw: dict, language: str, *, require_speakers: bool = True) -> dict:
    if raw.get("status") != "completed":
        raise ValueError(f"Gemini transcription not completed: {raw.get('status')}")
    texts, words = [], []
    missing_speakers = missing_times = 0
    for step in raw.get("steps", []):
        if step.get("type") != "model_output":
            continue
        for content in step.get("content", []):
            if content.get("type") != "text":
                continue
            texts.append(content.get("text", ""))
            for word in content.get("annotations", []):
                if word.get("type") != "word_info":
                    continue
                if word.get("start_offset") is None or word.get("end_offset") is None:
                    missing_times += 1
                    if require_speakers:
                        raise ValueError("Gemini word lacks requested time offsets")
                    # Preserve lexical text on a physical channel. A zero span is an explicit
                    # invalid-timing placeholder; temporal validation will reject this word.
                    a = b = 0.0
                else:
                    a, b = seconds(word["start_offset"]), seconds(word["end_offset"])
                if word.get("speaker") is None or not str(word["speaker"]).strip():
                    missing_speakers += 1
                    if require_speakers:
                        raise ValueError("Gemini word lacks speaker attribution")
                # Preserve reversed provider intervals. Text/speaker scoring remains valid;
                # the scoring input validator makes temporal metrics unavailable for them.
                words.append(
                    {
                        "text": word["text"],
                        "start": a,
                        "end": b,
                        "speaker": str(word["speaker"]) if word.get("speaker") is not None else None,
                    }
                )
    text = raw.get("output_text") or "".join(texts)
    if require_speakers and text.strip() and not words:
        raise ValueError("Gemini returned text without requested word/speaker annotations")
    # Speaker labels are not required for physical-channel ASR. If the API omitted
    # any, retain its complete text instead of inventing an unknown/adjacent speaker.
    if missing_speakers:
        words = [{**word, "speaker": None} for word in words]
    return {
        "text": text,
        "words": words,
        "model": MODEL,
        "language": language,
        "annotation_quality": {
            "missing_speaker_words": missing_speakers,
            "missing_timing_words": missing_times,
            "speaker_attribution_required": require_speakers,
        },
    }


def gemini_timeline(blob):
    """Omit reversed provider words, splitting turns at each omission.

    Successful strict timelines are returned verbatim. For malformed timelines,
    require the original provider response to reproduce the stored word timings and
    speaker-turn boundaries before deriving anything. No text or artifact is edited.
    """

    try:
        return _hyp_timeline(blob)
    except (ValueError, TypeError):
        pass
    metadata = blob.get("provider_meta") or {}
    if len(metadata) != 1 or blob.get("physical_channel_blobs"):
        raise ValueError("Gemini timing policy requires one mixed-audio response")
    channel, value = next(iter(metadata.items()))
    if not isinstance(value, dict) or not isinstance(value.get("response"), dict):
        raise ValueError("Gemini timing policy requires the original provider response")
    if value.get("word_alignment_complete") is False:
        raise ValueError("Incomplete Gemini word timing is outside the reversed-word policy")
    try:
        words = parse_response(value["response"], "", require_speakers=True)["words"]
    except KeyError as error:
        raise ValueError("Gemini raw response is missing required word annotations") from error
    stored = [(w.get("start_ms"), w.get("end_ms")) for w in blob.get("words", [])]
    if stored != [(w["start"] * 1000, w["end"] * 1000) for w in words]:
        raise ValueError("Gemini raw/stored word timing mismatch")
    original = _speaker_segments_from_words(words, int(channel))
    expected = [(s.speaker_external_id, s.start_ms, s.end_ms) for s in original]
    actual = [(_speaker_label(s), s.get("start_ms"), s.get("end_ms")) for s in blob["verbatim"]["segments"]]
    if expected != actual:
        raise ValueError("Gemini raw/stored speaker timeline mismatch")
    if not any(w["end"] < w["start"] for w in words):
        raise ValueError("Gemini invalid timing is outside the reversed-word policy")
    result, run = [], []

    def flush():
        result.extend(
            (s.speaker_external_id, s.start_ms, s.end_ms) for s in _speaker_segments_from_words(run, int(channel))
        )
        run.clear()

    for word in words:
        if word["end"] < word["start"]:
            flush()
        else:
            run.append(word)
    flush()
    return result
