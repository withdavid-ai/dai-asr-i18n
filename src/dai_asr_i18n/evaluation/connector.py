"""Score fresh connector responses against the dataset's aligned references."""

import json
from dataclasses import asdict
from pathlib import Path

from dai_asr_i18n.datasets.models import BenchmarkClip
from dai_asr_i18n.evaluation._reference_validation import merge_speech_intervals
from dai_asr_i18n.evaluation.references import BenchmarkReference
from dai_asr_i18n.evaluation.scoring import score_sample
from dai_asr_i18n.inference.base import Transcript
from dai_asr_i18n.inference.word_coverage import word_text_coverage_complete
from dai_asr_i18n.scoring import score_pair
from dai_asr_i18n.scoring.diarization import DiarizationSegment


def reference_from_clip(clip: BenchmarkClip) -> BenchmarkReference:
    """Use only supplied aligned words, including their unchanged timestamps."""
    if set(clip.word_alignments) != set(clip.audio):
        raise ValueError("scoring requires aligned references for every source channel")
    texts, timeline, counts, ordered = {}, [], [], []
    for channel, segments in sorted(clip.word_alignments.items()):
        speaker = channel.removeprefix("ch")
        words = [word for segment in segments for word in segment.words]
        texts[speaker] = " ".join(word.text for word in words)
        for word in words:
            if word.end_s > clip.audio[channel].duration_s + 0.05:
                raise ValueError("reference word exceeds channel duration")
            ordered.append((word.start_s, speaker, len(ordered), word.text))
        timeline.extend(
            DiarizationSegment(speaker, start, end)
            for start, end in merge_speech_intervals([(w.start_s, w.end_s) for w in words])
        )
        counts.extend(DiarizationSegment(speaker, s.start_s, s.end_s) for s in segments)
    return BenchmarkReference(
        language=clip.language.split("-")[0],
        text_by_speaker=texts,
        mixed_text=" ".join(item[3] for item in sorted(ordered)),
        speaker_count_timeline=tuple(counts),
        forced_timeline=tuple(timeline),
        evaluation_duration_s=max(a.duration_s for a in clip.audio.values()),
        pseudo_segments=sum(s.timing_source == "pseudo" for ss in clip.word_alignments.values() for s in ss),
    )


def _blob(
    transcript: Transcript, *, diarizes: bool, word_activity: bool = False, check_word_coverage: bool = False
) -> dict:
    if not diarizes:
        return {"verbatim": {"segments": [{"text": transcript.text}]}}
    segments = list(transcript.segments)
    if transcript.text.strip() and not segments:
        raise ValueError("nonempty diarized transcript has no speaker segments")
    if any(s.speaker is None for s in segments):
        raise ValueError("diarized output contains a segment without a speaker label")
    incomplete_words = transcript.metadata.get("word_alignment_complete") is False
    if word_activity:
        if check_word_coverage:
            incomplete_words |= not word_text_coverage_complete(
                ((s.speaker, s.text) for s in segments),
                ((w.speaker, w.text) for s in segments for w in s.words),
            )
        else:
            incomplete_words |= any(s.text.strip() and not s.words for s in segments)
    return {
        "activity_source": "speaker_words" if word_activity else "native_segments",
        **({"provider_meta": {"1": {"word_alignment_complete": False}}} if incomplete_words else {}),
        "verbatim": {
            "segments": [
                {
                    "text": s.text,
                    "speaker_external_id": s.speaker,
                    "start_ms": None if s.start_s is None else s.start_s * 1000,
                    "end_ms": None if s.end_s is None else s.end_s * 1000,
                }
                for s in segments
            ]
        },
        "words": [
            {
                "text": w.text,
                "speaker_external_id": w.speaker,
                "start_ms": None if w.start_s is None else w.start_s * 1000,
                "end_ms": None if w.end_s is None else w.end_s * 1000,
            }
            for s in segments
            for w in s.words
        ],
    }


def score_connector_output(clip: BenchmarkClip, transcript: Transcript, channel: str, model) -> dict:
    """Called immediately after inference; scoring failure never retries inference."""
    reference = reference_from_clip(clip)
    if channel != "mono":
        metrics = {}
        if model.transcribes:
            language = reference.language
            units = ["cer"] + (["ser"] if language == "vi" else ["wer"] if language not in {"zh", "ja", "th"} else [])
            for unit in units:
                counts = score_pair(
                    reference.text_by_speaker[channel[2:]], transcript.text, language=language, metric=unit
                ).counts
                metrics[unit] = {
                    **asdict(counts),
                    "numerator": counts.numerator,
                    "denominator": counts.denominator,
                    "value": counts.error_rate,
                }
        return {"language": reference.language, "condition": channel, "metrics": metrics, "unavailable": {}}
    word_activity = model.backend in {"elevenlabs", "mai-transcribe", "gemini-transcribe", "xai-grok"}
    blob = _blob(
        transcript,
        diarizes=model.diarizes,
        word_activity=word_activity,
        check_word_coverage=model.backend == "mai-transcribe",
    )
    result = score_sample(
        reference,
        blob,
        condition="mono",
        transcribes=model.transcribes,
        diarizes=model.diarizes,
        provider="gemini_transcribe" if model.backend == "gemini-transcribe" else None,
    )
    return result


def write_run_scores(records: list[dict], destination: Path) -> None:
    """Consolidate this run's scored responses, including pairs split across workers."""
    output, pairs = [], {}
    for record in records:
        result = record.get("scores")
        if result is None:
            continue
        row = {
            "sample_id": record["sample_id"],
            "model": record["model"],
            "run_id": record.get("scoring_run_id", record["dataset_revision"]),
            "run_config_sha256": record.get("run_config_sha256"),
            "dataset_source_id": record.get("dataset_source_id"),
            "result": result,
        }
        output.append(row)
        if record["channel"] in {"ch1", "ch2"}:
            key = (record["sample_id"], record["model"])
            channels = pairs.setdefault(key, {})
            if record["channel"] in channels:
                raise ValueError("duplicate physical channel in scored run")
            channels[record["channel"]] = row
    for channels in pairs.values():
        if set(channels) != {"ch1", "ch2"}:
            continue
        left, right = (channels[ch] for ch in ("ch1", "ch2"))
        metrics = {}
        for name in left["result"]["metrics"].keys() & right["result"]["metrics"].keys():
            a, b = left["result"]["metrics"][name], right["result"]["metrics"][name]
            combined = {k: a[k] + b[k] for k in a if k != "value"}
            combined["value"] = combined["numerator"] / combined["denominator"] if combined["denominator"] else None
            metrics[name] = combined
        unavailable = {ch: row["result"]["unavailable"] for ch, row in channels.items() if row["result"]["unavailable"]}
        output.append(
            {
                **left,
                "result": {
                    "language": left["result"]["language"],
                    "condition": "dual",
                    "metrics": metrics,
                    "unavailable": unavailable,
                },
            }
        )
    path = destination / "scores.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in output))
    temporary.replace(path)
