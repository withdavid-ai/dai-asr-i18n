"""Evaluate benchmark references against prepared model outputs."""

from collections.abc import Mapping
from dataclasses import asdict

from dai_asr_i18n.evaluation.hypotheses import hypothesis_text_by_speaker, hypothesis_timeline, validate_hypothesis_blob
from dai_asr_i18n.evaluation.references import BenchmarkReference
from dai_asr_i18n.scoring import score_pair, score_speakers
from dai_asr_i18n.scoring.diarization import der, jer, speaker_count
from dai_asr_i18n.scoring.models import EditCounts


def score_sample(
    reference: BenchmarkReference,
    hypothesis: Mapping,
    *,
    condition: str,
    transcribes: bool,
    diarizes: bool,
    provider: str | None = None,
) -> dict:
    """Return per-clip primitives, with timing failures separate from lexical scores.

    For dual, hypothesis is {ch1: prediction, ch2: prediction}; each physical
    channel is scored independently. For mono it is one prediction. Empty
    successful output is {}, not a failed request. Missing requests must be
    recorded separately; they must not be silently converted to empty output.
    """
    if condition not in {"dual", "mono"}:
        raise ValueError("condition must be dual or mono")
    if not (transcribes or diarizes) or (condition == "dual" and not transcribes):
        raise ValueError(
            "no applicable metrics: dual requires transcription; mono requires transcription or diarization"
        )
    if condition == "dual" and set(hypothesis) != {"ch1", "ch2"}:
        raise ValueError("dual scoring requires both ch1 and ch2 artifacts")
    for blob in hypothesis.values() if condition == "dual" else (hypothesis,):
        validate_hypothesis_blob(blob)
    metrics, unavailable = {}, {}
    lang = reference.language
    units = ["cer"] + (["ser"] if lang == "vi" else ["wer"] if lang not in {"zh", "ja", "th"} else [])

    def add_counts(key, counts):
        metrics[key] = {
            **asdict(counts),
            "numerator": counts.numerator,
            "denominator": counts.denominator,
            "value": counts.error_rate,
        }

    if transcribes:
        if condition == "dual":
            texts = {
                ch[2:]: " ".join(s.get("text") or "" for s in (blob.get("verbatim") or {}).get("segments", []))
                for ch, blob in hypothesis.items()
            }
            for unit in units:
                counts = EditCounts()
                for speaker in ("1", "2"):
                    counts += score_pair(
                        reference.text_by_speaker[speaker], texts[speaker], language=lang, metric=unit
                    ).counts
                add_counts(unit, counts)
        else:
            texts = hypothesis_text_by_speaker(hypothesis)
            for unit in units:
                if diarizes:
                    counts = score_speakers(reference.text_by_speaker, texts, language=lang, metric=unit).counts
                else:
                    counts = score_pair(
                        reference.mixed_text,
                        " ".join(s.get("text") or "" for s in (hypothesis.get("verbatim") or {}).get("segments", [])),
                        language=lang,
                        metric=unit,
                    ).counts
                add_counts(unit, counts)
        if condition == "dual" or diarizes:
            primary_unit = "cer" if lang in {"zh", "ja", "ko", "th"} else "ser" if lang == "vi" else "wer"
            for unit in [primary_unit] if sum(bool(t) for t in reference.text_by_speaker.values()) >= 2 else []:
                add_counts(
                    "cp" + unit, score_speakers(reference.text_by_speaker, texts, language=lang, metric=unit).counts
                )

    if condition == "mono" and diarizes:
        try:
            timeline = hypothesis_timeline(hypothesis, provider=provider)
        except (ValueError, TypeError) as error:
            unavailable["diarization"] = str(error)
        else:
            kwargs = {"evaluation_duration_s": reference.evaluation_duration_s}
            for suffix, ref in (("_forced", reference.forced_timeline),):
                for collar, collar_suffix in ((0.0, ""), (0.25, "_c25")):
                    result = der(ref, timeline, collar_s=collar, **kwargs)
                    for component, numerator in (
                        ("", result.numerator),
                        ("_miss", result.missed_s),
                        ("_fa", result.false_alarm_s),
                        ("_conf", result.confusion_s),
                    ):
                        # Metric rows store DER speaker-time in milliseconds.
                        metrics["der" + suffix + collar_suffix + component] = {
                            "numerator": numerator * 1000,
                            "denominator": result.denominator * 1000,
                            "value": numerator / result.denominator if result.denominator else None,
                        }
                result = jer(ref, timeline, **kwargs)
                metrics["jer" + suffix] = {
                    "numerator": result.numerator,
                    "denominator": result.denominator,
                    "value": result.value,
                }
            count = speaker_count(reference.speaker_count_timeline, timeline, **kwargs)
            for key, value in (("sca", count.accuracy), ("spk_count_mae", count.absolute_error)):
                metrics[key] = {"numerator": value, "denominator": 1, "value": value}
    return {"language": lang, "condition": condition, "metrics": metrics, "unavailable": unavailable}
