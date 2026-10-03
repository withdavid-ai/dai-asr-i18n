"""Release ingestion and malformed-output safeguards, including no input mutation."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from dai_asr_i18n import der, hypothesis_text_by_speaker, hypothesis_timeline, reference_from_alignment

FIXTURES = json.loads((Path(__file__).parent / "fixtures/scoring-cases.json").read_text())


def prepare(case):
    return reference_from_alignment(case["document"], channel_durations_s=case["durations"])


def test_reference_preserves_transcript_and_word_regions_without_mutating():
    case = deepcopy(FIXTURES["references"][0])
    before = deepcopy(case)
    r = prepare(case)
    assert r.text_by_speaker == {"1": "one two three", "2": "four"}
    assert [(s.start_s, s.end_s) for s in r.forced_timeline if s.speaker == "1"] == [(0.0, 0.7), (0.901, 1.2)]
    assert r.speaker_count_timeline[0].end_s == 2.0
    assert r.pseudo_segments == 0
    assert case == before
    pseudo = prepare(FIXTURES["references"][1])
    assert pseudo.pseudo_segments == 1
    assert pseudo.forced_timeline == r.forced_timeline


@pytest.mark.parametrize("bad", ["zero", "missing", "text", "speaker", "duration", "duplicate", "window"])
def test_reference_does_not_silently_repair_bad_artifacts(bad):
    case = deepcopy(FIXTURES["references"][0])
    a = case["document"]["annotations"][0]
    if bad == "zero":
        a["words"][0]["end"] = a["words"][0]["start"]
    elif bad == "missing":
        a["words"] = []
    elif bad == "text":
        a["silverText"] = "altered"
    elif bad == "speaker":
        a["speakerId"] = 2
    elif bad == "duration":
        case["durations"]["ch1"] = 0.9
    elif bad == "duplicate":
        case["document"]["annotations"].append(deepcopy(a))
    elif bad == "window":
        a["audioEnd"] = 0.1
    with pytest.raises(ValueError):
        prepare(case)


def test_gemini_drops_reversed_word_for_diarization_only_and_keeps_text():
    b = deepcopy(FIXTURES["gemini"][1]["blob"])
    original = deepcopy(b)
    text = hypothesis_text_by_speaker(b)
    good = hypothesis_timeline(b, provider="gemini_transcribe")
    assert [(s.start_s, s.end_s) for s in good] == [(0.0, 1.0), (4.0, 5.0), (6.0, 7.0)]
    assert "1" in text["1:A"].split()  # Bad word stays in the lexical transcript.
    assert b == original
    with pytest.raises(ValueError):
        hypothesis_timeline(b)
    with pytest.raises(ValueError):
        hypothesis_timeline(b, provider="other_model")


def test_all_reversed_words_score_as_standard_missing_speech():
    b = FIXTURES["gemini"][2]["blob"]
    result = der([("ref", 0.0, 1.0)], hypothesis_timeline(b, provider="gemini_transcribe"))
    assert result.value == 1 and result.missed_s == 1
    assert result.confusion_s == result.false_alarm_s == 0


def test_direct_gemini_and_fresh_word_activity_use_same_bounded_timeline():
    words = [
        {"text": text, "type": "word_info", "speaker": "A", "start_offset": start, "end_offset": end}
        for text, start, end in [("hello", 0.0, 1.0), ("again", 5.0, 6.0)]
    ]
    blob = {
        "verbatim": {
            "segments": [{"speaker_external_id": "1:A", "start_ms": 0.0, "end_ms": 6000.0, "text": "hello again"}]
        },
        "words": [{"start_ms": 0.0, "end_ms": 1000.0}, {"start_ms": 5000.0, "end_ms": 6000.0}],
        "provider_meta": {
            "1": {
                "response": {
                    "status": "completed",
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [{"type": "text", "text": "hello again", "annotations": words}],
                        }
                    ],
                }
            }
        },
    }
    before = deepcopy(blob)
    direct = hypothesis_timeline(blob, provider="gemini_transcribe")
    fresh = deepcopy(blob)
    fresh["activity_source"] = "speaker_words"
    for word in fresh["words"]:
        word["speaker_external_id"] = "1:A"
    assert direct == hypothesis_timeline(fresh, provider="gemini_transcribe")
    assert [(s.start_s, s.end_s) for s in direct] == [(0.0, 1.0), (5.0, 6.0)]
    assert blob == before
    assert hypothesis_timeline({}, provider="gemini_transcribe") == ()


@pytest.mark.parametrize("bad", ["raw", "stored", "missing_metadata", "incomplete", "segment", "dual"])
def test_gemini_requires_verifiable_original_response(bad):
    b = deepcopy(FIXTURES["gemini"][1]["blob"])
    if bad == "raw":
        b["provider_meta"]["1"]["response"]["status"] = "failed"
    elif bad == "stored":
        b["words"][0]["start_ms"] = 42
    elif bad == "missing_metadata":
        b["provider_meta"] = {}
    elif bad == "incomplete":
        b["provider_meta"]["1"]["word_alignment_complete"] = False
    elif bad == "segment":
        b["verbatim"]["segments"][0]["end_ms"] = 123
    elif bad == "dual":
        b["physical_channel_blobs"] = [deepcopy(b)]
    with pytest.raises(ValueError):
        hypothesis_timeline(b, provider="gemini_transcribe")


def test_alignment_only_preserves_text_regions_and_fallback():
    for case in FIXTURES["references"]:
        before = deepcopy(case["document"])
        aligned = reference_from_alignment(case["document"], channel_durations_s=case["durations"])
        pair = prepare(case)
        assert aligned.text_by_speaker == pair.text_by_speaker
        assert aligned.mixed_text == pair.mixed_text
        assert aligned.forced_timeline == pair.forced_timeline
        assert aligned.pseudo_segments == pair.pseudo_segments
        assert case["document"] == before


def test_alignment_only_preserves_sparse_annotation_ids():
    case = deepcopy(FIXTURES["references"][0])
    case["document"]["annotations"][0]["annotationId"] = case["document"]["jobId"] + ":ch1:12"
    result = reference_from_alignment(case["document"], channel_durations_s=case["durations"])
    assert result.text_by_speaker["1"] == "one two three"


@pytest.mark.parametrize("bad", ["zero", "missing", "text", "speaker", "duplicate", "window"])
def test_alignment_only_rejects_corrupt_words(bad):
    case = deepcopy(FIXTURES["references"][0])
    a = case["document"]["annotations"][0]
    if bad == "zero":
        a["words"][0]["end"] = a["words"][0]["start"]
    elif bad == "missing":
        a["words"] = []
    elif bad == "text":
        a["silverText"] = "altered"
    elif bad == "speaker":
        a["speakerId"] = 2
    elif bad == "duplicate":
        case["document"]["annotations"].append(deepcopy(a))
    elif bad == "window":
        a["audioEnd"] = 0.01
    with pytest.raises(ValueError):
        reference_from_alignment(case["document"], channel_durations_s=case["durations"])


@pytest.mark.parametrize("gap,merged", [(0.199, True), (0.2, True), (0.201, False)])
def test_native_word_activity_threshold(gap, merged):
    blob = {
        "activity_source": "speaker_words",
        "verbatim": {"segments": []},
        "words": [
            {"speaker_external_id": "A", "start_ms": (1 + gap) * 1000, "end_ms": 2000},
            {"speaker_external_id": "A", "start_ms": 0, "end_ms": 1000},
        ],
    }
    actual = hypothesis_timeline(blob)
    assert len(actual) == (1 if merged else 2)


@pytest.mark.parametrize("start,end", [(None, 1), (-1, 1), (float("nan"), 1), (float("inf"), 1), (2, 1)])
def test_native_word_activity_invalid_timing_is_explicit(start, end):
    blob = {
        "activity_source": "speaker_words",
        "verbatim": {"segments": []},
        "words": [
            {"speaker_external_id": "A", "start_ms": start, "end_ms": end},
        ],
    }
    with pytest.raises(ValueError):
        hypothesis_timeline(blob)
    if start == 2:
        assert hypothesis_timeline(blob, provider="gemini_transcribe") == ()
    else:
        with pytest.raises(ValueError):
            hypothesis_timeline(blob, provider="gemini_transcribe")
