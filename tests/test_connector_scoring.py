"""Fresh connector responses reach scoring without any external predictions."""

import json
from dataclasses import replace

import pytest
from test_inference import FakeBackend, _dataset

from dai_asr_i18n.datasets import load_selection
from dai_asr_i18n.evaluation.connector import reference_from_clip, score_connector_output, write_run_scores
from dai_asr_i18n.inference import Transcript, TranscriptSegment, TranscriptWord, load_run_config
from dai_asr_i18n.inference.runner import run_inference


def test_run_scores_fresh_responses_and_resumes_without_calls(tmp_path):
    dataset = _dataset(tmp_path / "data")
    backend = FakeBackend()
    config = replace(load_run_config("whisper-tiny-cpu-smoke"), channels=("ch1", "ch2"))
    output = tmp_path / "run"
    run_inference(dataset, config, output, backend=backend)
    rows = [json.loads(line) for line in (output / "scores.jsonl").read_text().splitlines()]
    dual = next(r["result"] for r in rows if r["result"]["condition"] == "dual")
    assert dual["metrics"]["wer"]["value"] == 0.5
    assert dual["metrics"]["wer"]["denominator"] == 2
    before = (output / "scores.jsonl").read_bytes()
    calls = len(backend.calls)
    run_inference(dataset, config, output, backend=backend)
    assert len(backend.calls) == calls
    assert before == (output / "scores.jsonl").read_bytes()


def test_native_forced_regions_and_hybrid_scores(tmp_path):
    clip = load_selection(_dataset(tmp_path / "data")).clips[0]
    reference = reference_from_clip(clip)
    assert [(s.start_s, s.end_s) for s in reference.forced_timeline] == [(0.001, 0.004), (0.006, 0.009)]
    model = load_run_config("openai-gpt-4o-transcribe-diarize").model
    transcript = Transcript(
        text="hello world",
        segments=(
            TranscriptSegment("hello", 0.001, 0.004, speaker="A"),
            TranscriptSegment("world", 0.006, 0.009, speaker="B"),
        ),
    )
    result = score_connector_output(clip, transcript, "mono", model)
    assert result["metrics"]["cpwer"]["value"] == 0
    assert result["metrics"]["der_forced"]["value"] == 0
    assert result["unavailable"] == {}
    empty = score_connector_output(clip, Transcript(text=""), "mono", model)
    assert empty["metrics"]["cpwer"]["value"] == 1
    assert empty["metrics"]["der_forced"]["value"] == 1
    diar = score_connector_output(clip, transcript, "mono", replace(model, transcribes=False))
    assert "cpwer" not in diar["metrics"]
    assert diar["metrics"]["jer_forced"]["value"] == 0


def test_pairs_can_be_scored_after_worker_consolidation(tmp_path):
    dataset = _dataset(tmp_path / "data")
    out = tmp_path / "run"
    run_inference(
        dataset, replace(load_run_config("whisper-tiny-cpu-smoke"), channels=("ch1", "ch2")), out, backend=FakeBackend()
    )
    records = [json.loads(s) for s in (out / "hypotheses.jsonl").read_text().splitlines()]
    write_run_scores(records[:1], out)
    assert '"condition": "dual"' not in (out / "scores.jsonl").read_text()
    write_run_scores(records, out)
    assert '"condition": "dual"' in (out / "scores.jsonl").read_text()
    with pytest.raises(ValueError, match="duplicate"):
        write_run_scores(records + records[:1], out)


def test_scoring_failure_does_not_retry_paid_request(tmp_path, monkeypatch):
    import dai_asr_i18n.evaluation.connector as connector

    def fail(*args):
        raise ValueError("bad reference")

    monkeypatch.setattr(connector, "score_connector_output", fail)
    backend = FakeBackend()
    report = run_inference(
        _dataset(tmp_path / "data"),
        replace(load_run_config("whisper-tiny-cpu-smoke"), channels=("ch1", "ch2")),
        tmp_path / "run",
        backend=backend,
    )
    assert report.failed == 0
    assert report.completed == 2
    rows = [json.loads(s) for s in (tmp_path / "run/scores.jsonl").read_text().splitlines()]
    assert all(not r["result"]["metrics"] for r in rows)
    assert all(r["result"]["unavailable"] for r in rows)


def test_gemini_reversed_words_are_missing_activity_not_missing_text(tmp_path):
    clip = load_selection(_dataset(tmp_path / "data")).clips[0]
    words = (TranscriptWord("hello", 0.004, 0.001, speaker="A"),)
    transcript = Transcript(
        text="hello world",
        segments=(
            TranscriptSegment("hello", 0.004, 0.001, words=words, speaker="A"),
            TranscriptSegment(
                "world", 0.006, 0.009, words=(TranscriptWord("world", 0.006, 0.009, speaker="B"),), speaker="B"
            ),
        ),
    )
    model = load_run_config("gemini-transcribe-3.5").model
    result = score_connector_output(clip, transcript, "mono", model)
    assert result["metrics"]["cpwer"]["value"] == 0
    assert result["metrics"]["der_forced_miss"]["value"] == pytest.approx(0.5)
    assert result["unavailable"] == {}
