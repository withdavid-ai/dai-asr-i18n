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


def test_schema_v2_reference_words_and_timestamps_are_unchanged(tmp_path):
    from dai_asr_i18n.datasets.models import AlignedWord, WordAlignmentSegment

    clip = load_selection(_dataset(tmp_path / "data")).clips[0]
    aligned = WordAlignmentSegment(
        0,
        0.005,
        "forced",
        tuple(AlignedWord(text, 0.001, 0.004) for text in ("I", "can", "t", "go", "100")),
    )
    clip = replace(clip, word_alignments={**clip.word_alignments, "ch1": (aligned,)})
    before = clip.to_dict()
    assert before["schema_version"] == 2
    assert "text" not in before["word_alignments"]["ch1"][0]
    assert clip.reference_text("ch1") == "I can t go 100"
    assert reference_from_clip(clip).text_by_speaker["1"] == "I can t go 100"
    model = load_run_config("whisper-large-v3-portable").model
    result = score_connector_output(clip, Transcript(text="I can t go 100"), "ch1", model)
    assert result["metrics"]["wer"]["value"] == 0
    assert clip.to_dict() == before


@pytest.mark.parametrize(
    "profile", ["elevenlabs-scribe-v2", "mai-transcribe-2", "gemini-transcribe-3.5", "xai-grok-voice-transcribe-2"]
)
def test_word_activity_is_separate_from_lexical_turns(tmp_path, profile):
    from dai_asr_i18n.evaluation.connector import _blob
    from dai_asr_i18n.evaluation.hypotheses import hypothesis_timeline

    words = (
        TranscriptWord("one", 0, 1, speaker="A"),
        TranscriptWord("two", 5, 6, speaker="A"),
        TranscriptWord("overlap", 0.5, 0.9, speaker="B"),
        TranscriptWord("three", 6.2, 7, speaker="A"),
        TranscriptWord("zero", 7.2, 7.2, speaker="A"),
        TranscriptWord("four", 7.4, 8, speaker="A"),
    )
    transcript = Transcript(
        text="one two overlap three zero four",
        segments=(
            TranscriptSegment(
                "one two three zero four", 0, 8, words=tuple(w for w in words if w.speaker == "A"), speaker="A"
            ),
            TranscriptSegment("overlap", 0.5, 0.9, words=(words[2],), speaker="B"),
        ),
    )
    blob = _blob(transcript, diarizes=True, word_activity=True)
    assert [(s.speaker, s.start_s, s.end_s) for s in hypothesis_timeline(blob)] == [
        ("A", 0, 1),
        ("B", 0.5, 0.9),
        ("A", 5, 7),
        ("A", 7.4, 8),
    ]
    assert blob["verbatim"]["segments"][0]["text"] == "one two three zero four"
    assert [(s.start_s, s.end_s) for s in hypothesis_timeline(_blob(transcript, diarizes=True))] == [(0, 8), (0.5, 0.9)]
    model = load_run_config(profile).model
    clip = load_selection(_dataset(tmp_path / "data")).clips[0]
    # Both predictions are inside this synthetic clip but their enclosing turn contains a gap.
    tiny = Transcript(
        text="hello",
        segments=(
            TranscriptSegment(
                "hello", 0, 0.01, speaker="A", words=(TranscriptWord("hello", 0.001, 0.004, speaker="A"),)
            ),
        ),
    )
    result = score_connector_output(clip, tiny, "mono", model)
    assert result["metrics"]["der_forced_fa"]["numerator"] == 0


def test_existing_manifest_runs_without_segment_text_or_schema_migration(tmp_path):
    dataset = _dataset(tmp_path / "data")
    path = dataset / "selection.jsonl"
    row = json.loads(path.read_text())
    assert row["schema_version"] == 2
    assert all("text" not in segment for segments in row["word_alignments"].values() for segment in segments)
    before = path.read_bytes()
    backend = FakeBackend()
    report = run_inference(
        dataset,
        replace(load_run_config("whisper-large-v3-portable"), channels=("ch1", "ch2")),
        tmp_path / "run",
        backend=backend,
    )
    assert report.completed == 2 and report.failed == 0
    assert len(backend.calls) == 2
    assert path.read_bytes() == before


@pytest.mark.parametrize("explicit_flag", [False, True])
def test_partial_word_annotations_preserve_cpwer_and_make_der_unavailable(tmp_path, explicit_flag):
    clip = load_selection(_dataset(tmp_path / "data")).clips[0]
    transcript = Transcript(
        text="hello world",
        segments=(
            TranscriptSegment(
                "hello", 0.001, 0.004, speaker="A", words=(TranscriptWord("hello", 0.001, 0.004, speaker="A"),)
            ),
            TranscriptSegment("world", 0.006, 0.009, speaker="B"),
        ),
        metadata={"word_alignment_complete": False} if explicit_flag else {},
    )
    model = load_run_config("mai-transcribe-2").model
    result = score_connector_output(clip, transcript, "mono", model)
    assert result["metrics"]["cpwer"]["value"] == 0
    assert "der_forced" not in result["metrics"]
    assert any("incomplete" in str(reason) for reason in result["unavailable"].values())
    empty = score_connector_output(clip, Transcript(text=""), "mono", model)
    assert empty["metrics"]["der_forced"]["value"] == 1


@pytest.mark.parametrize("profile", ["elevenlabs-scribe-v2", "gemini-transcribe-3.5", "xai-grok-voice-transcribe-2"])
def test_mai_partial_text_guard_does_not_change_other_word_provider_scoring(tmp_path, profile):
    clip = load_selection(_dataset(tmp_path / "data")).clips[0]
    transcript = Transcript(
        text="hello world",
        segments=(
            TranscriptSegment(
                "hello world", 0.001, 0.009, speaker="A", words=(TranscriptWord("hello", 0.001, 0.004, speaker="A"),)
            ),
        ),
    )
    # This synthetic disagreement exercises scope only. Other adapters retain
    # their existing text/timing validation policy; MAI gets the new stream check.
    unchanged = score_connector_output(clip, transcript, "mono", load_run_config(profile).model)
    guarded = score_connector_output(clip, transcript, "mono", load_run_config("mai-transcribe-2").model)
    assert "der_forced" in unchanged["metrics"]
    assert "der_forced" not in guarded["metrics"]
    assert "cpwer" in guarded["metrics"]


def test_fresh_runner_scores_existing_word_only_reference_after_manifest_roundtrip(tmp_path):
    dataset = _dataset(tmp_path / "data")
    path = dataset / "selection.jsonl"
    row = json.loads(path.read_text())
    row["word_alignments"]["ch1"][0]["words"] = [
        {"text": text, "start_s": 0.001, "end_s": 0.004} for text in ("I", "can", "t", "go", "100")
    ]
    path.write_text(json.dumps(row) + "\n")

    class ExactBackend(FakeBackend):
        def transcribe(self, audio_path, *, language):
            self.calls.append((audio_path.name, language))
            return Transcript(text="I can t go 100")

    backend = ExactBackend()
    config = replace(load_run_config("whisper-large-v3-portable"), channels=("ch1",))
    run_inference(dataset, config, tmp_path / "run", backend=backend)
    scored = json.loads((tmp_path / "run/scores.jsonl").read_text())
    assert scored["result"]["metrics"]["wer"]["value"] == 0
    assert len(backend.calls) == 1
