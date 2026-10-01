"""Consumer workflow tests: checksums, physical channels, empty output and bad timing."""

import hashlib
import json
from pathlib import Path

import pytest

from dai_asr_i18n import load_reference, reference_from_alignment, score_sample

CASE = json.loads((Path(__file__).parent / "fixtures/scoring-cases.json").read_text())["references"][0]


def reference():
    return reference_from_alignment(CASE["document"], channel_durations_s=CASE["durations"])


def blob(text, speaker="A", start=0, end=1000):
    return {
        "verbatim": {"segments": [{"text": text, "speaker_external_id": speaker, "start_ms": start, "end_ms": end}]}
    }


def test_dual_does_not_permute_physical_channels():
    result = score_sample(
        reference(),
        {"ch1": blob("four"), "ch2": blob("one two three")},
        condition="dual",
        transcribes=True,
        diarizes=True,
    )
    assert result["metrics"]["wer"]["numerator"] > 0
    assert result["metrics"]["cpwer"]["numerator"] == 0
    assert not any(k.startswith("der") for k in result["metrics"])


def test_successful_empty_hypothesis_is_all_miss():
    result = score_sample(reference(), {}, condition="mono", transcribes=True, diarizes=True)
    assert result["metrics"]["cpwer"]["value"] == 1
    assert result["metrics"]["der_forced"]["value"] == 1
    assert result["metrics"]["jer_forced"]["value"] == 1
    assert result["metrics"]["spk_count_mae"]["value"] == 2


def test_invalid_timestamps_do_not_remove_lexical_metrics():
    result = score_sample(
        reference(), blob("one two", start=2000, end=1000), condition="mono", transcribes=True, diarizes=True
    )
    assert "cpwer" in result["metrics"]
    assert "der_forced" not in result["metrics"]
    assert "diarization" in result["unavailable"]


def test_missing_channel_is_not_an_empty_success():
    with pytest.raises(ValueError, match="both ch1 and ch2"):
        score_sample(reference(), {"ch1": {}}, condition="dual", transcribes=True, diarizes=False)


def release_files(tmp_path):
    metadata = {"audio": {ch: {"duration_s": duration} for ch, duration in CASE["durations"].items()}}
    metadata.update(item_key=CASE["document"]["jobId"], metadata={"language": CASE["document"]["lang"]})
    for name, key, value in (("scoring_reference", "reference", CASE["document"]),):
        body = json.dumps(value).encode()
        (tmp_path / (name + ".json")).write_bytes(body)
        metadata[key] = {"sha256": hashlib.sha256(body).hexdigest()}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))


def test_loader_rejects_changed_reference(tmp_path):
    release_files(tmp_path)
    (tmp_path / "scoring_reference.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_reference(tmp_path)


def test_pooling_preserves_missing_models_and_weights():
    from dai_asr_i18n.evaluation.aggregate import pool_comparison

    def row(sample, model, errors, units):
        return {
            "sample_id": sample,
            "model": model,
            "run_id": "v1",
            "result": {
                "language": "en",
                "condition": "dual",
                "metrics": {"wer": {"numerator": errors, "denominator": units}},
            },
        }

    rows = [row("1", "a", 1, 1), row("2", "a", 0, 9), row("2", "b", 2, 9)]
    kwargs = dict(models=["a", "b"], language="en", condition="dual", metric="wer")
    matched = pool_comparison(rows, **kwargs)
    assert [r["n_clips"] for r in matched] == [1, 1]
    assert [r["value"] for r in matched] == [0, 2 / 9]
    assert pool_comparison(rows, cohort="available", **kwargs)[0]["value"] == 0.1
    assert all(r["n_clips"] == 0 for r in pool_comparison(rows, **(kwargs | {"models": ["a", "missing"]})))
    with pytest.raises(ValueError, match="duplicate"):
        pool_comparison(rows + [rows[0]], **kwargs)
    with pytest.raises(ValueError, match="different runs"):
        pool_comparison([rows[0], rows[1] | {"run_id": "v2"}], **kwargs)


def test_unknown_schema_cannot_silently_score_as_empty():
    with pytest.raises(ValueError, match="verbatim.segments"):
        score_sample(reference(), {"transcript": {"text": "hello"}}, condition="mono", transcribes=True, diarizes=False)


def test_local_inference_text_is_preserved_without_claiming_diarization():
    from dai_asr_i18n.evaluation.hypotheses import hypothesis_from_local_record

    local = hypothesis_from_local_record({"channel": "ch1", "transcript": {"text": "one two three"}})
    result = score_sample(
        reference(), {"ch1": local, "ch2": blob("four")}, condition="dual", transcribes=True, diarizes=False
    )
    assert result["metrics"]["wer"]["value"] == 0
    result = score_sample(reference(), local, condition="mono", transcribes=True, diarizes=True)
    assert result["unavailable"]["diarization"]


def test_loader_rejects_metadata_from_another_clip(tmp_path):
    release_files(tmp_path)
    path = tmp_path / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["item_key"] = "different"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="identity"):
        load_reference(tmp_path)
