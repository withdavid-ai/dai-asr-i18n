from __future__ import annotations

import json
from pathlib import Path

import pytest

from dai_asr_i18n.inference import merge_run_shards


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _shard(root: Path, index: int, count: int) -> Path:
    root.mkdir(parents=True)
    selection = {"languages": ["en"], "sample_ids": [], "shard_index": index, "shard_count": count}
    spec = {
        "schema_version": 2,
        "dataset_source_type": "huggingface",
        "dataset_source_id": "example/public-asr",
        "dataset_revision": "1" * 40,
        "selection_sha256": "2" * 64,
        "config": {"name": "test"},
        "config_sha256": "3" * 64,
        "backend": {"backend": "fake"},
        "implementation": {"package_version": "0.5.0", "source_sha256": "4" * 64},
        "inference_protocol": "dai-asr-i18n-inference-v5",
        "selection": selection,
    }
    _json(root / "effective-run-spec.json", spec)
    _json(
        root / "run-manifest.json",
        {
            "status": "complete",
            "dataset": {"source_id": "example/public-asr"},
            "config_sha256": spec["config_sha256"],
            "selection": selection,
            "backend": spec["backend"],
            "implementation": spec["implementation"],
            "inference_protocol": spec["inference_protocol"],
            "coverage": {"total": 1, "successful_total": 1, "remaining": 0, "failure_attempts_total": 0},
        },
    )
    (root / "hypotheses.jsonl").write_text(
        json.dumps({"task_id": f"task-{index}", "sample_id": f"sample-{1 - index}", "channel": "ch1"}) + "\n",
        encoding="utf-8",
    )
    return root


def test_merge_run_shards_validates_coverage_and_sorts_output(tmp_path: Path):
    shards = [_shard(tmp_path / f"shard-{index}", index, 2) for index in range(2)]

    destination = merge_run_shards(shards, tmp_path / "combined")

    rows = [json.loads(line) for line in (destination / "hypotheses.jsonl").read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == ["sample-0", "sample-1"]
    manifest = json.loads((destination / "run-manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["coverage"] == {
        "failure_attempts_total": 0,
        "remaining": 0,
        "successful_total": 2,
        "total": 2,
    }


def test_merge_run_shards_rejects_missing_or_duplicate_indices(tmp_path: Path):
    first = _shard(tmp_path / "first", 0, 2)
    duplicate = _shard(tmp_path / "duplicate", 0, 2)

    with pytest.raises(ValueError, match="unique"):
        merge_run_shards([first, duplicate], tmp_path / "combined")


def test_merge_run_shards_rejects_manifest_coverage_that_disagrees_with_rows(tmp_path: Path):
    shards = [_shard(tmp_path / f"shard-{index}", index, 2) for index in range(2)]
    (shards[1] / "hypotheses.jsonl").unlink()

    with pytest.raises(ValueError, match="reports 1 successful tasks but 0 hypothesis rows exist"):
        merge_run_shards(shards, tmp_path / "combined")

    assert not (tmp_path / "combined").exists()


def test_merge_run_shards_rejects_manifest_from_a_different_shard(tmp_path: Path):
    shards = [_shard(tmp_path / f"shard-{index}", index, 2) for index in range(2)]
    wrong_manifest = json.loads((shards[0] / "run-manifest.json").read_text(encoding="utf-8"))
    _json(shards[1] / "run-manifest.json", wrong_manifest)

    with pytest.raises(ValueError, match="does not match its effective run spec"):
        merge_run_shards(shards, tmp_path / "combined")


@pytest.mark.parametrize(
    "payload",
    [
        '{"task_id":"task-0","sample_id":"sample-1","channel":"ch1","latency_s":NaN}\n',
        '{"task_id":"task-0","task_id":"duplicate","sample_id":"sample-1","channel":"ch1"}\n',
    ],
)
def test_merge_run_shards_rejects_nonstandard_or_ambiguous_json(tmp_path: Path, payload: str):
    shards = [_shard(tmp_path / f"shard-{index}", index, 2) for index in range(2)]
    (shards[0] / "hypotheses.jsonl").write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        merge_run_shards(shards, tmp_path / "combined")


def test_merge_run_shards_can_replace_a_valid_generated_output(tmp_path: Path):
    shards = [_shard(tmp_path / f"shard-{index}", index, 2) for index in range(2)]
    destination = merge_run_shards(shards, tmp_path / "combined")
    stale = destination / "stale.txt"
    stale.write_text("old combined generation", encoding="utf-8")

    assert merge_run_shards(shards, destination, replace=True) == destination

    assert not stale.exists()
    rows = [json.loads(line) for line in (destination / "hypotheses.jsonl").read_text().splitlines()]
    assert len(rows) == 2


def test_merge_run_shards_never_replaces_a_directory_containing_source_shards(tmp_path: Path):
    shards = [_shard(tmp_path / "runs" / f"shard-{index}", index, 2) for index in range(2)]

    with pytest.raises(ValueError, match="must not contain"):
        merge_run_shards(shards, tmp_path / "runs", replace=True)

    assert all(shard.exists() for shard in shards)


def test_merge_run_shards_never_replaces_through_an_output_symlink(tmp_path: Path):
    shards = [_shard(tmp_path / f"shard-{index}", index, 2) for index in range(2)]
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    output = tmp_path / "combined"
    output.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        merge_run_shards(shards, output, replace=True)

    assert marker.read_text(encoding="utf-8") == "keep"
