"""Validation and deterministic consolidation for distributed inference shards."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dai_asr_i18n.inference.runner import (
    EFFECTIVE_CONFIG_FILENAME,
    FAILURES_FILENAME,
    HYPOTHESES_FILENAME,
    RUN_MANIFEST_FILENAME,
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _load_json(value: str, path: Path) -> Any:
    try:
        return json.loads(value, parse_constant=_reject_constant, object_pairs_hook=_object_without_duplicates)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"could not read strict JSON from {path}: {error}") from error


def _object(path: Path) -> dict[str, Any]:
    try:
        value = _load_json(path.read_text(encoding="utf-8"), path)
    except OSError as error:
        raise ValueError(f"could not read valid JSON from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = _load_json(line, path)
            except ValueError as error:
                raise ValueError(f"invalid JSON in {path} line {line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} must contain an object")
            result.append(value)
    return result


def merge_run_shards(shards: list[str | Path], output: str | Path, *, replace: bool = False) -> Path:
    """Validate a complete shard set and write one deterministic combined run."""

    roots = tuple(Path(path).expanduser().resolve() for path in shards)
    if not roots:
        raise ValueError("at least one shard directory is required")
    requested_destination = Path(output).expanduser()
    if requested_destination.is_symlink():
        raise ValueError(f"combined output must not be a symbolic link: {requested_destination}")
    destination = requested_destination.resolve()
    if any(root == destination or root.is_relative_to(destination) for root in roots):
        raise ValueError("combined output must not contain a source shard directory")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"combined output must be a directory, not a file: {destination}")
    if destination.exists() and any(destination.iterdir()) and not replace:
        raise ValueError(f"combined output must be a new or empty directory: {destination}")

    specs = [_object(root / EFFECTIVE_CONFIG_FILENAME) for root in roots]
    manifests = [_object(root / RUN_MANIFEST_FILENAME) for root in roots]
    first = specs[0]
    selection = first.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("shard run spec is missing its selection")
    shard_count = selection.get("shard_count")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise ValueError("shard_count must be a positive integer")
    if len(roots) != shard_count:
        raise ValueError(f"expected {shard_count} shard directories, got {len(roots)}")

    invariant_keys = {
        "dataset_source_type",
        "dataset_source_id",
        "dataset_revision",
        "selection_sha256",
        "config",
        "config_sha256",
        "backend",
        "implementation",
        "inference_protocol",
    }
    expected_invariants = {key: first.get(key) for key in invariant_keys}
    indices: set[int] = set()
    for spec in specs:
        if {key: spec.get(key) for key in invariant_keys} != expected_invariants:
            raise ValueError("shards do not share the same dataset, backend, and run configuration")
        shard_selection = spec.get("selection")
        if not isinstance(shard_selection, dict) or shard_selection.get("shard_count") != shard_count:
            raise ValueError("shards disagree on shard_count")
        if shard_selection.get("languages") != selection.get("languages") or shard_selection.get(
            "sample_ids"
        ) != selection.get("sample_ids"):
            raise ValueError("shards disagree on language or sample filters")
        index = shard_selection.get("shard_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in indices:
            raise ValueError("shard indices must be unique integers")
        indices.add(index)
    if indices != set(range(shard_count)):
        raise ValueError("shard indices do not cover the complete zero-based range")

    hypothesis_groups = [_rows(root / HYPOTHESES_FILENAME) for root in roots]
    failure_groups = [_rows(root / FAILURES_FILENAME) for root in roots]
    for root, spec, manifest, shard_hypotheses, shard_failures in zip(
        roots, specs, manifests, hypothesis_groups, failure_groups, strict=True
    ):
        manifest_identity = {
            "config_sha256": manifest.get("config_sha256"),
            "selection": manifest.get("selection"),
            "backend": manifest.get("backend"),
            "implementation": manifest.get("implementation"),
            "inference_protocol": manifest.get("inference_protocol"),
        }
        spec_identity = {key: spec.get(key) for key in manifest_identity}
        if manifest_identity != spec_identity:
            raise ValueError(f"{root / RUN_MANIFEST_FILENAME} does not match its effective run spec")
        coverage = manifest.get("coverage")
        if not isinstance(coverage, dict):
            raise ValueError(f"{root / RUN_MANIFEST_FILENAME} is missing coverage")
        values = {name: coverage.get(name) for name in ("total", "successful_total", "remaining")}
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
            raise ValueError(f"{root / RUN_MANIFEST_FILENAME} has invalid coverage counts")
        total = values["total"]
        successful = values["successful_total"]
        remaining = values["remaining"]
        if successful != len(shard_hypotheses):
            raise ValueError(
                f"{root / RUN_MANIFEST_FILENAME} reports {successful} successful tasks "
                f"but {len(shard_hypotheses)} hypothesis rows exist"
            )
        if total != successful + remaining:
            raise ValueError(f"{root / RUN_MANIFEST_FILENAME} has inconsistent coverage counts")
        failure_attempts = coverage.get("failure_attempts_total")
        if failure_attempts is not None and (
            isinstance(failure_attempts, bool)
            or not isinstance(failure_attempts, int)
            or failure_attempts != len(shard_failures)
        ):
            raise ValueError(f"{root / RUN_MANIFEST_FILENAME} disagrees with its failure rows")
        status = manifest.get("status")
        if status == "complete" and remaining != 0:
            raise ValueError(f"{root / RUN_MANIFEST_FILENAME} claims completion with unfinished tasks")
        if status not in {"complete", "incomplete_with_failures", "incomplete_systematic_failure"}:
            raise ValueError(f"{root / RUN_MANIFEST_FILENAME} has invalid status {status!r}")

    hypotheses = [row for group in hypothesis_groups for row in group]
    failures = [row for group in failure_groups for row in group]
    task_ids = [row.get("task_id") for row in hypotheses]
    if any(not isinstance(task_id, str) for task_id in task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError("hypothesis task IDs must be present and unique across shards")
    hypotheses.sort(key=lambda row: (str(row.get("sample_id")), str(row.get("channel"))))
    failures.sort(key=lambda row: (str(row.get("sample_id")), str(row.get("channel")), str(row.get("recorded_at"))))

    if replace and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    combined_spec = {
        **first,
        "selection": {
            "languages": selection.get("languages", []),
            "sample_ids": selection.get("sample_ids", []),
            "shard_count": shard_count,
            "shard_indices": list(range(shard_count)),
        },
    }
    (destination / EFFECTIVE_CONFIG_FILENAME).write_text(
        json.dumps(combined_spec, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    for filename, rows in ((HYPOTHESES_FILENAME, hypotheses), (FAILURES_FILENAME, failures)):
        if rows:
            (destination / filename).write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
                ),
                encoding="utf-8",
            )
    from dai_asr_i18n.evaluation.connector import write_run_scores

    write_run_scores(hypotheses, destination)
    totals = [manifest.get("coverage", {}) for manifest in manifests]
    complete = all(manifest.get("status") == "complete" for manifest in manifests)
    combined_manifest = {
        "schema_version": 2,
        "status": "complete" if complete else "incomplete_with_failures",
        "finished_at": datetime.now(UTC).isoformat(),
        "dataset": manifests[0].get("dataset"),
        "config": first.get("config"),
        "config_sha256": first.get("config_sha256"),
        "selection": combined_spec["selection"],
        "backend": first.get("backend"),
        "implementation": first.get("implementation"),
        "inference_protocol": first.get("inference_protocol"),
        "coverage": {
            "total": sum(int(value.get("total", 0)) for value in totals),
            "successful_total": len(hypotheses),
            "failure_attempts_total": len(failures),
            "remaining": sum(int(value.get("remaining", 0)) for value in totals),
        },
        "shard_manifests": manifests,
    }
    (destination / RUN_MANIFEST_FILENAME).write_text(
        json.dumps(combined_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return destination


__all__ = ["merge_run_shards"]
