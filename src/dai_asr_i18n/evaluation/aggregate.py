"""Pool one explicit model/language/condition/metric comparison."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence


def pool_comparison(
    scores: Sequence[Mapping],
    *,
    models: Sequence[str],
    language: str,
    condition: str,
    metric: str,
    cohort: str = "matched",
) -> list[dict]:
    """Pool primitives, never clip percentages; require an explicit model roster.

    Each input has sample_id, model, run_id and result (score_sample output).
    Missing/unavailable metrics do not become zero errors. In matched mode a model
    with no available clips makes the intersection empty, rather than disappearing.
    Use globally unique sample IDs when combining splits. Only one run is accepted.
    """
    if cohort not in {"matched", "available"}:
        raise ValueError("cohort must be matched or available")
    if not models or len(set(models)) != len(models):
        raise ValueError("models must be nonempty and unique")
    selected = {model: {} for model in models}
    seen, runs = set(), set()
    for row in scores:
        result = row["result"]
        model = row["model"]
        if model not in selected or result["language"] != language or result["condition"] != condition:
            continue
        runs.add(row["run_id"])
        identity = (row["sample_id"], model)
        if identity in seen:
            raise ValueError("duplicate sample/model in comparison")
        seen.add(identity)
        if metric in result["metrics"]:
            selected[model][row["sample_id"]] = result["metrics"][metric]
    if len(runs) > 1:
        raise ValueError("cannot pool different runs")
    shared = set.intersection(*(set(rows) for rows in selected.values()))
    out = []
    for model, rows in selected.items():
        keys = sorted(shared if cohort == "matched" else rows)
        numerator = math.fsum(rows[k]["numerator"] for k in keys)
        denominator = math.fsum(rows[k]["denominator"] for k in keys)
        out.append(
            {
                "model": model,
                "language": language,
                "condition": condition,
                "metric": metric,
                "cohort": cohort,
                "run_id": next(iter(runs), None),
                "n_clips": len(keys),
                "available_clips": len(rows),
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator if denominator else None,
                "clip_ids_sha256": hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest(),
            }
        )
    return out
