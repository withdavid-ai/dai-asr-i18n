"""Command-line entrypoint for offline normalization and scoring."""

from __future__ import annotations

import argparse
import hmac
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dai_asr_i18n import __version__
from dai_asr_i18n.audio import build_mono_mix, mono_mix_manifest
from dai_asr_i18n.model_configs import list_model_configs, model_catalog, model_config
from dai_asr_i18n.normalization import normalization_manifest, normalize_with_metadata
from dai_asr_i18n.scoring import aggregate_record_scores, score_pair, score_records

_PROFILES = ("dai_asr_i18n", "whisper_baseline")


def _write_json(value: object, destination: str | None = None) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination and destination != "-":
        Path(destination).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite number {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate object key {key!r}")
        value[key] = item
    return value


def _normalize_command(args: argparse.Namespace) -> int:
    text = args.text if args.text is not None else sys.stdin.read()
    result = normalize_with_metadata(text, args.language, args.profile)
    if args.json:
        _write_json(result.to_dict())
    else:
        print(result.text)
    return 0


def _score_command(args: argparse.Namespace) -> int:
    if args.input is not None:
        return _score_batch_command(args)

    result = score_pair(
        args.reference,
        args.hypothesis,
        language=args.language,
        profile=args.profile,
        metric=None if args.metric == "auto" else args.metric,
    )
    payload = result.to_dict() | {"provenance": normalization_manifest(include_opencc_hashes=False)}
    _write_json(payload, args.output)
    return 0


def _load_jsonl(source: str) -> list[dict[str, Any]]:
    stream = sys.stdin if source == "-" else Path(source).open(encoding="utf-8")
    try:
        records = []
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_object_without_duplicate_keys,
                )
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from error
            except ValueError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(value)
        return records
    finally:
        if stream is not sys.stdin:
            stream.close()


def _score_batch_command(args: argparse.Namespace) -> int:
    records = _load_jsonl(args.input)
    if not records:
        raise ValueError("input contains no records")
    scored = score_records(
        records,
        profile=args.profile,
        metric=None if args.metric == "auto" else args.metric,
    )
    rows = []
    for record, result in scored:
        metadata = {key: value for key, value in record.items() if key not in {"reference", "hypothesis"}}
        rows.append({"input": metadata, **result.to_dict()})
    group_by = tuple(field.strip() for field in (args.group_by or "").split(",") if field.strip())
    payload = {
        "provenance": normalization_manifest(include_opencc_hashes=False),
        "rows": rows,
        "aggregates": [value.to_dict() for value in aggregate_record_scores(scored, group_by=group_by)],
    }
    _write_json(payload, args.output)
    return 0


def _manifest_command(args: argparse.Namespace) -> int:
    _write_json(normalization_manifest(include_opencc_hashes=not args.no_resource_hashes), args.output)
    return 0


def _mix_mono_command(args: argparse.Namespace) -> int:
    result = build_mono_mix(args.channel1, args.channel2)
    if args.expected_sha256 is not None and not hmac.compare_digest(result.sha256, args.expected_sha256.lower()):
        raise ValueError(f"mono mix checksum mismatch: expected {args.expected_sha256.lower()}, got {result.sha256}")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(result.wav_bytes)
    _write_json(result.to_dict() | {"output": str(destination), "manifest": mono_mix_manifest()})
    return 0


def _dataset_pull_command(args: argparse.Namespace) -> int:
    from dai_asr_i18n.datasets import pull_huggingface_dataset

    common = {
        "revision": args.revision,
        "manifest": args.manifest,
        "output": args.output,
        "languages": tuple(args.language or ()),
        "clip_ids": tuple(args.clip_id or ()),
        "limit": args.limit,
        "authenticated": args.authenticated,
        "workers": args.workers,
        "allow_floating_revision": args.allow_floating_revision,
        "dataset_format": args.dataset_format,
    }
    result = pull_huggingface_dataset(repo_id=args.repo, **common)
    _write_json(result.to_dict())
    return 0


def _dataset_verify_command(args: argparse.Namespace) -> int:
    from dai_asr_i18n.datasets import verify_selection

    _write_json(verify_selection(args.dataset).to_dict())
    return 0


def _run_command(args: argparse.Namespace) -> int:
    from dai_asr_i18n.inference import load_run_config, preflight_run, run_inference

    config = load_run_config(args.config)
    selection = {
        "languages": tuple(args.language or ()),
        "sample_ids": tuple(args.sample_id or ()),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    if args.preflight:
        report = preflight_run(args.dataset, config, **selection)
        _write_json(report.to_dict())
        return 0 if report.ok else 2
    if args.output is None:
        raise ValueError("--output is required unless --preflight is used")
    report = run_inference(args.dataset, config, args.output, resume=not args.no_resume, **selection)
    _write_json(report.to_dict())
    return 0 if report.status == "complete" else 2


def _models_command(args: argparse.Namespace) -> int:
    if args.key:
        _write_json(model_config(args.key).to_dict())
        return 0
    if args.json:
        _write_json(model_catalog().to_dict())
        return 0
    headers = ("KEY", "TASK", "ACCESS", "EXECUTION", "REPRODUCIBILITY", "MODEL")
    rows = [
        (config.key, config.task, config.access, config.execution, config.reproducibility, config.model_id)
        for config in list_model_configs()
    ]
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    return 0


def parser() -> argparse.ArgumentParser:
    """Build the public command-line parser."""

    value = argparse.ArgumentParser(
        prog="dai-asr-i18n",
        description="Reproducible multilingual ASR benchmarking.",
    )
    value.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = value.add_subparsers(dest="command")
    normalize_parser = subcommands.add_parser("normalize", help="normalize text with the benchmark or stock Whisper")
    normalize_parser.add_argument("text", nargs="?", help="text to normalize; reads stdin when omitted")
    normalize_parser.add_argument("--language", "-l", required=True, help="BCP-47 language tag")
    normalize_parser.add_argument("--profile", choices=_PROFILES, default="dai_asr_i18n")
    normalize_parser.add_argument("--json", action="store_true", help="include normalization provenance")
    normalize_parser.set_defaults(handler=_normalize_command)

    score_parser = subcommands.add_parser("score", help="score one pair or a JSONL batch")
    score_input = score_parser.add_mutually_exclusive_group(required=True)
    score_input.add_argument("--input", help="JSONL path, or - for stdin")
    score_input.add_argument("--reference", help="reference text for single-pair scoring")
    score_parser.add_argument("--hypothesis", help="hypothesis text for single-pair scoring")
    score_parser.add_argument("--language", "-l", help="BCP-47 language tag for single-pair scoring")
    score_parser.add_argument("--output", "-o", help="JSON output path; stdout when omitted")
    score_parser.add_argument("--profile", choices=_PROFILES, default="dai_asr_i18n")
    score_parser.add_argument(
        "--metric",
        choices=("auto", "wer", "cer", "ser"),
        default="auto",
        help="metric override; WER needs word boundaries and SER is Vietnamese-only",
    )
    score_parser.add_argument(
        "--group-by",
        help="comma-separated additional input fields; model identity is always retained",
    )
    score_parser.set_defaults(handler=_score_command)

    jsonl_parser = subcommands.add_parser(
        "score-jsonl",
        help="deprecated alias for 'score --input'",
    )
    jsonl_parser.add_argument("input", help="JSONL path, or - for stdin")
    jsonl_parser.add_argument("--output", "-o", help="JSON output path; stdout when omitted")
    jsonl_parser.add_argument("--profile", choices=_PROFILES, default="dai_asr_i18n")
    jsonl_parser.add_argument("--metric", choices=("auto", "wer", "cer", "ser"), default="auto")
    jsonl_parser.add_argument(
        "--group-by",
        help="comma-separated additional input fields; model identity is always retained",
    )
    jsonl_parser.set_defaults(handler=_score_batch_command)

    manifest_parser = subcommands.add_parser("manifest", help="print runtime and normalizer provenance")
    manifest_parser.add_argument("--output", "-o", help="JSON output path; stdout when omitted")
    manifest_parser.add_argument("--no-resource-hashes", action="store_true")
    manifest_parser.set_defaults(handler=_manifest_command)

    mix_parser = subcommands.add_parser(
        "mix-mono",
        help="reconstruct the canonical mono inference input from two released channels",
    )
    mix_parser.add_argument(
        "--channel-1",
        dest="channel1",
        required=True,
        help="first lossless speaker-channel audio file",
    )
    mix_parser.add_argument(
        "--channel-2",
        dest="channel2",
        required=True,
        help="second lossless speaker-channel audio file",
    )
    mix_parser.add_argument("--output", "-o", required=True, help="destination PCM16 WAV path")
    mix_parser.add_argument(
        "--expected-sha256",
        help="fail without writing when the reconstructed WAV does not match this checksum",
    )
    mix_parser.set_defaults(handler=_mix_mono_command)

    dataset_parser = subcommands.add_parser("dataset", help="download and verify pinned benchmark datasets")
    dataset_commands = dataset_parser.add_subparsers(dest="dataset_command", required=True)

    pull_parser = dataset_commands.add_parser("pull", help="selectively download a Hugging Face dataset")
    pull_parser.add_argument("--repo", required=True, help="Hugging Face dataset repository, organization/name")
    pull_parser.add_argument(
        "--revision",
        required=True,
        help="full Hugging Face commit SHA",
    )
    pull_parser.add_argument(
        "--manifest",
        help="legacy manifest path; omit for the release-table dataset format",
    )
    pull_parser.add_argument(
        "--format",
        dest="dataset_format",
        choices=("auto", "release", "manifest"),
        default="auto",
        help="remote repository layout; auto detects release Parquet tables (default: auto)",
    )
    pull_parser.add_argument("--output", help="local dataset directory; defaults under .artifacts/datasets")
    pull_parser.add_argument("--language", action="append", help="include one language; repeat to include several")
    pull_parser.add_argument(
        "--sample-id",
        "--clip-id",
        dest="clip_id",
        action="append",
        help="include one sample ID; repeat to include several (--clip-id is a compatibility alias)",
    )
    pull_parser.add_argument("--limit", type=int, help="take the first N rows after filtering")
    pull_parser.add_argument("--workers", type=int, default=8, help="parallel file downloads")
    pull_parser.add_argument(
        "--authenticated",
        action="store_true",
        help="use the logged-in Hugging Face identity instead of anonymous access",
    )
    pull_parser.add_argument(
        "--allow-floating-revision",
        action="store_true",
        help="allow an unpinned revision for exploration; unsuitable for published results",
    )
    pull_parser.set_defaults(handler=_dataset_pull_command)

    verify_parser = dataset_commands.add_parser("verify", help="verify selection provenance and audio checksums")
    verify_parser.add_argument("dataset", help="materialized dataset selection directory")
    verify_parser.set_defaults(handler=_dataset_verify_command)

    run_parser = subcommands.add_parser("run", help="run a pinned local or hosted model over a downloaded dataset")
    run_parser.add_argument("--dataset", required=True, help="materialized dataset selection directory")
    run_parser.add_argument(
        "--config",
        required=True,
        help="TOML path or built-in config name, such as whisper-large-v3-portable",
    )
    run_parser.add_argument("--output", help="run output directory")
    run_parser.add_argument("--language", action="append", help="run one language; repeat to select several")
    run_parser.add_argument("--sample-id", action="append", help="run one sample ID; repeat to select several")
    run_parser.add_argument("--shard-index", type=int, default=0, help="zero-based deterministic shard index")
    run_parser.add_argument("--shard-count", type=int, default=1, help="total number of deterministic shards")
    run_parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate the dataset, configuration, and runtime without loading weights or writing files",
    )
    run_parser.add_argument("--no-resume", action="store_true", help="refuse a non-empty output instead of resuming")
    run_parser.set_defaults(handler=_run_command)

    models_parser = subcommands.add_parser("models", help="inspect the frozen 16-system model catalog")
    models_parser.add_argument("key", nargs="?", help="show one model config as JSON")
    models_parser.add_argument("--json", action="store_true", help="show the complete catalog as JSON")
    models_parser.set_defaults(handler=_models_command)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the DAI-ASR-I18N CLI."""

    command = parser()
    args = command.parse_args(argv)
    if args.command == "score":
        if args.input is not None and (args.hypothesis is not None or args.language is not None):
            command.error("--hypothesis and --language cannot be used with --input")
        if args.input is None and (args.hypothesis is None or args.language is None):
            command.error("--reference requires --hypothesis and --language")
        if args.input is None and args.group_by is not None:
            command.error("--group-by can be used only with --input")
    if args.command == "models" and args.key is not None and args.json:
        command.error("--json cannot be combined with a model key")
    if handler := getattr(args, "handler", None):
        try:
            return handler(args)
        except (RuntimeError, ValueError) as error:
            command.error(str(error))
    command.print_help()
    return 0
