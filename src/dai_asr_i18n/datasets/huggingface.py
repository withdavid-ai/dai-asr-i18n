"""Revision-pinned downloads from Hugging Face dataset repositories."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from dai_asr_i18n.datasets.models import (
    RECEIPT_FILENAME,
    SELECTION_FILENAME,
    AlignedWord,
    AudioAsset,
    BenchmarkClip,
    WordAlignmentSegment,
    huggingface_repo_id,
)
from dai_asr_i18n.datasets.source import DatasetSource, DatasetSourceIdentity, PullResult, materialize_dataset
from dai_asr_i18n.datasets.validation import load_selection, sha256_file, verify_selection

_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
_RELEASE_TABLE = re.compile(r"data/([a-z]{2,3}(?:-[a-z0-9]{1,8})*)/\1\.parquet")
_RELEASE_COLUMNS = {
    "sample_id",
    "language",
    "locale",
    "duration_type",
    "duration_s",
    "n_speakers",
    "topic",
    "overlap_type",
    "clip_overlap_frac",
    "speaker1_gender",
    "speaker2_gender",
    "benchmark_set",
    "word_alignments",
}
_OPTIONAL_RELEASE_COLUMNS = {"audio_ch1", "audio_ch2", "speaker1_id", "speaker2_id"}
DatasetFormat = Literal["auto", "release", "manifest"]


def _hub_api():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise ValueError('Hugging Face support is not installed; run: pip install "dai-asr-i18n[hf]"') from error
    return HfApi, hf_hub_download


def _parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ValueError('release-table support is not installed; run: pip install "dai-asr-i18n[hf]"') from error
    return parquet.read_table(path).to_pylist()


def _resolve_revision(
    repo_id: str,
    revision: str,
    *,
    authenticated: bool,
    allow_floating_revision: bool,
) -> tuple[Any, Any, str, bool]:
    pinned = revision.lower() if _FULL_COMMIT.fullmatch(revision.lower()) else None
    if not allow_floating_revision and pinned is None:
        raise ValueError(
            "Hugging Face revision must be a full 40-character commit SHA; "
            "use --allow-floating-revision only for exploration"
        )
    HfApi, hf_hub_download = _hub_api()
    token = True if authenticated else False
    api = HfApi()
    info = api.dataset_info(repo_id=repo_id, revision=pinned or revision, token=token)
    resolved = str(info.sha).lower()
    if not _FULL_COMMIT.fullmatch(resolved):
        raise ValueError(f"Hugging Face returned an invalid resolved revision {resolved!r}")
    if pinned is not None and resolved != pinned:
        raise ValueError(f"requested commit {pinned} resolved unexpectedly to {resolved}")
    return api, hf_hub_download, resolved, token


@dataclass
class HuggingFaceDatasetSource(DatasetSource):
    """Legacy manifest adapter retained for compatible public repositories."""

    repo_id: str
    revision: str
    authenticated: bool = False
    allow_floating_revision: bool = False

    def __post_init__(self) -> None:
        self.repo_id = huggingface_repo_id(self.repo_id)
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("revision must be a non-empty string")
        self.revision = self.revision.strip()
        if not isinstance(self.authenticated, bool) or not isinstance(self.allow_floating_revision, bool):
            raise ValueError("authenticated and allow_floating_revision must be booleans")

    def prepare(self, manifest: str, staging: Path) -> tuple[DatasetSourceIdentity, Path]:
        del staging
        _, hf_hub_download, resolved, token = _resolve_revision(
            self.repo_id,
            self.revision,
            authenticated=self.authenticated,
            allow_floating_revision=self.allow_floating_revision,
        )
        manifest_path = Path(
            hf_hub_download(
                repo_id=self.repo_id,
                filename=manifest,
                repo_type="dataset",
                revision=resolved,
                token=token,
            )
        )
        return (
            DatasetSourceIdentity(
                source_type="huggingface",
                source_id=self.repo_id,
                revision=self.revision,
                resolved_revision=resolved,
                revision_type="git_commit",
                source_manifest_sha256=sha256_file(manifest_path),
                authenticated=self.authenticated,
            ),
            manifest_path,
        )

    def download_file(self, filename: str, root: Path, identity: DatasetSourceIdentity) -> None:
        _, hf_hub_download = _hub_api()
        cached = Path(
            hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                repo_type="dataset",
                revision=identity.resolved_revision,
                token=True if self.authenticated else False,
            )
        )
        _copy_atomic(cached, root.joinpath(*PurePosixPath(filename).parts))


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError(f"refusing to replace unsafe dataset path {destination}")
    temporary = destination.with_name(f".{destination.name}.download")
    if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
        raise ValueError(f"refusing to replace unsafe temporary download {temporary}")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_bytes_atomic(data: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError(f"refusing to replace unsafe dataset path {destination}")
    if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
        raise ValueError(f"refusing to replace unsafe temporary download {temporary}")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_bytes(data)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _speaker_channel(value: object, field: str) -> str:
    if isinstance(value, bool) or value not in {1, 2}:
        raise ValueError(f"{field} must be speaker 1 or 2")
    return f"ch{value}"


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _release_alignments(row: dict[str, Any], field: str) -> dict[str, tuple[WordAlignmentSegment, ...]]:
    values = row.get(field)
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list")
    result: dict[str, list[WordAlignmentSegment]] = {"ch1": [], "ch2": []}
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        channel = _speaker_channel(value.get("speaker"), f"{field}[{index}].speaker")
        words_value = value.get("words")
        if not isinstance(words_value, list):
            raise ValueError(f"{field}[{index}].words must be a list")
        words: list[AlignedWord] = []
        for word_index, word in enumerate(words_value):
            if not isinstance(word, dict) or not isinstance(word.get("text"), str):
                raise ValueError(f"{field}[{index}].words[{word_index}] must contain string text")
            words.append(
                AlignedWord(
                    text=word["text"],
                    start_s=_number(word.get("start"), f"{field}[{index}].words[{word_index}].start"),
                    end_s=_number(word.get("end"), f"{field}[{index}].words[{word_index}].end"),
                )
            )
        timing_source = value.get("timing_source")
        if timing_source is not None and not isinstance(timing_source, str):
            raise ValueError(f"{field}[{index}].timing_source must be a string or null")
        result[channel].append(
            WordAlignmentSegment(
                start_s=_number(value.get("start"), f"{field}[{index}].start"),
                end_s=_number(value.get("end"), f"{field}[{index}].end"),
                timing_source=timing_source,
                words=tuple(words),
            )
        )
    return {channel: tuple(segments) for channel, segments in result.items()}


def _validate_release_row(row: object, table: str, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{table} row {index} must be an object")
    missing = _RELEASE_COLUMNS - set(row)
    if missing:
        raise ValueError(f"{table} row {index} is missing columns: {', '.join(sorted(missing))}")
    unexpected = set(row) - _RELEASE_COLUMNS - _OPTIONAL_RELEASE_COLUMNS
    if unexpected:
        raise ValueError(f"{table} contains noncanonical release columns: {', '.join(sorted(unexpected))}")
    sample_id = row.get("sample_id")
    language = row.get("language")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{table} row {index} has an invalid sample_id")
    if not isinstance(language, str) or not language:
        raise ValueError(f"{table} row {index} has an invalid language")
    for field in ("speaker1_id", "speaker2_id"):
        value = row.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{table} row {index} has an invalid {field}")
    table_match = _RELEASE_TABLE.fullmatch(table)
    if table_match is None or language != table_match.group(1):
        raise ValueError(f"{table} row {index} language must match its per-language table")
    return row


def _combined_table_digest(tables: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for filename, path in sorted(tables):
        digest.update(filename.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _write_release_metadata(
    destination: Path,
    clips: tuple[BenchmarkClip, ...],
    identity: DatasetSourceIdentity,
) -> None:
    selection = destination / SELECTION_FILENAME
    selection_tmp = destination / f".{SELECTION_FILENAME}.tmp"
    receipt_tmp = destination / f".{RECEIPT_FILENAME}.tmp"
    rendered = "".join(
        json.dumps(clip.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for clip in clips
    )
    selection_tmp.write_text(rendered, encoding="utf-8")
    receipt = identity.receipt("data/parquet-index", sha256_file(selection_tmp))
    receipt_tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection_tmp.replace(selection)
    receipt_tmp.replace(destination / RECEIPT_FILENAME)


def _embedded_audio(row: dict[str, Any], channel: str) -> bytes | None:
    value = row.get(f"audio_{channel}")
    if not isinstance(value, dict):
        return None
    data = value.get("bytes")
    if data is None:
        return None
    if not isinstance(data, bytes):
        raise ValueError(f"audio_{channel}.bytes must be bytes")
    return data


def _pull_release_dataset(
    *,
    repo_id: str,
    revision: str,
    resolved: str,
    repo_files: list[str],
    hf_hub_download: Any,
    token: bool,
    output: str | Path | None,
    languages: tuple[str, ...],
    clip_ids: tuple[str, ...],
    limit: int | None,
    authenticated: bool,
    workers: int,
) -> PullResult:
    available = {match.group(1): filename for filename in repo_files if (match := _RELEASE_TABLE.fullmatch(filename))}
    requested_languages = {language.replace("_", "-").lower() for language in languages}
    if requested_languages:
        missing_languages = requested_languages - set(available)
        if missing_languages:
            raise ValueError(f"requested languages are not present: {', '.join(sorted(missing_languages))}")
        table_filenames = [available[language] for language in sorted(requested_languages)]
    else:
        table_filenames = [available[language] for language in sorted(available)]
    if not table_filenames:
        raise ValueError("the Hugging Face repository contains no data/<lang>/<lang>.parquet release tables")

    tables = [
        (
            filename,
            Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    revision=resolved,
                    token=token,
                )
            ),
        )
        for filename in table_filenames
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for filename, path in tables:
        for index, raw_row in enumerate(_parquet_rows(path)):
            row = _validate_release_row(raw_row, filename, index)
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id {sample_id!r} across release tables")
            seen.add(sample_id)
            rows.append(row)
    requested_ids = set(clip_ids)
    if requested_ids:
        missing_ids = requested_ids - seen
        if missing_ids:
            raise ValueError(f"requested sample IDs were not found after filtering: {', '.join(sorted(missing_ids))}")
        rows = [row for row in rows if row["sample_id"] in requested_ids]
    rows.sort(key=lambda row: (row["language"], row["sample_id"]))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("the requested filters selected no clips")

    destination = (
        (
            Path(output)
            if output is not None
            else Path(".artifacts") / "datasets" / repo_id.replace("/", "--") / resolved
        )
        .expanduser()
        .resolve()
    )
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"dataset output path is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"refusing to use dataset output containing symbolic link {path}")
    receipt_path = destination / RECEIPT_FILENAME
    if receipt_path.is_file():
        existing = load_selection(destination)
        if existing.source_id != repo_id or existing.resolved_revision != resolved:
            raise ValueError("output directory belongs to a different dataset source or revision")
        if {clip.clip_id for clip in existing.clips} != {row["sample_id"] for row in rows}:
            raise ValueError("output directory already contains a different clip selection; choose a new output")
        report = verify_selection(destination)
        return PullResult(
            output=destination,
            clips=len(existing.clips),
            files=report.files_checked,
            bytes=report.total_bytes,
            source_type="huggingface",
            source_id=repo_id,
            resolved_revision=resolved,
        )
    elif any(destination.iterdir()):
        expected = {f"audio/{row['sample_id']}/{channel}.flac" for row in rows for channel in ("ch1", "ch2")}
        existing = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
        unexpected = {
            path for path in existing if path not in expected and not PurePosixPath(path).name.startswith(".")
        }
        if unexpected:
            raise ValueError(
                f"refusing to write into non-empty dataset directory; unexpected files: {', '.join(sorted(unexpected))}"
            )

    def materialize_audio(item: tuple[dict[str, Any], str]) -> tuple[str, str, AudioAsset]:
        row, channel = item
        sample_id = row["sample_id"]
        relative = f"audio/{sample_id}/{channel}.flac"
        target = destination.joinpath(*PurePosixPath(relative).parts)
        data = _embedded_audio(row, channel)
        if data is not None:
            _write_bytes_atomic(data, target)
        else:
            cached = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=relative,
                    repo_type="dataset",
                    revision=resolved,
                    token=token,
                )
            )
            _copy_atomic(cached, target)
        return sample_id, channel, AudioAsset(relative, sha256_file(target), float(row["duration_s"]))

    items = [(row, channel) for row in rows for channel in ("ch1", "ch2")]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        materialized = list(pool.map(materialize_audio, items))
    assets = {(sample_id, channel): asset for sample_id, channel, asset in materialized}

    clips: list[BenchmarkClip] = []
    for row in rows:
        sample_id = row["sample_id"]
        metadata = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "sample_id",
                "language",
                "duration_s",
                "word_alignments",
                "audio_ch1",
                "audio_ch2",
            }
        }
        clip = BenchmarkClip(
            clip_id=sample_id,
            language=row["language"],
            duration_s=float(row["duration_s"]),
            audio={channel: assets[(sample_id, channel)] for channel in ("ch1", "ch2")},
            references={},
            word_alignments=_release_alignments(row, "word_alignments"),
            metadata=metadata,
        )
        clips.append(BenchmarkClip.from_dict(clip.to_dict()))
    clip_tuple = tuple(clips)
    identity = DatasetSourceIdentity(
        source_type="huggingface",
        source_id=repo_id,
        revision=revision,
        resolved_revision=resolved,
        revision_type="git_commit",
        source_manifest_sha256=_combined_table_digest(tables),
        authenticated=authenticated,
    )
    _write_release_metadata(destination, clip_tuple, identity)
    report = verify_selection(destination)
    return PullResult(
        output=destination,
        clips=len(clip_tuple),
        files=report.files_checked,
        bytes=report.total_bytes,
        source_type="huggingface",
        source_id=repo_id,
        resolved_revision=resolved,
    )


def pull_dataset(
    *,
    repo_id: str,
    revision: str,
    manifest: str | None = None,
    output: str | Path | None = None,
    languages: tuple[str, ...] = (),
    clip_ids: tuple[str, ...] = (),
    limit: int | None = None,
    authenticated: bool = False,
    workers: int = 8,
    allow_floating_revision: bool = False,
    dataset_format: DatasetFormat = "auto",
) -> PullResult:
    """Materialize a release-table or legacy-manifest Hugging Face dataset revision."""

    repo_id = huggingface_repo_id(repo_id)
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty string")
    if dataset_format not in {"auto", "release", "manifest"}:
        raise ValueError("dataset_format must be auto, release, or manifest")
    if manifest is not None and dataset_format == "release":
        raise ValueError("--manifest cannot be combined with the release dataset format")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if manifest is not None or dataset_format == "manifest":
        return materialize_dataset(
            HuggingFaceDatasetSource(
                repo_id=repo_id,
                revision=revision,
                authenticated=authenticated,
                allow_floating_revision=allow_floating_revision,
            ),
            manifest=manifest or "manifest/public-v1.jsonl",
            output=output,
            languages=languages,
            clip_ids=clip_ids,
            limit=limit,
            workers=workers,
        )

    api, hf_hub_download, resolved, token = _resolve_revision(
        repo_id,
        revision.strip(),
        authenticated=authenticated,
        allow_floating_revision=allow_floating_revision,
    )
    try:
        repo_files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=resolved, token=token))
    except AttributeError:
        if dataset_format == "release":
            raise ValueError("the installed huggingface-hub client cannot enumerate release tables") from None
        return materialize_dataset(
            HuggingFaceDatasetSource(
                repo_id=repo_id,
                revision=resolved,
                authenticated=authenticated,
                allow_floating_revision=False,
            ),
            manifest="manifest/public-v1.jsonl",
            output=output,
            languages=languages,
            clip_ids=clip_ids,
            limit=limit,
            workers=workers,
        )
    if any(_RELEASE_TABLE.fullmatch(filename) for filename in repo_files):
        return _pull_release_dataset(
            repo_id=repo_id,
            revision=revision.strip(),
            resolved=resolved,
            repo_files=repo_files,
            hf_hub_download=hf_hub_download,
            token=token,
            output=output,
            languages=languages,
            clip_ids=clip_ids,
            limit=limit,
            authenticated=authenticated,
            workers=workers,
        )
    if dataset_format == "release":
        raise ValueError("the Hugging Face repository does not contain release-table Parquet files")
    return materialize_dataset(
        HuggingFaceDatasetSource(
            repo_id=repo_id,
            revision=resolved,
            authenticated=authenticated,
            allow_floating_revision=False,
        ),
        manifest="manifest/public-v1.jsonl",
        output=output,
        languages=languages,
        clip_ids=clip_ids,
        limit=limit,
        workers=workers,
    )


__all__ = ["DatasetFormat", "HuggingFaceDatasetSource", "PullResult", "pull_dataset"]
