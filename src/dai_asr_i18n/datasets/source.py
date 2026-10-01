"""Provider-neutral dataset selection and materialization."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from dai_asr_i18n.datasets.models import RECEIPT_FILENAME, SELECTION_FILENAME, BenchmarkClip, safe_relative_path
from dai_asr_i18n.datasets.validation import load_manifest, sha256_file, verify_clip_assets


@dataclass(frozen=True)
class DatasetSourceIdentity:
    """Immutable provenance for one prepared remote dataset source."""

    source_type: str
    source_id: str
    revision: str
    resolved_revision: str
    revision_type: str
    source_manifest_sha256: str
    authenticated: bool

    def receipt(self, manifest: str, selection_sha256: str) -> dict[str, object]:
        return {
            "schema_version": 2,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "revision": self.revision,
            "resolved_revision": self.resolved_revision,
            "revision_type": self.revision_type,
            "source_manifest": manifest,
            "source_manifest_sha256": self.source_manifest_sha256,
            "selection_sha256": selection_sha256,
            "authenticated": self.authenticated,
        }

    def output_slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "--", self.source_id).strip("-.")


class DatasetSource(ABC):
    """Remote object source capable of resolving and downloading one manifest tree."""

    @abstractmethod
    def prepare(self, manifest: str, staging: Path) -> tuple[DatasetSourceIdentity, Path]:
        """Resolve immutable identity and fetch the source manifest."""

    @abstractmethod
    def download_file(self, filename: str, root: Path, identity: DatasetSourceIdentity) -> None:
        """Download one safe manifest-relative filename beneath ``root``."""


@dataclass(frozen=True)
class PullResult:
    """Summary of a fully verified local dataset selection."""

    output: Path
    clips: int
    files: int
    bytes: int
    source_type: str
    source_id: str
    resolved_revision: str

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": True,
            "output": str(self.output),
            "clips": self.clips,
            "files": self.files,
            "bytes": self.bytes,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "resolved_revision": self.resolved_revision,
        }


def _write_selection(path: Path, clips: tuple[BenchmarkClip, ...]) -> None:
    rendered = "".join(
        json.dumps(clip.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for clip in clips
    )
    path.write_text(rendered, encoding="utf-8")


def _select_clips(
    clips: tuple[BenchmarkClip, ...],
    *,
    languages: tuple[str, ...],
    clip_ids: tuple[str, ...],
    limit: int | None,
) -> tuple[BenchmarkClip, ...]:
    language_set = {language.replace("_", "-").lower() for language in languages}
    clip_id_set = set(clip_ids)
    if language_set:
        clips = tuple(clip for clip in clips if clip.language in language_set)
    if clip_id_set:
        missing = clip_id_set - {clip.clip_id for clip in clips}
        if missing:
            raise ValueError(f"requested sample IDs were not found after filtering: {', '.join(sorted(missing))}")
        clips = tuple(clip for clip in clips if clip.clip_id in clip_id_set)
    if limit is not None:
        clips = clips[:limit]
    if not clips:
        raise ValueError("the requested filters selected no clips")
    return clips


def _existing_receipt_identity(receipt: object) -> tuple[object, ...]:
    if not isinstance(receipt, dict):
        return ()
    if receipt.get("schema_version") == 1:
        return (
            "huggingface",
            receipt.get("repo_id"),
            receipt.get("resolved_revision"),
            receipt.get("source_manifest"),
        )
    return (
        receipt.get("source_type"),
        receipt.get("source_id"),
        receipt.get("resolved_revision"),
        receipt.get("source_manifest"),
    )


def _assert_safe_download_path(root: Path, filename: str) -> Path:
    """Reject existing links or non-directories that could redirect a download."""

    parts = PurePosixPath(filename).parts
    current = root
    for part in parts[:-1]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"refusing to download through symbolic-link directory {current}")
        if current.exists() and not current.is_dir():
            raise ValueError(f"refusing to download through non-directory path {current}")
    path = root.joinpath(*parts)
    if path.is_symlink():
        raise ValueError(f"refusing to replace symbolic-link asset {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"refusing to replace non-file asset {path}")
    return path


def _known_incomplete_files(filenames: list[str]) -> set[str]:
    # A crash can occur after the selection rename but before the receipt rename.
    result = {SELECTION_FILENAME, f".{SELECTION_FILENAME}.tmp", f".{RECEIPT_FILENAME}.tmp"}
    for filename in filenames:
        path = PurePosixPath(filename)
        result.add((path.parent / f".{path.name}.download").as_posix())
    return result


def _clear_atomic_temporary(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        if not path.is_file():
            raise ValueError(f"refusing to replace non-file temporary output {path}")
        path.unlink()


def materialize_dataset(
    source: DatasetSource,
    *,
    manifest: str = "manifest/public-v1.jsonl",
    output: str | Path | None = None,
    languages: tuple[str, ...] = (),
    clip_ids: tuple[str, ...] = (),
    limit: int | None = None,
    workers: int = 8,
) -> PullResult:
    """Download, filter, materialize, and verify a selection from any source adapter."""

    manifest = safe_relative_path(manifest, "manifest")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")

    with TemporaryDirectory(prefix="dai-asr-i18n-manifest-") as temporary:
        identity, manifest_path = source.prepare(manifest, Path(temporary))
        clips = _select_clips(
            load_manifest(manifest_path),
            languages=languages,
            clip_ids=clip_ids,
            limit=limit,
        )

    assets = {}
    for clip in clips:
        for asset in clip.audio.values():
            existing = assets.get(asset.path)
            if existing is not None and existing != asset:
                raise ValueError(f"manifest uses audio path {asset.path!r} with conflicting metadata")
            assets[asset.path] = asset
    filenames = sorted(assets)
    destination = (
        Path(output)
        if output is not None
        else Path(".artifacts") / "datasets" / identity.output_slug() / identity.resolved_revision
    )
    destination = destination.expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"dataset output path is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for existing in destination.rglob("*"):
        if existing.is_symlink():
            raise ValueError(f"refusing to use dataset output containing symbolic link {existing}")
    for filename in filenames:
        _assert_safe_download_path(destination, filename)
    receipt_path = destination / RECEIPT_FILENAME
    if any(destination.iterdir()):
        if not receipt_path.is_file():
            existing_files = {
                path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()
            }
            unexpected = existing_files - set(filenames) - _known_incomplete_files(filenames)
            if unexpected:
                raise ValueError(
                    f"refusing to write into non-empty directory without {RECEIPT_FILENAME}; "
                    f"unexpected files: {', '.join(sorted(unexpected))}"
                )
        else:
            try:
                existing_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid existing {receipt_path}") from error
            expected_identity = (
                identity.source_type,
                identity.source_id,
                identity.resolved_revision,
                manifest,
            )
            if _existing_receipt_identity(existing_receipt) != expected_identity:
                raise ValueError("output directory belongs to a different dataset source, revision, or manifest")
            selection_path = destination / SELECTION_FILENAME
            if not selection_path.is_file():
                raise ValueError(f"existing dataset receipt is missing {SELECTION_FILENAME}")
            existing_clips = load_manifest(selection_path)
            if existing_clips != clips:
                raise ValueError("output directory already contains a different clip selection; choose a new output")

    def download(filename: str) -> None:
        path = _assert_safe_download_path(destination, filename)
        if path.is_file() and sha256_file(path) == assets[filename].sha256:
            return
        source.download_file(filename, destination, identity)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(download, filenames))

    files_checked, total_bytes = verify_clip_assets(destination, clips)
    selection_tmp = destination / f".{SELECTION_FILENAME}.tmp"
    receipt_tmp = destination / f".{RECEIPT_FILENAME}.tmp"
    _clear_atomic_temporary(selection_tmp)
    _clear_atomic_temporary(receipt_tmp)
    _write_selection(selection_tmp, clips)
    receipt = identity.receipt(manifest, sha256_file(selection_tmp))
    receipt_tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    selection_tmp.replace(destination / SELECTION_FILENAME)
    receipt_tmp.replace(receipt_path)
    return PullResult(
        output=destination,
        clips=len(clips),
        files=files_checked,
        bytes=total_bytes,
        source_type=identity.source_type,
        source_id=identity.source_id,
        resolved_revision=identity.resolved_revision,
    )


__all__ = ["DatasetSource", "DatasetSourceIdentity", "PullResult", "materialize_dataset"]
