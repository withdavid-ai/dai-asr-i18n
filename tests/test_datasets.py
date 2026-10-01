from __future__ import annotations

import hashlib
import json
import shutil
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
import soundfile as sf

import dai_asr_i18n.datasets.huggingface as huggingface
from dai_asr_i18n.datasets.huggingface import pull_dataset
from dai_asr_i18n.datasets.models import RECEIPT_FILENAME, SELECTION_FILENAME, BenchmarkClip
from dai_asr_i18n.datasets.validation import load_manifest, verify_selection

REVISION = "1" * 40


def _wav(path: Path, *, frames: int = 160) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * frames)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(root: Path, clip_id: str = "clip-1", language: str = "en") -> dict[str, object]:
    ch1 = root / "audio" / clip_id / "ch1.wav"
    ch2 = root / "audio" / clip_id / "ch2.wav"
    return {
        "schema_version": 1,
        "clip_id": clip_id,
        "language": language,
        "duration_s": 0.01,
        "audio": {
            "ch1": {"path": f"audio/{clip_id}/ch1.wav", "sha256": _wav(ch1), "duration_s": 0.01},
            "ch2": {"path": f"audio/{clip_id}/ch2.wav", "sha256": _wav(ch2), "duration_s": 0.01},
        },
        "references": {
            "ch1": [{"text": "hello", "start_s": 0.0, "end_s": 0.005}],
            "ch2": [{"text": "world", "start_s": 0.005, "end_s": 0.01}],
        },
        "metadata": {"source": "test"},
    }


def _materialized(root: Path) -> Path:
    row = _row(root)
    (root / SELECTION_FILENAME).write_text(json.dumps(row) + "\n", encoding="utf-8")
    (root / RECEIPT_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_id": "example/public-asr",
                "revision": REVISION,
                "resolved_revision": REVISION,
                "source_manifest": "manifest/public-v1.jsonl",
                "authenticated": False,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_manifest_rejects_unsafe_paths(tmp_path: Path):
    row = _row(tmp_path)
    row["audio"]["ch1"]["path"] = "../secret.wav"  # type: ignore[index]
    with pytest.raises(ValueError, match="safe repository-relative"):
        BenchmarkClip.from_dict(row)


def test_manifest_rejects_windows_style_paths(tmp_path: Path):
    row = _row(tmp_path)
    row["audio"]["ch1"]["path"] = "..\\secret.wav"  # type: ignore[index]
    with pytest.raises(ValueError, match="safe repository-relative POSIX"):
        BenchmarkClip.from_dict(row)


def test_manifest_allows_float32_timing_rounding_but_rejects_material_overshoot(tmp_path: Path):
    row = _row(tmp_path)
    row["references"]["ch2"][0]["end_s"] = row["duration_s"] + 0.01  # type: ignore[index,operator]
    BenchmarkClip.from_dict(row)

    row["references"]["ch2"][0]["end_s"] = row["duration_s"] + 0.051  # type: ignore[index,operator]
    with pytest.raises(ValueError, match="exceeds clip duration"):
        BenchmarkClip.from_dict(row)


def test_manifest_uses_dataset_card_sample_id_and_accepts_legacy_clip_id(tmp_path: Path):
    row = _row(tmp_path)
    row["sample_id"] = row.pop("clip_id")
    clip = BenchmarkClip.from_dict(row)
    assert clip.clip_id == "clip-1"
    assert clip.to_dict()["sample_id"] == "clip-1"
    assert "clip_id" not in clip.to_dict()

    row["clip_id"] = "legacy-id"
    with pytest.raises(ValueError, match="both sample_id and clip_id"):
        BenchmarkClip.from_dict(row)


def test_verify_selection_checks_every_digest(tmp_path: Path):
    root = _materialized(tmp_path)
    report = verify_selection(root)
    assert report.files_checked == 2
    assert report.to_dict()["languages"] == ["en"]

    (root / "audio/clip-1/ch1.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_selection(root)


def test_manifest_and_verifier_accept_release_flac_assets(tmp_path: Path):
    root = _materialized(tmp_path)
    row = json.loads((root / SELECTION_FILENAME).read_text(encoding="utf-8"))
    for channel in ("ch1", "ch2"):
        path = root / f"audio/clip-1/{channel}.flac"
        sf.write(path, [0.0] * 160, 16000, format="FLAC", subtype="PCM_16")
        row["audio"][channel] = {  # type: ignore[index]
            "path": f"audio/clip-1/{channel}.flac",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "duration_s": 0.01,
        }
    (root / SELECTION_FILENAME).write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = verify_selection(root)

    assert report.files_checked == 2


def test_offline_verification_rejects_symlinked_audio(tmp_path: Path):
    root = _materialized(tmp_path)
    asset = root / "audio/clip-1/ch1.wav"
    target = root / "audio/clip-1/ch1-real.wav"
    asset.replace(target)
    asset.symlink_to(target.name)

    with pytest.raises(ValueError, match="symbolic-link path"):
        verify_selection(root)


def test_load_manifest_rejects_duplicate_sample_ids(tmp_path: Path):
    row = _row(tmp_path)
    manifest = tmp_path / "duplicates.jsonl"
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_manifest(manifest)


def test_verify_selection_rejects_unpinned_receipt(tmp_path: Path):
    root = _materialized(tmp_path)
    receipt_path = root / RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["resolved_revision"] = "main"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="full 40-character commit SHA"):
        verify_selection(root)


def test_pull_dataset_is_anonymous_selective_and_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    remote = tmp_path / "remote"
    manifest = remote / "manifest/public-v1.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = [_row(remote, "clip-en", "en"), _row(remote, "clip-hi", "hi")]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    tokens: list[bool] = []

    class FakeApi:
        def dataset_info(self, *, repo_id: str, revision: str, token: bool):
            assert repo_id == "example/public-asr"
            assert revision == REVISION
            tokens.append(token)
            return SimpleNamespace(sha=REVISION)

    def fake_download(*, filename: str, local_dir: Path | None = None, token: bool, **_: object) -> str:
        tokens.append(token)
        source = remote / filename
        if local_dir is None:
            return str(source)
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return str(destination)

    monkeypatch.setattr("dai_asr_i18n.datasets.huggingface._hub_api", lambda: (FakeApi, fake_download))
    output = tmp_path / "selection"
    incomplete = output / "audio/clip-hi/.ch1.wav.download"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_bytes(b"interrupted")
    result = pull_dataset(
        repo_id="example/public-asr",
        revision=REVISION,
        output=output,
        languages=("hi",),
    )
    assert result.clips == 1
    assert not any(tokens)
    assert not (output / "audio/clip-en/ch1.wav").exists()
    assert (output / "audio/clip-hi/ch1.wav").exists()
    assert not incomplete.exists()
    selection = verify_selection(output).dataset
    assert selection.source_type == "huggingface"
    assert selection.source_id == "example/public-asr"

    with pytest.raises(ValueError, match="different clip selection"):
        pull_dataset(
            repo_id="example/public-asr",
            revision=REVISION,
            output=output,
            languages=("en",),
        )
    assert [clip.clip_id for clip in verify_selection(output).dataset.clips] == ["clip-hi"]
    receipt = json.loads((output / RECEIPT_FILENAME).read_text(encoding="utf-8"))
    assert receipt["source_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert receipt["selection_sha256"] == hashlib.sha256((output / SELECTION_FILENAME).read_bytes()).hexdigest()

    selected_row = json.loads((output / SELECTION_FILENAME).read_text(encoding="utf-8"))
    selected_row["metadata"]["tampered"] = True
    (output / SELECTION_FILENAME).write_text(json.dumps(selected_row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selection checksum mismatch"):
        verify_selection(output)


def _release_row(sample_id: str = "8656cddd-a0d3-5348-931d-ace4c8614292") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "language": "ar",
        "locale": "ar-EG",
        "duration_type": "short",
        "duration_s": 0.01,
        "n_speakers": 2,
        "topic": "travel",
        "overlap_type": "low",
        "clip_overlap_frac": 0.0,
        "speaker1_gender": "Male",
        "speaker2_gender": "Female",
        "speaker1_id": "d2b9f017-9e88-558b-89ab-63f33df29612",
        "speaker2_id": "8fc7141a-3344-58bb-a525-6cd73bc48bb8",
        "benchmark_set": "public",
        "word_alignments": [
            {
                "speaker": 1,
                "start": 0.0,
                "end": 0.005,
                "timing_source": None,
                "words": [{"start": 0.0, "end": 0.005, "text": "hello"}],
            },
            {
                "speaker": 2,
                "start": 0.005,
                "end": 0.01,
                "timing_source": "pseudo",
                "words": [{"start": 0.005, "end": 0.01, "text": "world"}],
            },
        ],
    }


def test_release_rows_reject_noncanonical_columns_and_language_mismatch():
    row = _release_row()
    row["conversation_id"] = "must-not-leak"
    with pytest.raises(ValueError, match="noncanonical release columns: conversation_id"):
        huggingface._validate_release_row(row, "data/ar/ar.parquet", 0)

    row.pop("conversation_id")
    row["language"] = "en"
    with pytest.raises(ValueError, match="language must match"):
        huggingface._validate_release_row(row, "data/ar/ar.parquet", 0)


def test_release_rows_reject_removed_reference_column():
    row = _release_row()
    row["reference"] = []

    with pytest.raises(ValueError, match="noncanonical release columns: reference"):
        huggingface._validate_release_row(row, "data/ar/ar.parquet", 0)


def test_schema_v2_requires_complete_word_alignments_and_forbids_references(tmp_path: Path):
    row = _row(tmp_path)
    row["schema_version"] = 2
    row["word_alignments"] = {
        "ch1": [
            {
                "start_s": 0.0,
                "end_s": 0.005,
                "timing_source": "forced",
                "words": [{"text": "hello", "start_s": 0.0, "end_s": 0.005}],
            }
        ]
    }

    with pytest.raises(ValueError, match="references is not allowed"):
        BenchmarkClip.from_dict(row)

    del row["references"]
    with pytest.raises(ValueError, match="word_alignments must contain exactly"):
        BenchmarkClip.from_dict(row)


def test_pull_release_tables_maps_two_channel_schema_and_word_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    remote = tmp_path / "release"
    table = remote / "data/ar/ar.parquet"
    table.parent.mkdir(parents=True)
    table.write_bytes(b"parquet-fixture")
    row = _release_row()
    for channel in ("ch1", "ch2"):
        path = remote / f"audio/{row['sample_id']}/{channel}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, [0.0] * 160, 16000, format="FLAC", subtype="PCM_16")

    class FakeApi:
        def dataset_info(self, *, repo_id: str, revision: str, token: bool):
            assert (repo_id, revision, token) == ("example/public-asr", REVISION, False)
            return SimpleNamespace(sha=REVISION)

        def list_repo_files(self, **_: object):
            return ["README.md", "data/ar/ar.parquet", *[f"audio/{row['sample_id']}/ch{i}.flac" for i in (1, 2)]]

    downloads: list[str] = []

    def fake_download(*, filename: str, **_: object) -> str:
        downloads.append(filename)
        return str(remote / filename)

    monkeypatch.setattr("dai_asr_i18n.datasets.huggingface._hub_api", lambda: (FakeApi, fake_download))
    monkeypatch.setattr("dai_asr_i18n.datasets.huggingface._parquet_rows", lambda _: [row])
    output = tmp_path / "selection"

    result = pull_dataset(
        repo_id="example/public-asr",
        revision=REVISION,
        output=output,
        languages=("ar",),
    )

    assert result.clips == 1
    assert result.files == 2
    clip = verify_selection(output).dataset.clips[0]
    assert clip.reference_text("mono") == "hello world"
    assert clip.word_alignments["ch1"][0].words[0].text == "hello"
    assert clip.word_alignments["ch2"][0].timing_source == "pseudo"
    assert clip.metadata["benchmark_set"] == "public"
    assert clip.metadata["speaker1_id"] == "d2b9f017-9e88-558b-89ab-63f33df29612"
    assert clip.metadata["speaker2_id"] == "8fc7141a-3344-58bb-a525-6cd73bc48bb8"
    assert set(clip.audio) == {"ch1", "ch2"}
    selection_row = json.loads((output / SELECTION_FILENAME).read_text(encoding="utf-8"))
    assert selection_row["schema_version"] == 2
    assert "references" not in selection_row

    audio_downloads = [filename for filename in downloads if filename.startswith("audio/")]
    resumed = pull_dataset(
        repo_id="example/public-asr",
        revision=REVISION,
        output=output,
        languages=("ar",),
    )
    assert resumed.clips == 1
    assert [filename for filename in downloads if filename.startswith("audio/")] == audio_downloads


def test_pull_release_tables_supports_future_embedded_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    remote = tmp_path / "release"
    table = remote / "data/ar/ar.parquet"
    table.parent.mkdir(parents=True)
    table.write_bytes(b"parquet-fixture")
    row = _release_row()
    for channel in ("ch1", "ch2"):
        path = remote / f"{channel}.flac"
        sf.write(path, [0.0] * 160, 16000, format="FLAC", subtype="PCM_16")
        row[f"audio_{channel}"] = {"bytes": path.read_bytes(), "path": f"{row['sample_id']}_{channel}.flac"}

    class FakeApi:
        def dataset_info(self, **_: object):
            return SimpleNamespace(sha=REVISION)

        def list_repo_files(self, **_: object):
            return ["data/ar/ar.parquet"]

    def fake_download(*, filename: str, **_: object) -> str:
        assert filename == "data/ar/ar.parquet"
        return str(table)

    monkeypatch.setattr("dai_asr_i18n.datasets.huggingface._hub_api", lambda: (FakeApi, fake_download))
    monkeypatch.setattr("dai_asr_i18n.datasets.huggingface._parquet_rows", lambda _: [row])

    result = pull_dataset(
        repo_id="example/public-asr",
        revision=REVISION,
        output=tmp_path / "embedded",
        dataset_format="release",
    )

    assert result.files == 2


def test_pull_dataset_requires_a_commit_by_default():
    with pytest.raises(ValueError, match="full 40-character commit SHA"):
        pull_dataset(repo_id="example/public-asr", revision="main")

    with pytest.raises(ValueError, match="non-empty string"):
        pull_dataset(repo_id="example/public-asr", revision=" ", allow_floating_revision=True)


@pytest.mark.parametrize(
    "repo_id",
    ["../dataset", "organization/repository/extra", "organization/.repository", "organization/repo--name"],
)
def test_pull_dataset_rejects_invalid_hugging_face_repo_ids(repo_id: str):
    with pytest.raises(ValueError, match="valid Hugging Face"):
        pull_dataset(repo_id=repo_id, revision=REVISION)
