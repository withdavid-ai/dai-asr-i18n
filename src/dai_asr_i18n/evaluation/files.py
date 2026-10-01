"""Checksum-verified loading of the downloaded release directory."""

import hashlib
import json
from pathlib import Path

from dai_asr_i18n.evaluation.references import reference_from_alignment


def load_reference(directory: str | Path):
    """Read metadata and verify the aligned transcript against its recorded SHA-256.

    Metadata must come from a trusted, pinned release manifest. Its duration values
    are the physical audio durations.
    This does not download or alter any audio.
    """
    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    body = (root / "scoring_reference.json").read_bytes()
    if hashlib.sha256(body).hexdigest() != metadata["reference"]["sha256"]:
        raise ValueError("scoring_reference.json checksum does not match metadata")
    document = json.loads(body)
    if document["jobId"] != metadata["item_key"] or document["lang"] != metadata["metadata"]["language"]:
        raise ValueError("reference identity/language does not match metadata")
    return reference_from_alignment(
        document,
        channel_durations_s={ch: metadata["audio"][ch]["duration_s"] for ch in ("ch1", "ch2")},
    )
