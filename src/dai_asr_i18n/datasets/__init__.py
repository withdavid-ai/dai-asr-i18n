"""Public dataset manifests, remote-source downloads, and integrity checks."""

from dai_asr_i18n.datasets.huggingface import HuggingFaceDatasetSource
from dai_asr_i18n.datasets.huggingface import pull_dataset as pull_huggingface_dataset
from dai_asr_i18n.datasets.models import (
    AlignedWord,
    AudioAsset,
    BenchmarkClip,
    DatasetSelection,
    ReferenceSegment,
    WordAlignmentSegment,
)
from dai_asr_i18n.datasets.source import DatasetSource, DatasetSourceIdentity, PullResult, materialize_dataset
from dai_asr_i18n.datasets.validation import VerificationReport, load_selection, verify_selection

__all__ = [
    "AlignedWord",
    "AudioAsset",
    "BenchmarkClip",
    "DatasetSource",
    "DatasetSourceIdentity",
    "DatasetSelection",
    "HuggingFaceDatasetSource",
    "PullResult",
    "ReferenceSegment",
    "WordAlignmentSegment",
    "VerificationReport",
    "load_selection",
    "materialize_dataset",
    "pull_huggingface_dataset",
    "verify_selection",
]
