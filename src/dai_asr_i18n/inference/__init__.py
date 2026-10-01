"""Local and hosted inference backends with resumable benchmark execution."""

from dai_asr_i18n.inference.base import InferenceBackend, Transcript, TranscriptSegment, TranscriptWord
from dai_asr_i18n.inference.config import RunConfig, load_run_config
from dai_asr_i18n.inference.runner import PreflightReport, RunReport, preflight_run, run_inference
from dai_asr_i18n.inference.shards import merge_run_shards

__all__ = [
    "InferenceBackend",
    "PreflightReport",
    "RunConfig",
    "RunReport",
    "Transcript",
    "TranscriptSegment",
    "TranscriptWord",
    "load_run_config",
    "merge_run_shards",
    "preflight_run",
    "run_inference",
]
