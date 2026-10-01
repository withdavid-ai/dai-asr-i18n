"""Local Qwen3-ASR backend."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

from dai_asr_i18n.inference.base import InferenceBackend, Transcript
from dai_asr_i18n.inference.config import ModelConfig
from dai_asr_i18n.inference.windowing import planned_audio_windows, stitch_transcripts

_OPTIONS = {"max_new_tokens", "language_hint", "max_input_s", "window_target_factor"}


class Qwen3AsrBackend(InferenceBackend):
    def __init__(self, config: ModelConfig) -> None:
        if config.backend != "qwen3-asr":
            raise ValueError(f"Qwen3AsrBackend cannot load backend {config.backend!r}")
        if config.revision is None:
            raise ValueError("Qwen3-ASR configs must pin model.revision to a full commit SHA")
        unknown = set(config.options) - _OPTIONS
        if unknown:
            raise ValueError(f"unsupported Qwen3-ASR options: {', '.join(sorted(unknown))}")
        self.config = config
        self._model = None

    @staticmethod
    def _version() -> str | None:
        try:
            return importlib.metadata.version("qwen-asr")
        except importlib.metadata.PackageNotFoundError:
            return None

    @property
    def identity(self) -> dict[str, object]:
        return {
            "backend": self.config.backend,
            "backend_version": self._version(),
            "provider": "Qwen",
            "model_id": self.config.model_id,
            "model_revision": self.config.revision,
            "model_loading_policy": "pinned-snapshot-v1",
            "provider_managed": False,
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "environment": list(self.config.environment),
            "transcribes": self.config.transcribes,
            "diarizes": self.config.diarizes,
            "options": self.config.options,
        }

    def preflight(self) -> dict[str, object]:
        try:
            import qwen_asr  # noqa: F401
            import torch
        except ImportError:
            return {
                "ok": False,
                "error": "Qwen3-ASR inference requires qwen-asr and torch",
                "identity": self.identity,
            }
        device_ok = not self.config.device.startswith("cuda") or torch.cuda.is_available()
        return {
            "ok": device_ok,
            "error": None if device_ok else "the configuration requires CUDA but torch found no CUDA device",
            "identity": self.identity,
        }

    def _load(self):
        if self._model is None:
            try:
                import torch
                from huggingface_hub import snapshot_download
                from qwen_asr import Qwen3ASRModel
            except ImportError as error:
                raise ValueError("Qwen3-ASR inference requires huggingface-hub, qwen-asr, and torch") from error
            model_path = os.environ.get("DAI_ASR_I18N_QWEN_MODEL_PATH")
            if model_path is None:
                model_path = snapshot_download(self.config.model_id, revision=self.config.revision)
            self._model = Qwen3ASRModel.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map=self.config.device,
                max_new_tokens=int(self.config.options.get("max_new_tokens", 2048)),
            )
        return self._model

    def _transcribe_one(self, audio_path: Path, language: str | None) -> Transcript:
        result = self._load().transcribe(audio=str(audio_path), language=None)[0]
        return Transcript(
            text=str(result.text).strip(),
            language=language,
            metadata={"detected_language": getattr(result, "language", None)},
        )

    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        maximum = float(self.config.options.get("max_input_s", 120.0))
        factor = float(self.config.options.get("window_target_factor", 0.98))
        with planned_audio_windows(audio_path, max_input_s=maximum, target_factor=factor) as windows:
            results = [(window, self._transcribe_one(window.path, language)) for window in windows]
        return results[0][1] if len(results) == 1 else stitch_transcripts(results, language=language)


__all__ = ["Qwen3AsrBackend"]
