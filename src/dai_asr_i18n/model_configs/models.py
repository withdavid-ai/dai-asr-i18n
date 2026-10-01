"""Immutable public data models for benchmark invocation configurations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

MODEL_CATALOG_VERSION = "dai-asr-i18n-model-configs-v2"


def freeze(value: Any) -> Any:
    """Recursively make TOML collections read-only."""

    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """One selected model family and its explicit benchmark invocation contract.

    These records describe the audited operating points. A config is not a promise that the
    standalone package already implements the corresponding provider or local runtime.
    """

    schema_version: int
    key: str
    display_name: str
    task: str
    execution: str
    access: str
    provider: str
    model_id: str
    model_revision: str | None
    reproducibility: str
    language_policy: str
    languages: tuple[str, ...]
    channels: tuple[str, ...]
    transcribes: bool
    diarizes: bool
    credentials: tuple[str, ...]
    parameters: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "key": self.key,
            "display_name": self.display_name,
            "task": self.task,
            "execution": self.execution,
            "access": self.access,
            "provider": self.provider,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "reproducibility": self.reproducibility,
            "language_policy": self.language_policy,
            "languages": list(self.languages),
            "channels": list(self.channels),
            "transcribes": self.transcribes,
            "diarizes": self.diarizes,
            "credentials": list(self.credentials),
            "parameters": _thaw(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """The roster and provenance envelope for packaged model configs."""

    schema_version: int
    catalog_version: str
    model_keys: tuple[str, ...]
    models: Mapping[str, ModelConfig]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "catalog_version": self.catalog_version,
            "model_keys": list(self.model_keys),
            "models": [self.models[key].to_dict() for key in self.model_keys],
        }


__all__ = ["MODEL_CATALOG_VERSION", "ModelCatalog", "ModelConfig"]
