"""Strict TOML loading and validation for the built-in model catalog."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from functools import cache
from importlib import resources
from types import MappingProxyType
from typing import Any

from dai_asr_i18n.languages import BENCHMARK_LANGUAGES
from dai_asr_i18n.model_configs.models import MODEL_CATALOG_VERSION, ModelCatalog, ModelConfig, freeze

_KEY = re.compile(r"[a-z][a-z0-9_]*")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TASKS = frozenset({"asr", "asr_diarization", "diarization"})
_EXECUTIONS = frozenset({"hosted_api", "local"})
_REPRODUCIBILITY = frozenset({"mutable_checkpoint", "pinned", "provider_managed"})
_ACCESS = frozenset({"open", "proprietary"})
_LANGUAGE_POLICIES = frozenset({"attempt_all", "explicit", "language_independent", "unspecified"})
_CHANNELS = frozenset({"ch1", "ch2", "mono"})


def _string(raw: Mapping[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _strings(raw: Mapping[str, Any], field: str, *, nonempty: bool = True) -> tuple[str, ...]:
    value = raw.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must be an array of non-empty strings")
    if nonempty and not value:
        raise ValueError(f"{field} must not be empty")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(value)


def _load_toml(resource: Any) -> dict[str, Any]:
    try:
        value = tomllib.loads(resource.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid TOML in {resource.name}: {error}") from error
    if not isinstance(value, dict):  # pragma: no cover - tomllib always returns a dict
        raise ValueError(f"{resource.name} must contain a TOML table")
    return value


def _parse_model(raw: dict[str, Any], *, filename: str) -> ModelConfig:
    allowed = {
        "schema_version",
        "key",
        "display_name",
        "task",
        "execution",
        "access",
        "provider",
        "model_id",
        "model_revision",
        "reproducibility",
        "language_policy",
        "languages",
        "channels",
        "transcribes",
        "diarizes",
        "credentials",
        "parameters",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown fields in {filename}: {', '.join(sorted(unknown))}")
    if raw.get("schema_version") != 1:
        raise ValueError(f"{filename} must use schema_version = 1")
    key = _string(raw, "key")
    if not _KEY.fullmatch(key) or filename != f"{key}.toml":
        raise ValueError(f"model key {key!r} must be a lowercase slug matching {filename}")
    task = _string(raw, "task")
    execution = _string(raw, "execution")
    access = _string(raw, "access")
    reproducibility = _string(raw, "reproducibility")
    language_policy = _string(raw, "language_policy")
    if task not in _TASKS:
        raise ValueError(f"{key}: unsupported task {task!r}")
    if execution not in _EXECUTIONS:
        raise ValueError(f"{key}: unsupported execution mode {execution!r}")
    if access not in _ACCESS:
        raise ValueError(f"{key}: unsupported access value {access!r}")
    if reproducibility not in _REPRODUCIBILITY:
        raise ValueError(f"{key}: unsupported reproducibility value {reproducibility!r}")
    if language_policy not in _LANGUAGE_POLICIES:
        raise ValueError(f"{key}: unsupported language_policy {language_policy!r}")
    revision = raw.get("model_revision")
    if revision is not None and (not isinstance(revision, str) or not _COMMIT.fullmatch(revision)):
        raise ValueError(f"{key}: model_revision must be a full lowercase commit SHA")
    if reproducibility == "pinned" and revision is None:
        raise ValueError(f"{key}: pinned configs require model_revision")
    if reproducibility != "pinned" and revision is not None:
        raise ValueError(f"{key}: only pinned configs may set model_revision")
    channels = _strings(raw, "channels")
    if set(channels) - _CHANNELS:
        raise ValueError(f"{key}: channels must contain only ch1, ch2, and/or mono")
    transcribes = raw.get("transcribes")
    diarizes = raw.get("diarizes")
    if not isinstance(transcribes, bool) or not isinstance(diarizes, bool):
        raise ValueError(f"{key}: transcribes and diarizes must be booleans")
    expected = {
        "asr": (True, False),
        "asr_diarization": (True, True),
        "diarization": (False, True),
    }[task]
    if (transcribes, diarizes) != expected:
        raise ValueError(f"{key}: task {task!r} conflicts with transcribes/diarizes")
    if task == "diarization" and channels != ("mono",):
        raise ValueError(f"{key}: diarization-only systems must run on mono")
    languages = _strings(raw, "languages", nonempty=language_policy != "unspecified")
    if language_policy == "unspecified" and languages:
        raise ValueError(f"{key}: unspecified language policy requires an empty languages array")
    if any(not re.fullmatch(r"[a-z]{2}", language) for language in languages):
        raise ValueError(f"{key}: languages must use two-letter benchmark codes")
    if set(languages) - set(BENCHMARK_LANGUAGES):
        raise ValueError(f"{key}: languages must be drawn from the public benchmark roster")
    if language_policy in {"attempt_all", "language_independent"} and set(languages) != set(BENCHMARK_LANGUAGES):
        raise ValueError(f"{key}: {language_policy} must cover every public benchmark language")
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"{key}: parameters must be a TOML table")
    credentials = _strings(raw, "credentials", nonempty=False)
    if any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) for name in credentials):
        raise ValueError(f"{key}: credentials must contain environment-variable names only")
    return ModelConfig(
        schema_version=1,
        key=key,
        display_name=_string(raw, "display_name"),
        task=task,
        execution=execution,
        access=access,
        provider=_string(raw, "provider"),
        model_id=_string(raw, "model_id"),
        model_revision=revision,
        reproducibility=reproducibility,
        language_policy=language_policy,
        languages=languages,
        channels=channels,
        transcribes=transcribes,
        diarizes=diarizes,
        credentials=credentials,
        parameters=freeze(parameters),
    )


@cache
def model_catalog() -> ModelCatalog:
    """Load and validate the immutable built-in 16-model catalog."""

    root = resources.files("dai_asr_i18n.model_configs")
    raw = _load_toml(root.joinpath("catalog.toml"))
    allowed = {
        "schema_version",
        "catalog_version",
        "model_keys",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown catalog fields: {', '.join(sorted(unknown))}")
    if raw.get("schema_version") != 1:
        raise ValueError("catalog.toml must use schema_version = 1")
    version = _string(raw, "catalog_version")
    if version != MODEL_CATALOG_VERSION:
        raise ValueError(f"unsupported model catalog version {version!r}")
    model_keys = _strings(raw, "model_keys")
    if len(model_keys) != 16:
        raise ValueError(f"{MODEL_CATALOG_VERSION} must contain exactly 16 model families")
    parsed = {key: _parse_model(_load_toml(root.joinpath(f"{key}.toml")), filename=f"{key}.toml") for key in model_keys}
    shipped = {
        item.name.removesuffix(".toml")
        for item in root.iterdir()
        if item.name != "catalog.toml" and item.name.endswith(".toml")
    }
    if shipped != set(model_keys):
        raise ValueError("catalog model_keys do not match the packaged model TOML files")
    return ModelCatalog(
        schema_version=1,
        catalog_version=version,
        model_keys=model_keys,
        models=MappingProxyType(parsed),
    )


def list_model_configs() -> tuple[ModelConfig, ...]:
    """Return model configs in the paper roster's stable display order."""

    catalog = model_catalog()
    return tuple(catalog.models[key] for key in catalog.model_keys)


def model_config(key: str) -> ModelConfig:
    """Resolve a model config by its stable subject key."""

    if not isinstance(key, str):
        raise TypeError("model config key must be a string")
    try:
        return model_catalog().models[key]
    except KeyError as error:
        raise ValueError(f"unknown model config {key!r}") from error


__all__ = ["list_model_configs", "model_catalog", "model_config"]
