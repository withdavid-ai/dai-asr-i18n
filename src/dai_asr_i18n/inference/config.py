"""Strict TOML configuration for local and hosted inference runs."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from dai_asr_i18n.audio import MONO_SAMPLE_RATE

_COMMIT = re.compile(r"[0-9a-f]{40}")
_ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
_SENSITIVE_OPTION = re.compile(r"(?i)(api[_-]?key|authorization|credential|password|secret|token)")


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    model_id: str
    revision: str | None
    device: str
    compute_type: str
    authenticated: bool
    environment: tuple[str, ...]
    transcribes: bool
    diarizes: bool
    options: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "compute_type": self.compute_type,
            "authenticated": self.authenticated,
            "environment": list(self.environment),
            "transcribes": self.transcribes,
            "diarizes": self.diarizes,
            "options": self.options,
        }


@dataclass(frozen=True)
class RunConfig:
    schema_version: int
    name: str
    channels: tuple[str, ...]
    sample_rate_hz: int
    model: ModelConfig
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "channels": list(self.channels),
            "sample_rate_hz": self.sample_rate_hz,
            "model": self.model.to_dict(),
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()


def _read_source(value: str | Path) -> tuple[str, bytes]:
    path = Path(value).expanduser()
    if path.is_file():
        # Manifests may be published; retain the file identity without leaking a workstation path.
        return f"file:{path.name}", path.read_bytes()
    name = str(value)
    if not name.endswith(".toml"):
        name = f"{name}.toml"
    candidate = resources.files("dai_asr_i18n.run_configs").joinpath(name)
    if not candidate.is_file():
        raise ValueError(f"run config {value!r} was not found as a path or built-in name")
    return f"builtin:{name}", candidate.read_bytes()


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def load_run_config(value: str | Path) -> RunConfig:
    source, content = _read_source(value)
    try:
        raw = tomllib.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid TOML in {source}: {error}") from error
    if set(raw) - {"schema_version", "name", "channels", "sample_rate_hz", "model"}:
        raise ValueError(f"unknown top-level fields in {source}")
    if raw.get("schema_version") != 1:
        raise ValueError(f"{source} must use schema_version = 1")
    channels = raw.get("channels")
    if (
        not isinstance(channels, list)
        or not channels
        or any(channel not in {"ch1", "ch2", "mono"} for channel in channels)
    ):
        raise ValueError("channels must be a non-empty array containing ch1, ch2, and/or mono")
    if len(channels) != len(set(channels)):
        raise ValueError("channels must not contain duplicates")
    sample_rate = raw.get("sample_rate_hz")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate < 1:
        raise ValueError("sample_rate_hz must be a positive integer")
    if "mono" in channels and sample_rate != MONO_SAMPLE_RATE:
        raise ValueError(f"mono runs must use the canonical {MONO_SAMPLE_RATE} Hz sample rate")
    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be a table")
    known_model = {
        "backend",
        "model_id",
        "revision",
        "device",
        "compute_type",
        "authenticated",
        "environment",
        "transcribes",
        "diarizes",
        "options",
    }
    if set(model) - known_model:
        raise ValueError(f"unknown model fields in {source}: {', '.join(sorted(set(model) - known_model))}")
    authenticated = model.get("authenticated", False)
    if not isinstance(authenticated, bool):
        raise ValueError("model.authenticated must be true or false")
    options = model.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("model.options must be a table")
    sensitive_options = [key for key in options if key != "max_new_tokens" and _SENSITIVE_OPTION.search(key)]
    if sensitive_options:
        raise ValueError(
            "model.options must not contain credentials; declare environment-variable names instead: "
            + ", ".join(sorted(sensitive_options))
        )
    raw_environment = model.get("environment", [])
    if (
        not isinstance(raw_environment, list)
        or any(not isinstance(name, str) or not _ENVIRONMENT_NAME.fullmatch(name) for name in raw_environment)
        or len(raw_environment) != len(set(raw_environment))
    ):
        raise ValueError("model.environment must contain unique environment-variable names")
    transcribes = model.get("transcribes", True)
    diarizes = model.get("diarizes", False)
    if not isinstance(transcribes, bool) or not isinstance(diarizes, bool) or not (transcribes or diarizes):
        raise ValueError("model.transcribes and model.diarizes must describe at least one task")
    raw_revision = model.get("revision")
    revision = None if raw_revision is None else _string(raw_revision, "model.revision").lower()
    if revision is not None and not _COMMIT.fullmatch(revision):
        raise ValueError("model.revision must be a full 40-character commit SHA when provided")
    backend = _string(model.get("backend"), "model.backend")
    if backend == "faster-whisper" and revision is None:
        raise ValueError("faster-whisper configs must pin model.revision to a full commit SHA")
    return RunConfig(
        schema_version=1,
        name=_string(raw.get("name"), "name"),
        channels=tuple(channels),
        sample_rate_hz=sample_rate,
        model=ModelConfig(
            backend=backend,
            model_id=_string(model.get("model_id"), "model.model_id"),
            revision=revision,
            device=_string(model.get("device", "remote"), "model.device"),
            compute_type=_string(model.get("compute_type", "provider-managed"), "model.compute_type"),
            authenticated=authenticated,
            environment=tuple(raw_environment),
            transcribes=transcribes,
            diarizes=diarizes,
            options=options,
        ),
        source=source,
    )
