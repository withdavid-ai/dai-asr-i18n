"""Shared, credential-safe support for hosted speech APIs."""

from __future__ import annotations

import importlib.metadata
import math
import os
import random
import time
from collections.abc import Callable
from typing import Any

from dai_asr_i18n.inference.base import InferenceBackend, TranscriptSegment, TranscriptWord
from dai_asr_i18n.inference.config import ModelConfig


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def require_httpx():
    try:
        import httpx
    except ImportError as error:
        raise ValueError('hosted inference requires: pip install "dai-asr-i18n[hosted]"') from error
    return httpx


def _retry_delay(response: Any | None, attempt: int) -> float:
    minimum = min(2**attempt, 30)
    retry_after = None if response is None else response.headers.get("retry-after")
    if retry_after is not None:
        try:
            requested = float(retry_after)
        except (TypeError, ValueError):
            requested = 0.0
        if math.isfinite(requested) and requested >= 0:
            minimum = max(minimum, min(requested, 120.0))
    return minimum + random.uniform(0.0, min(1.0, minimum * 0.25))


def checked_request(call: Callable[[], Any], *, attempts: int = 6):
    """Retry transient transport/server failures and disclose no response body."""

    httpx = require_httpx()
    last_error: Exception | None = None
    for attempt in range(attempts):
        response = None
        try:
            response = call()
        except httpx.TransportError as error:
            last_error = error
        else:
            if response.status_code < 400:
                return response
            request_id = response.headers.get("request-id") or response.headers.get("x-request-id")
            error = RuntimeError(
                f"hosted API returned HTTP {response.status_code}"
                + (f" (request_id={request_id})" if request_id else "")
            )
            if response.status_code not in {408, 409, 429} and response.status_code < 500:
                raise error
            last_error = error
        if attempt + 1 < attempts:
            time.sleep(_retry_delay(response, attempt))
    raise RuntimeError(f"hosted API failed after {attempts} attempts: {last_error}") from last_error


def segments_from_words(
    words: tuple[TranscriptWord, ...], *, require_speakers: bool = False
) -> tuple[TranscriptSegment, ...]:
    """Build contiguous speaker turns from word-level provider output."""

    if not words:
        return ()
    if require_speakers and any(word.text.strip() and word.speaker is None for word in words):
        raise ValueError("speaker-attributed transcript contains a word without a speaker label")
    groups: list[list[TranscriptWord]] = []
    for word in words:
        if not groups or groups[-1][-1].speaker != word.speaker:
            groups.append([word])
        else:
            groups[-1].append(word)
    return tuple(
        TranscriptSegment(
            text=" ".join(word.text.strip() for word in group if word.text.strip()),
            start_s=next((word.start_s for word in group if word.start_s is not None), None),
            end_s=next((word.end_s for word in reversed(group) if word.end_s is not None), None),
            words=tuple(group),
            speaker=group[0].speaker,
        )
        for group in groups
    )


class HostedBackend(InferenceBackend):
    """Base contract for provider-managed models invoked with local audio files."""

    provider: str

    def __init__(self, config: ModelConfig, *, credentials: dict[str, str] | None = None) -> None:
        self.config = config
        self._credentials = dict(credentials or {})

    @property
    def identity(self) -> dict[str, object]:
        return {
            "backend": self.config.backend,
            "backend_version": _distribution_version("httpx"),
            "provider": self.provider,
            "model_id": self.config.model_id,
            "model_revision": self.config.revision,
            "provider_managed": self.config.revision is None,
            "device": "remote",
            "compute_type": "provider-managed",
            "environment": list(self.config.environment),
            "transcribes": self.config.transcribes,
            "diarizes": self.config.diarizes,
            "options": self.config.options,
        }

    def credential(self, name: str) -> str:
        value = self._credentials.get(name) or os.environ.get(name)
        if not value:
            raise ValueError(f"missing required environment variable {name}")
        return value

    def preflight(self) -> dict[str, object]:
        missing = [
            name for name in self.config.environment if not (self._credentials.get(name) or os.environ.get(name))
        ]
        try:
            require_httpx()
        except ValueError as error:
            return {"ok": False, "error": str(error), "identity": self.identity}
        return {
            "ok": not missing,
            "error": f"missing required environment variables: {', '.join(missing)}" if missing else None,
            "identity": self.identity,
        }


__all__ = ["HostedBackend", "checked_request", "require_httpx", "segments_from_words"]
