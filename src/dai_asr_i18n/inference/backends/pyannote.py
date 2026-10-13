"""pyannoteAI batch diarization from a local audio file."""

from __future__ import annotations

import math
import re
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dai_asr_i18n.inference.backends.hosted import HostedBackend, checked_request, require_httpx
from dai_asr_i18n.inference.base import Transcript, TranscriptSegment
from dai_asr_i18n.inference.config import ModelConfig

DIARIZE_ENDPOINT = "https://api.pyannote.ai/v1/diarize"
MEDIA_ENDPOINT = "https://api.pyannote.ai/v1/media/input"
JOBS_ENDPOINT = "https://api.pyannote.ai/v1/jobs"

_API_TIMEOUT_S = 30
_UPLOAD_TIMEOUT_S = 300
_JOB_TIMEOUT_S = 30 * 60.0
_POLL_INTERVAL_S = 5.0
_JOB_ID = re.compile(r"[A-Za-z0-9-]{1,128}")
_ACTIVE_JOB_STATUSES = {"created", "pending", "running"}
_FAILED_JOB_STATUSES = {"failed", "canceled"}
_BOOLEAN_OPTIONS = {
    "turnLevelConfidence",
    "speechProbability",
    "crosstalkProbability",
    "speakerProbability",
    "exclusive",
    "transcription",
}
_SENSITIVITY_OPTIONS = {"vadSensitivity", "crosstalkSensitivity"}
_SUPPORTED_OPTIONS = {"api_endpoint", *_BOOLEAN_OPTIONS, *_SENSITIVITY_OPTIONS}


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _response_object(response: object, description: str) -> dict[str, object]:
    try:
        value = response.json()  # type: ignore[attr-defined]
    except (TypeError, ValueError) as error:
        raise ValueError(f"pyannote {description} response is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"pyannote {description} response must be an object")
    return value


def _audio_stream(path: Path) -> Iterator[bytes]:
    with path.open("rb") as audio:
        while chunk := audio.read(1024 * 1024):
            yield chunk


def _validate_config(config: ModelConfig) -> None:
    if config.backend != "pyannote-api":
        raise ValueError(f"PyannoteApiBackend cannot load backend {config.backend!r}")
    if config.model_id != "precision-3":
        raise ValueError("pyannote-api currently supports only precision-3")
    if "PYANNOTE_API_KEY" not in config.environment:
        raise ValueError("pyannote-api configs must declare PYANNOTE_API_KEY")
    if config.transcribes or not config.diarizes:
        raise ValueError("pyannote-api currently supports diarization-only configs")

    unknown = set(config.options) - _SUPPORTED_OPTIONS
    if unknown:
        raise ValueError(f"unsupported pyannote options: {', '.join(sorted(unknown))}")
    if config.options.get("api_endpoint") != DIARIZE_ENDPOINT:
        raise ValueError(f"pyannote api_endpoint must be {DIARIZE_ENDPOINT}")
    if config.options.get("transcription") is not False:
        raise ValueError("pyannote diarization-only configs must set transcription = false")

    for name in _BOOLEAN_OPTIONS:
        if name in config.options and not isinstance(config.options[name], bool):
            raise ValueError(f"pyannote {name} must be true or false")
    for name in _SENSITIVITY_OPTIONS:
        value = config.options.get(name)
        if value is not None and (not _is_finite_number(value) or not -5.0 <= value <= 5.0):
            raise ValueError(f"pyannote {name} must be a finite number between -5 and 5")


def _request_options(config: ModelConfig) -> dict[str, object]:
    return {name: value for name, value in config.options.items() if name != "api_endpoint"}


@dataclass(frozen=True)
class _PyannoteClient:
    http: Any
    api_key: str = field(repr=False)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        url: str,
        description: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {"headers": self._headers, "timeout": _API_TIMEOUT_S}
        if payload is not None:
            kwargs["json"] = payload
        response = checked_request(lambda: self.http.request(method, url, **kwargs))
        return _response_object(response, description)

    def upload(self, audio_path: Path) -> str:
        media_url = f"media://dai-asr-i18n/{uuid.uuid4().hex}"
        media = self._request("POST", MEDIA_ENDPOINT, "media", payload={"url": media_url})
        upload_url = media.get("url")
        if not isinstance(upload_url, str):
            raise ValueError("pyannote media response lacks an upload URL")

        parsed = urlsplit(upload_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("pyannote media response returned an unsafe upload URL")

        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(audio_path.stat().st_size),
        }
        try:
            checked_request(
                lambda: self.http.put(
                    upload_url,
                    headers=headers,
                    content=_audio_stream(audio_path),
                    timeout=_UPLOAD_TIMEOUT_S,
                )
            )
        except RuntimeError as error:
            raise RuntimeError("pyannote media upload failed") from error
        return media_url

    def submit(self, media_url: str, model_id: str, options: dict[str, object]) -> str:
        job = self._request(
            "POST",
            DIARIZE_ENDPOINT,
            "job submission",
            payload={"url": media_url, "model": model_id, **options},
        )
        job_id = job.get("jobId")
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
            raise ValueError("pyannote job submission response lacks a valid jobId")
        return job_id

    def wait(self, job_id: str) -> dict[str, object]:
        deadline = time.monotonic() + _JOB_TIMEOUT_S
        while True:
            job = self._request("GET", f"{JOBS_ENDPOINT}/{job_id}", "job")
            status = job.get("status")
            if status == "succeeded":
                return job
            if status in _FAILED_JOB_STATUSES:
                raise RuntimeError(f"pyannote diarization job {status}")
            if status not in _ACTIVE_JOB_STATUSES:
                raise ValueError("pyannote job response has an invalid status")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("pyannote diarization job timed out")
            time.sleep(min(_POLL_INTERVAL_S, remaining))


def _parse_segments(job: dict[str, object]) -> tuple[TranscriptSegment, ...]:
    output = job.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("diarization"), list):
        raise ValueError("pyannote job response lacks diarization output")

    segments = []
    for index, value in enumerate(output["diarization"]):
        if not isinstance(value, dict):
            raise ValueError(f"pyannote diarization segment {index} must be an object")
        speaker = value.get("speaker")
        start = value.get("start")
        end = value.get("end")
        if not isinstance(speaker, str) or not speaker:
            raise ValueError(f"pyannote diarization segment {index} lacks a speaker")
        if not _is_finite_number(start) or not _is_finite_number(end) or start < 0 or end < start:
            raise ValueError(f"pyannote diarization segment {index} has invalid timing")
        segments.append(
            TranscriptSegment(
                text="",
                start_s=float(start),
                end_s=float(end),
                speaker=speaker,
            )
        )

    return tuple(sorted(segments, key=lambda segment: (segment.start_s, segment.end_s, segment.speaker)))


class PyannoteApiBackend(HostedBackend):
    provider = "pyannoteAI"

    def __init__(self, config: ModelConfig, *, credentials: dict[str, str] | None = None) -> None:
        super().__init__(config, credentials=credentials)
        _validate_config(config)

    def transcribe(self, audio_path: Path, *, language: str | None) -> Transcript:
        del language  # Diarization is language-independent.
        client = _PyannoteClient(require_httpx(), self.credential("PYANNOTE_API_KEY"))
        options = _request_options(self.config)
        media_url = client.upload(audio_path)
        job_id = client.submit(media_url, self.config.model_id, options)
        segments = _parse_segments(client.wait(job_id))

        return Transcript(
            text="",
            duration_s=max((segment.end_s or 0.0 for segment in segments), default=0.0),
            segments=segments,
            metadata={
                "job_id": job_id,
                "request": {
                    "endpoint": DIARIZE_ENDPOINT,
                    "model": self.config.model_id,
                    "parameters": options,
                    "input": "temporary-media",
                },
            },
        )


__all__ = ["PyannoteApiBackend"]
