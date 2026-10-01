"""Runtime and resource provenance for normalization runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import unicodedata
from pathlib import Path

from dai_asr_i18n.languages import PUBLIC_METRIC
from dai_asr_i18n.normalization.profiles import DAI_ASR_I18N_NORMALIZERS, POLICY_ID, WHISPER_BASELINE_NORMALIZERS
from dai_asr_i18n.policies import (
    DER_POLICY_VERSION,
    JER_POLICY_VERSION,
    METRIC_POLICY_VERSION,
    SCORING_POLICY_VERSION,
    SPEAKER_COUNT_POLICY_VERSION,
)


def _package_version() -> str:
    try:
        return importlib.metadata.version("dai-asr-i18n")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+uninstalled"


def _policy_implementation_sha256() -> str:
    package_root = Path(__file__).parents[1]
    policy_files = [package_root / "languages.py", package_root / "policies.py"]
    policy_files.extend((package_root / "normalization").glob("*.py"))
    policy_files.extend((package_root / "scoring").glob("*.py"))
    policy_files.extend((package_root / "evaluation").glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(policy_files):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def normalization_manifest(*, include_opencc_hashes: bool = True) -> dict[str, object]:
    """Return runtime, dependency, normalization, and scoring provenance."""

    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "opencc",
            "indic-nlp-library",
            "whisper-normalizer",
            "more-itertools",
            "regex",
            "jiwer",
            "rapidfuzz",
        )
    }
    resource_hashes: dict[str, str] = {}
    if include_opencc_hashes:
        import opencc

        data = Path(opencc.__file__).parent / "clib/share/opencc"
        resource_hashes = {
            str(path.relative_to(data)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(data.rglob("*"))
            if path.is_file()
        }
    return {
        "dai_asr_i18n": {
            "version": _package_version(),
            "policy_implementation_sha256": _policy_implementation_sha256(),
        },
        "policy": POLICY_ID,
        "scoring_policy": SCORING_POLICY_VERSION,
        "metric_policy": {
            "version": METRIC_POLICY_VERSION,
            "public_metric": dict(sorted(PUBLIC_METRIC.items())),
        },
        "diarization_policy": {
            "der": DER_POLICY_VERSION,
            "jer": JER_POLICY_VERSION,
            "speaker_count": SPEAKER_COUNT_POLICY_VERSION,
        },
        "input_preparation": {
            "reference_merge_gap_s": 0.2,
            "gemini_timing": "gemini-invalid-words-missing-v1",
            "reference_text": "transcript text from the supplied reference",
            "speaker_count_reference": "speaker identities in the supplied reference",
        },
        "python": platform.python_version(),
        "unicode_version": unicodedata.unidata_version,
        "packages": packages,
        "dai_asr_i18n_normalizers": dict(sorted(DAI_ASR_I18N_NORMALIZERS.items())),
        "whisper_baseline_normalizers": dict(sorted(WHISPER_BASELINE_NORMALIZERS.items())),
        "opencc_config": "t2s.json",
        "resource_hashes_included": include_opencc_hashes,
        "opencc_resource_sha256": resource_hashes,
        "cer_unit": "non-whitespace Unicode code point",
        "wer_unit": "whitespace-delimited word; unavailable for zh, ja, and th",
        "ser_unit": "whitespace-delimited orthographic syllable; Vietnamese only",
        "formatting_escape_allowlist": ["U+00A0", "U+200C", "U+200D", "U+3000"],
        "parentheses": "strip parenthesized content, including nested and Unicode variants",
        "cjk_spacing": "ja/zh: remove spaces only between Han/kana characters",
    }


__all__ = ["normalization_manifest"]
