"""The packaged model catalog is strict, complete, and safe to inspect offline."""

from __future__ import annotations

import json
from importlib import resources

import pytest

from dai_asr_i18n import list_model_configs, model_catalog, model_config

EXPECTED_KEYS = (
    "mai_transcribe",
    "gemini_transcribe",
    "meta_muse",
    "xai_stt",
    "openai_diarize",
    "openai_transcribe",
    "voxtral_transcribe",
    "voxtral_small",
    "whisper",
    "qwen_asr",
    "nemotron35_asr",
    "parakeet",
    "nemotron3_diarize",
    "sortformer4v21_diarize",
    "pyannote_diarize",
    "diarizen",
)


def test_catalog_matches_the_16_system_public_roster():
    catalog = model_catalog()
    assert catalog.catalog_version == "dai-asr-i18n-model-configs-v2"
    assert catalog.model_keys == EXPECTED_KEYS
    assert len(list_model_configs()) == 16
    assert {config.access for config in list_model_configs()} == {"open", "proprietary"}
    assert sum(config.access == "proprietary" for config in list_model_configs()) == 7


def test_catalog_has_no_private_implementation_provenance():
    rendered = json.dumps(model_catalog().to_dict())
    assert "source_paths" not in rendered
    assert "source_repository" not in rendered
    assert "invocation_id" not in rendered


def test_capabilities_match_task_and_channel_contracts():
    for config in list_model_configs():
        if config.task == "asr":
            assert config.transcribes and not config.diarizes
        elif config.task == "asr_diarization":
            assert config.transcribes and config.diarizes
        elif config.task == "diarization":
            assert not config.transcribes and config.diarizes
            assert config.channels == ("mono",)
        else:  # pragma: no cover - the strict loader rejects this first
            raise AssertionError(config.task)


def test_language_eligibility_preserves_known_roster_limits():
    assert model_config("parakeet").languages == ("en", "es", "fr", "de", "it", "pt", "ru")
    assert "ta" not in model_config("gemini_transcribe").languages
    assert "ru" not in model_config("meta_muse").languages
    assert set(model_config("qwen_asr").languages).isdisjoint({"bn", "mr", "te", "ta"})
    assert model_config("voxtral_small").language_policy == "unspecified"
    assert model_config("voxtral_small").languages == ()
    assert model_config("diarizen").language_policy == "language_independent"


def test_explicit_code_parameters_override_appendix_shorthand():
    assert model_config("whisper").parameters["beam_size"] == 5
    assert model_config("mai_transcribe").parameters["transcribe_style"] == "verbatim"
    assert model_config("gemini_transcribe").parameters["diarization_mode"] == "speaker"
    assert model_config("voxtral_transcribe").parameters["temperature"] == 0.0


def test_diarizen_public_checkpoints_do_not_claim_a_gated_token_requirement():
    config = model_config("diarizen")
    assert config.credentials == ()
    assert config.parameters["hugging_face_gated"] is False


def test_released_nemotron_diarization_config_is_open_and_pinned():
    config = model_config("nemotron3_diarize")
    assert config.model_id == "nvidia/Nemotron-3-Diarization"
    assert config.model_revision == "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
    assert config.reproducibility == "pinned"
    assert config.credentials == ()
    assert config.parameters["license"] == "OpenMDW-1.1"


def test_configs_and_nested_parameters_are_read_only():
    config = model_config("sortformer4v21_diarize")
    with pytest.raises(TypeError):
        model_catalog().models["new"] = config
    with pytest.raises(TypeError):
        config.parameters["precision"] = "float16"
    with pytest.raises(TypeError):
        config.parameters["streaming"]["chunk_len"] = 1


def test_config_serialization_is_plain_json_data():
    payload = model_catalog().to_dict()
    assert len(payload["models"]) == 16
    assert json.loads(json.dumps(payload))["models"][0]["key"] == "mai_transcribe"


def test_model_lookup_rejects_unknown_and_non_string_keys():
    with pytest.raises(ValueError, match="unknown model config"):
        model_config("missing")
    with pytest.raises(TypeError, match="must be a string"):
        model_config(1)  # type: ignore[arg-type]


def test_wheel_resource_set_has_exactly_one_toml_per_model():
    root = resources.files("dai_asr_i18n.model_configs")
    model_files = {item.name.removesuffix(".toml") for item in root.iterdir() if item.name.endswith(".toml")}
    assert model_files == {*EXPECTED_KEYS, "catalog"}


def test_configs_contain_only_credential_names_not_values():
    for config in list_model_configs():
        assert all(name == name.upper() and name.replace("_", "").isalnum() for name in config.credentials)
        rendered = json.dumps(config.to_dict())
        assert "sk-" not in rendered
        assert "api_key=" not in rendered.lower()
