"""The installed package exposes a minimal, stable CLI surface."""

from __future__ import annotations

import json

import pytest

from dai_asr_i18n import __version__
from dai_asr_i18n.cli import main, parser


def test_empty_command_prints_help(capsys):
    assert main([]) == 0
    assert "Reproducible multilingual ASR benchmarking" in capsys.readouterr().out


def test_version_flag_uses_distribution_version(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    assert f"dai-asr-i18n {__version__}" in capsys.readouterr().out


def test_models_command_lists_the_16_system_catalog(capsys):
    assert main(["models"]) == 0
    output = capsys.readouterr().out
    assert "MAI-Transcribe-2" in output
    assert "BUT-FIT/diarizen-wavlm-large-s80-md-v2" in output
    assert "provider_managed" in output
    assert len(output.strip().splitlines()) == 17


def test_models_command_emits_catalog_json(capsys):
    assert main(["models", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["catalog_version"] == "dai-asr-i18n-model-configs-v2"
    assert len(result["models"]) == 16


def test_models_command_resolves_one_config(capsys):
    assert main(["models", "whisper"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model_id"] == "large-v3"
    assert result["parameters"]["beam_size"] == 5


def test_models_command_rejects_ambiguous_json_arguments(capsys):
    with pytest.raises(SystemExit) as error:
        main(["models", "whisper", "--json"])
    assert error.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_normalize_command(capsys):
    assert main(["normalize", "學習語言", "--language", "zh"]) == 0
    assert capsys.readouterr().out.strip() == "学习语言"


def test_score_command_emits_counts(capsys):
    assert main(["score", "--language", "en", "--reference", "alpha beta", "--hypothesis", "alpha"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["metric"] == "wer"
    assert result["deletions"] == 1
    assert result["denominator"] == 2
    assert len(result["provenance"]["dai_asr_i18n"]["policy_implementation_sha256"]) == 64


def test_score_input_emits_rows_and_micro_aggregates(tmp_path, capsys):
    source = tmp_path / "pairs.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "language": "en", "reference": "alpha", "hypothesis": "wrong"}),
                json.dumps({"id": "b", "language": "en", "reference": "alpha beta", "hypothesis": "alpha beta"}),
            ]
        ),
        encoding="utf-8",
    )
    assert main(["score", "--input", str(source)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [row["input"]["id"] for row in result["rows"]] == ["a", "b"]
    assert result["aggregates"][0]["numerator"] == 1
    assert result["aggregates"][0]["denominator"] == 3
    assert result["provenance"]["resource_hashes_included"] is False


def test_score_input_namespaces_metadata_that_collides_with_score_fields(tmp_path, capsys):
    source = tmp_path / "pairs.jsonl"
    source.write_text(
        json.dumps(
            {
                "language": "en-US",
                "metric": "source-label",
                "reference": "alpha",
                "hypothesis": "alpha",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["score", "--input", str(source)]) == 0
    row = json.loads(capsys.readouterr().out)["rows"][0]
    assert row["input"]["language"] == "en-US"
    assert row["input"]["metric"] == "source-label"
    assert row["language"] == "en"
    assert row["metric"] == "wer"


def test_score_jsonl_remains_as_compatibility_alias(tmp_path, capsys):
    source = tmp_path / "pairs.jsonl"
    source.write_text(
        json.dumps({"language": "en", "reference": "alpha", "hypothesis": "alpha"}) + "\n",
        encoding="utf-8",
    )
    assert main(["score-jsonl", str(source)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["aggregates"][0]["value"] == 0


def test_score_input_rejects_an_empty_dataset(tmp_path, capsys):
    source = tmp_path / "empty.jsonl"
    source.write_text("\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["score", "--input", str(source)])
    assert error.value.code == 2
    assert "input contains no records" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"language":"en","reference":"a","hypothesis":"a","value":NaN}', "non-finite number"),
        ('{"language":"en","language":"zh","reference":"a","hypothesis":"a"}', "duplicate object key"),
    ],
)
def test_score_input_rejects_nonstandard_or_ambiguous_json(tmp_path, capsys, payload, message):
    source = tmp_path / "invalid.jsonl"
    source.write_text(payload + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["score", "--input", str(source)])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_score_pair_requires_hypothesis_and_language(capsys):
    with pytest.raises(SystemExit) as error:
        main(["score", "--reference", "alpha"])
    assert error.value.code == 2
    assert "--reference requires --hypothesis and --language" in capsys.readouterr().err


def test_score_input_rejects_pair_arguments(tmp_path, capsys):
    source = tmp_path / "pairs.jsonl"
    source.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["score", "--input", str(source), "--language", "en"])
    assert error.value.code == 2
    assert "cannot be used with --input" in capsys.readouterr().err


def test_score_pair_rejects_batch_grouping(capsys):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "score",
                "--reference",
                "alpha",
                "--hypothesis",
                "alpha",
                "--language",
                "en",
                "--group-by",
                "model",
            ]
        )
    assert error.value.code == 2
    assert "--group-by can be used only with --input" in capsys.readouterr().err


def test_score_reports_unsupported_metric_without_traceback(capsys):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "score",
                "--reference",
                "你好世界",
                "--hypothesis",
                "你好世间",
                "--language",
                "zh",
                "--metric",
                "wer",
            ]
        )
    assert error.value.code == 2
    assert "word segmenter" in capsys.readouterr().err


def test_dataset_pull_requires_a_hugging_face_repository(capsys):
    command = parser()
    with pytest.raises(SystemExit) as error:
        command.parse_args(
            [
                "dataset",
                "pull",
                "--revision",
                "1" * 40,
            ]
        )
    assert error.value.code == 2
    assert "--repo" in capsys.readouterr().err
