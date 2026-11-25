"""The transliteration maps: their fingerprint, file format, lookup and folding."""

from __future__ import annotations

import csv

import pytest

from dai_asr_i18n.normalization import transliteration_maps as maps
from dai_asr_i18n.normalization.tokens import tokenize
from dai_asr_i18n.normalization.transliteration import tokenization


def test_committed_maps_match_their_fingerprint():
    assert maps.maps_digest() == maps.MAPS_SHA256
    assert sorted(path.stem for path in maps.MAPS_DIR.glob("*.tsv")) == sorted(maps.LANGUAGES)


@pytest.mark.parametrize("language", sorted(maps.LANGUAGES))
def test_map_files_hold_only_the_columns_scoring_reads(language):
    with maps.map_path(language).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    assert reader.fieldnames == ["native", "target", "decision", "occurrences", "preferred"]
    assert {row["decision"] for row in rows} <= {"map", "native_block", "preferred"}
    assert {row["preferred"] for row in rows} <= {"yes", "no"}
    assert all(row["preferred"] == "yes" for row in rows if row["decision"] == "preferred")
    assert all(row["occurrences"].isdigit() for row in rows if row["decision"] == "map")
    assert maps.load_map(language).exact


def test_preferred_rows_do_not_change_lookup():
    loaded = maps.load_map("hi")
    assert "यूट्यूब" in maps.preferred_spellings("hi")
    preferred_only = {
        row["native"]
        for row in csv.DictReader(maps.map_path("hi").open(encoding="utf-8", newline=""), delimiter="\t")
        if row["decision"] == "preferred"
    }
    mapped, _ = maps.read_map_file(maps.map_path("hi"))
    assert all(row["decision"] == "map" for row in mapped)
    assert {native for native in preferred_only if native in loaded.exact} <= {row["native"] for row in mapped}


@pytest.mark.parametrize(
    "language,left,right",
    [
        ("hi", "कम्प्यूटर", "कंप्यूटर"),
        ("hi", "फ़ोन", "फोन"),
        ("ar", "إيميل", "ايميل"),
        ("ru", "ёлка", "елка"),
        ("ja", "コンピュータ", "コンピューター"),
    ],
)
def test_spelling_variants_share_a_key(language, left, right):
    assert maps.variant_key(language, left) == maps.variant_key(language, right)


def test_a_class_folds_native_spellings_and_its_latin_form_to_one_representative():
    computer = maps.fold("hi", "computer")
    assert computer != "computer"
    assert maps.fold("hi", "कंप्यूटर") == computer
    assert maps.fold("hi", "कम्प्यूटर") == computer


def test_text_that_matches_no_class_is_unchanged():
    assert maps.fold("hi", "नमस्ते दोस्त") == "नमस्ते दोस्त"
    assert maps.fold("es", "computer") == "computer"


def test_classes_are_named_by_the_base_normalizer():
    assert maps.latin_class("hi", "It's") == "it s"
    assert maps.latin_class("ja", "Wi-Fi") == "wi fi"


def test_japanese_is_read_by_character_and_other_languages_by_word():
    assert tokenization("ja") == "cjk_character" and tokenization("hi") == "word"
    assert tokenize("ラインstamp", "ja", "cjk_character") == ["ラ", "イ", "ン", "stamp"]
    assert tokenize("मेरा फ़ोन", "hi", "word") == ["मेरा", "फ़ोन"]
