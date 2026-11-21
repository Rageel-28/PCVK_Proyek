"""Tes pencocokan baris ke templat teks tetap."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.templates import Templates, snap_to_template

TEMPLATES = [
    "SERBA ADA SERBA MUDAH",
    "TERIMA KASIH ATAS KUNJUNGAN ANDA",
    "BARANG YANG SUDAH DIBELI",
    "TIDAK DAPAT DITUKAR KEMBALI",
    "JL. RAYA KNO 12 KEL. SUKAPURA",
    "TOTAL 47.200",
]


def test_snaps_mangled_boilerplate_line() -> None:
    assert (
        snap_to_template("S--XRBA ADA SERBA MUDAH", TEMPLATES)
        == "SERBA ADA SERBA MUDAH"
    )


def test_snaps_when_only_one_character_is_wrong() -> None:
    assert (
        snap_to_template("BARANG YANG SUDAH DIBELJ", TEMPLATES)
        == "BARANG YANG SUDAH DIBELI"
    )


def test_leaves_unrelated_line_alone() -> None:
    text = "MINYAK GORENG SANIA 1L 19.800"
    assert snap_to_template(text, TEMPLATES) == text


def test_refuses_to_snap_when_digits_differ() -> None:
    # Angka berbeda = baris berbeda. Ini yang melindungi harga.
    text = "TOTAL 48.200"
    assert snap_to_template(text, TEMPLATES) == text


def test_snaps_price_line_when_digits_match() -> None:
    assert snap_to_template("TOTAI 47.200", TEMPLATES) == "TOTAL 47.200"


def test_ignores_short_lines() -> None:
    assert snap_to_template("ABC", ["ABD"]) == "ABC"


def test_handles_empty_templates() -> None:
    assert snap_to_template("SERBA ADA SERBA MUDAH", []) == "SERBA ADA SERBA MUDAH"


def test_snaps_digit_free_template_even_if_letter_read_as_digit() -> None:
    # Huruf yang salah baca sering muncul sebagai digit; templat tanpa angka
    # tidak punya isi angka yang bisa dirusak, jadi tetap boleh dicocokkan.
    assert (
        snap_to_template("SERBA ADA SERBA APJUDAH", ["SERBA ADA SERBA MUDAH"])
        == "SERBA ADA SERBA MUDAH"
    )


def test_templates_roundtrip(tmp_path: Path) -> None:
    store = Templates(TEMPLATES + ["SERBA ADA SERBA MUDAH"])
    path = store.save(tmp_path / "templates.json")
    loaded = Templates.load(path)
    assert len(loaded) == len(store)
    assert loaded.snap("S--XRBA ADA SERBA MUDAH") == "SERBA ADA SERBA MUDAH"


def test_templates_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        snap_to_template(None, TEMPLATES)  # type: ignore[arg-type]


def test_snaps_address_whose_digit_is_really_a_letter() -> None:
    """``DIPONEGOR0`` bukan data angka; alamat tetap boleh diperbaiki."""
    templates = ["JL. DIPONEGORO 88 KEL. DAMAI"]
    assert (
        snap_to_template("JL. DIPONEGOR0 88 KEL. DAMAI", templates)
        == "JL. DIPONEGORO 88 KEL. DAMAI"
    )


def test_number_signature_ignores_digits_inside_words() -> None:
    from ocr.templates import _number_signature

    assert _number_signature("STORE NO. 0005 021-5777148") == "0005021-5777148"
    assert _number_signature("JL. DIPONEGOR0 88 KEL. DAMAI") == "88"


def test_snaps_store_line_with_misread_digit() -> None:
    templates = ["STORE NO. 0005 021-5777148"]
    assert (
        snap_to_template("STORE NO. 000S 021-5777148", templates)
        == "STORE NO. 0005 021-5777148"
    )
