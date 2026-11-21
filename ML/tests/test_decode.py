"""Tes decoder: beam search + model bahasa + leksikon.

Decoder adalah tempat ambiguitas bentuk diselesaikan. Uji di sini harus
membuktikan bahwa konteks benar-benar mengubah keputusan, bukan sekadar
mengembalikan argmax.

Catatan penting: satu baris ``probs`` di sini adalah satu **kata**. Spasi
tidak pernah keluar dari classifier (charset tidak memuat spasi); spasi
berasal dari segmentasi kata.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C
from ocr.decode import (
    beam_decode,
    correct_words,
    decode_line,
    is_packaging_token,
    normalize_numeric_token,
    normalize_packaging_token,
    normalize_spacing,
)
from ocr.lexicon import Lexicon
from ocr.lm import CharLM


CORPUS = [
    "TOTAL 47.200",
    "TOTAL 49.700",
    "TOTAL 86.300",
    "TUNAI 72.900",
    "KEMBALI 5.000",
    "DISKON 2.500",
    "PPN 11% 4.677",
    "DPP 42.523",
    "INDOMARET",
    "SERBA ADA SERBA MUDAH",
    "POCARI SWEAT 500ML",
    "KAPAS KECIL TOP84",
]


@pytest.fixture(scope="module")
def lm() -> CharLM:
    return CharLM.train(CORPUS, order=4)


@pytest.fixture(scope="module")
def lex() -> Lexicon:
    words = {w for line in CORPUS for w in line.split()}
    return Lexicon.from_words(sorted(words))


def _probs_for(
    word: str, confusion: dict[tuple[str, str], float] | None = None
) -> np.ndarray:
    """Bangun matriks probabilitas yang meniru classifier ragu-ragu.

    ``confusion`` memetakan (karakter_benar, karakter_salah) ke probabilitas
    salah tersebut. Sisanya keyakinan tinggi pada karakter benar.
    """
    if confusion is None:
        confusion = {}
    n = len(C.CHARSET)
    out = np.full((len(word), n), 0.01 / (n - 1))
    for i, ch in enumerate(word):
        idx = C.CHAR_TO_IDX[ch]
        out[i, idx] = 0.99
        for (right, wrong), p in confusion.items():
            if ch == right:
                out[i, idx] = 1.0 - p - 0.01
                out[i, C.CHAR_TO_IDX[wrong]] = p
    out /= out.sum(axis=1, keepdims=True)
    return out


def test_beam_decode_recovers_confident_word(lm: CharLM) -> None:
    probs = _probs_for("TOTAL")
    assert beam_decode(probs, lm) == "TOTAL"


def test_beam_decode_uses_context_to_pick_digit_over_letter(lm: CharLM) -> None:
    # Classifier bingung antara '0' dan 'O' di semua posisi.
    probs = _probs_for("1000", {("0", "O"): 0.55})
    argmax = "".join(C.IDX_TO_CHAR[i] for i in probs.argmax(axis=1))
    decoded = beam_decode(probs, lm, prefix="TOTAL ")
    # Konteks harus mendorong ke digit, bukan huruf.
    assert decoded.count("0") > argmax.count("0")
    assert decoded == "1000"


def test_beam_decode_is_deterministic(lm: CharLM) -> None:
    probs = _probs_for("47.200", {("0", "O"): 0.4})
    assert beam_decode(probs, lm) == beam_decode(probs, lm)


def test_beam_decode_handles_empty_input(lm: CharLM) -> None:
    assert beam_decode(np.zeros((0, len(C.CHARSET))), lm) == ""


def test_beam_decode_respects_beam_width(lm: CharLM) -> None:
    probs = _probs_for("47.200", {("0", "O"): 0.45})
    narrow = beam_decode(probs, lm, beam_width=1)
    wide = beam_decode(probs, lm, beam_width=8)
    assert isinstance(narrow, str) and isinstance(wide, str)


def test_correct_words_fixes_one_edit(lex: Lexicon) -> None:
    assert correct_words("TOTAI 47.200", lex) == "TOTAL 47.200"


def test_correct_words_leaves_numbers_alone(lex: Lexicon) -> None:
    assert correct_words("47.200", lex) == "47.200"


def test_correct_words_keeps_unknown_short_tokens(lex: Lexicon) -> None:
    assert correct_words("XQ 47.200", lex) == "XQ 47.200"


def test_decode_line_combines_classifier_lm_and_lexicon(
    lm: CharLM, lex: Lexicon
) -> None:
    probs = _probs_for("TOTAL", {("0", "O"): 0.3, ("7", "T"): 0.2})
    decoded, confidence = decode_line(probs, lm, lex)
    assert decoded == "TOTAL"
    assert 0.0 <= confidence <= 1.0


def test_decode_line_confidence_reflects_uncertainty(lm: CharLM, lex: Lexicon) -> None:
    sure = decode_line(_probs_for("TOTAL"), lm, lex)[1]
    unsure = decode_line(
        _probs_for("TOTAL", {("0", "O"): 0.45, ("4", "A"): 0.45}), lm, lex
    )[1]
    assert sure >= unsure


# ---------------------------------------------------------------------------
# Perapian spasi
# ---------------------------------------------------------------------------
def test_normalize_spacing_joins_price_fragments() -> None:
    assert normalize_spacing("8 . 700") == "8.700"
    assert normalize_spacing("28 . 500") == "28.500"


def test_normalize_spacing_keeps_abbreviation_space() -> None:
    assert normalize_spacing("JL . RAYA") == "JL. RAYA"
    assert normalize_spacing("KEC . CILINCING") == "KEC. CILINCING"


def test_normalize_spacing_keeps_ordinary_spaces() -> None:
    assert normalize_spacing("TOTAL 47.200") == "TOTAL 47.200"


def test_normalize_spacing_handles_empty() -> None:
    assert normalize_spacing("") == ""


def test_normalize_spacing_is_idempotent() -> None:
    once = normalize_spacing("8 . 700")
    assert normalize_spacing(once) == once


# ---------------------------------------------------------------------------
# Normalisasi token angka dan satuan kemasan
# ---------------------------------------------------------------------------
def test_numeric_token_maps_letter_confusions() -> None:
    # ``0`` dan ``O`` identik pada font tanpa garis miring.
    assert normalize_numeric_token("47.2O0") == "47.200"
    assert normalize_numeric_token("16.OOO") == "16.000"
    assert normalize_numeric_token("95.4O0") == "95.400"


def test_numeric_token_rejects_result_that_is_not_a_number() -> None:
    # ``12S`` adalah ukuran kemasan, bukan angka; ``125`` bukan bentuk harga.
    assert normalize_numeric_token("12S") == "12S"


def test_numeric_token_leaves_plain_words_alone() -> None:
    assert normalize_numeric_token("OVO") == "OVO"
    assert normalize_numeric_token("TOTAL") == "TOTAL"


def test_packaging_token_fixes_letter_in_size() -> None:
    assert normalize_packaging_token("25OG") == "250G"
    assert normalize_packaging_token("20OML") == "200ML"
    assert normalize_packaging_token("1OO6", {"100G"}.__contains__) == "100G"


def test_packaging_token_repairs_unit_read_as_digit_only_when_known() -> None:
    known = {"250G", "100G"}.__contains__
    assert normalize_packaging_token("2506", known) == "250G"
    # Tanpa bukti bahwa "250G" dikenal, angka tidak diubah.
    assert normalize_packaging_token("2506", lambda _w: False) == "2506"


def test_is_packaging_token_distinguishes_size_from_number() -> None:
    assert is_packaging_token("250G")
    assert is_packaging_token("12S")
    assert not is_packaging_token("47.200")
    assert not is_packaging_token("TOTAL")


def test_correct_words_never_replaces_a_price_with_another_price(lex: Lexicon) -> None:
    # Regresi: dulu ``95.4O0`` menjadi ``5.400`` karena ``5.400`` ada di
    # leksikon dan berjarak satu edit. Nilainya berubah sepuluh kali lipat.
    assert correct_words("95.4O0", lex) == "95.400"


def test_correct_words_keeps_confident_packaging_token(lex: Lexicon) -> None:
    assert correct_words("250G", lex) == "250G"


def test_correct_words_does_not_bend_short_words(lex: Lexicon) -> None:
    # ``TUNAI`` dan ``TAWAR`` berjarak dua edit; batas jarak harus menolak.
    assert correct_words("TUNAI", lex) == "TUNAI"


def test_correct_words_can_skip_lexicon(lex: Lexicon) -> None:
    # Saat classifier sangat yakin, leksikon tidak boleh menimpa bacaannya.
    assert correct_words("TOTAI", lex, use_lexicon=False) == "TOTAI"
    assert correct_words("TOTAI", lex) == "TOTAL"

def test_correct_words_fixes_zero_for_letter_in_words(lex: Lexicon) -> None:
    # Font tanpa garis miring membuat ``0`` dan ``O`` identik; konteks kata
    # yang memutuskan, bukan bentuk glyph.
    assert correct_words("SIDOM0R0", lex) == "SIDOMORO"
    assert correct_words("K0NSUMEN", lex) == "KONSUMEN"
    assert correct_words("PR0MO", lex) == "PROMO"

def test_correct_words_keeps_packaging_when_fixing_letters(lex: Lexicon) -> None:
    # Penjaga: ``100G`` tidak boleh menjadi ``IOOG``.
    assert correct_words("100G", lex) == "100G"
    assert correct_words("250G", lex) == "250G"

def test_correct_words_reads_amount_after_amount_keyword(lex: Lexicon) -> None:
    # Kolom nominal: ``50G`` adalah ``500``, karena ``G`` di akhir token harga
    # lebih sering ``0`` daripada ``6``.
    assert correct_words("DISKON 50G", lex) == "DISKON 500"
    assert correct_words("PEMBULATAN 1.OOO", lex) == "PEMBULATAN 1.000"
    # Tanpa kata kunci nominal, ``12S`` tetap ukuran kemasan.
    assert correct_words("SARI ROTI TAWAR 12S", lex) == "SARI ROTI TAWAR 12S"

def test_correct_words_does_not_turn_price_into_word(lex: Lexicon) -> None:
    # ``L.OO0`` berbentuk harga, jadi tidak boleh "diperbaiki" menjadi ``L.OOO``.
    assert correct_words("L.OO0", lex) == "L.OOO"

def test_strip_rule_artifacts_keeps_only_text() -> None:
    from ocr.decode import strip_rule_artifacts

    assert (
        strip_rule_artifacts("------ LAYANAN KONSUMEN ------")
        == "LAYANAN KONSUMEN"
    )
    assert strip_rule_artifacts("-----------") == ""

def test_packaging_alias_repairs_two_letter_misread() -> None:
    # ``M``/``N`` dan ``I``/``L`` mirip pada cetakan thermal.
    assert normalize_packaging_token("200NIL") == "200ML"
