"""Tes penghapusan garis, barcode, dan sampah.

Garis putus-putus dan barcode adalah sumber error massal: kalau ikut
diserahkan ke classifier karakter, satu barcode bisa menghasilkan puluhan
"huruf" palsu dan merusak rekonsiliasi total.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C
from ocr.lines import LineBox, detect_lines
from ocr.rules import classify_band, filter_bands, is_barcode, is_rule


def _rule_band() -> tuple[np.ndarray, LineBox]:
    img = np.zeros((40, 300), dtype=np.uint8)
    # garis putus-putus: segmen 8 px dengan celah 6 px
    for x in range(10, 290, 14):
        img[18:21, x : x + 8] = C.INK
    return img, LineBox(top=0, bottom=40, left=10, right=290)


def _text_band() -> tuple[np.ndarray, LineBox]:
    img = np.zeros((60, 300), dtype=np.uint8)
    rng = np.random.default_rng(0)
    # huruf-huruf kecil yang tersebar
    for x in range(12, 280, 12):
        w = int(rng.integers(5, 9))
        h = int(rng.integers(20, 34))
        top = 15 + int(rng.integers(0, 8))
        img[top : top + h, x : x + w] = C.INK
    return img, LineBox(top=0, bottom=60, left=12, right=288)


def _barcode_band() -> tuple[np.ndarray, LineBox]:
    img = np.zeros((200, 600), dtype=np.uint8)
    rng = np.random.default_rng(1)
    x = 40
    while x < 560:
        w = int(rng.integers(1, 5))
        img[40:170, x : x + w] = C.INK
        x += w + int(rng.integers(1, 5))
    return img, LineBox(top=0, bottom=200, left=40, right=560)


def test_rule_is_recognized() -> None:
    img, band = _rule_band()
    assert is_rule(img, band)


def test_text_is_not_a_rule() -> None:
    img, band = _text_band()
    assert not is_rule(img, band)


def test_barcode_is_recognized() -> None:
    img, band = _barcode_band()
    assert is_barcode(img, band)


def test_text_is_not_a_barcode() -> None:
    img, band = _text_band()
    assert not is_barcode(img, band)


def test_rule_is_not_mistaken_for_barcode() -> None:
    img, band = _rule_band()
    assert not is_barcode(img, band)


def test_classify_band_labels_all_three() -> None:
    assert classify_band(*_rule_band()) == "rule"
    assert classify_band(*_barcode_band()) == "barcode"
    assert classify_band(*_text_band()) == "text"


def test_filter_bands_drops_rules_and_barcodes() -> None:
    text_img, _ = _text_band()
    rule_img, _ = _rule_band()
    bar_img, _ = _barcode_band()

    # Tiga wilayah yang tidak bertumpuk.
    canvas = np.zeros((420, 600), dtype=np.uint8)
    canvas[0:60, 0:300] = text_img
    canvas[80:120, 0:300] = rule_img
    canvas[140:340, 0:600] = bar_img

    boxes = [
        LineBox(top=0, bottom=60, left=12, right=288),
        LineBox(top=95, bottom=105, left=10, right=290),
        LineBox(top=140, bottom=340, left=40, right=560),
    ]
    kept = filter_bands(canvas, boxes)
    assert [b.top for b in kept] == [0]


def test_filter_bands_keeps_order_and_returns_new_list() -> None:
    img, box = _text_band()
    kept = filter_bands(img, [box])
    assert kept == [box]
    assert kept is not [box]


def test_rule_detection_handles_solid_line() -> None:
    img = np.zeros((40, 300), dtype=np.uint8)
    img[18:22, 10:290] = C.INK
    band = LineBox(top=0, bottom=40, left=10, right=290)
    assert is_rule(img, band)


def test_text_band_with_dashes_inside_is_still_text() -> None:
    # Baris "JL. RAYA KNO 12 - KEL" mengandung tanda hubung tapi bukan garis.
    img = np.zeros((50, 400), dtype=np.uint8)
    for x in range(10, 380, 14):
        img[12:38, x : x + 8] = C.INK      # badan huruf
    img[22:25, 300:340] = C.INK           # satu tanda hubung panjang
    band = LineBox(top=0, bottom=50, left=10, right=380)
    assert not is_rule(img, band)


def test_detect_and_filter_pipeline_on_synthetic_page() -> None:
    canvas = np.zeros((300, 400), dtype=np.uint8)
    for row in range(4):
        top = 20 + row * 60
        for x in range(20, 360, 14):
            canvas[top : top + 26, x : x + 8] = C.INK
    canvas[250:253, 20:380] = C.INK        # garis penutup

    boxes = detect_lines(canvas)
    kept = filter_bands(canvas, boxes)
    assert len(kept) == 4
