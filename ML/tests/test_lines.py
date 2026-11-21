"""Tes deteksi baris teks.

Segmentasi baris adalah titik kegagalan paling mahal: satu baris tergabung
dengan tetangganya membuat seluruh isi baris itu salah dibaca.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C
from ocr.lines import LineBox, detect_lines, split_tall_bands


def _three_lines() -> np.ndarray:
    """Tiga pita teks terpisah dengan celah jelas."""
    img = np.zeros((200, 300), dtype=np.uint8)
    img[20:40, 30:270] = C.INK
    img[80:100, 30:270] = C.INK
    img[150:170, 30:270] = C.INK
    return img


def test_detects_three_separate_lines() -> None:
    lines = detect_lines(_three_lines())
    assert len(lines) == 3


def test_line_boxes_are_ordered_top_to_bottom() -> None:
    lines = detect_lines(_three_lines())
    tops = [b.top for b in lines]
    assert tops == sorted(tops)


def test_line_boxes_cover_their_ink_rows() -> None:
    img = _three_lines()
    lines = detect_lines(img)
    assert lines[0].top <= 20 and lines[0].bottom >= 40
    assert lines[1].top <= 80 and lines[1].bottom >= 100
    assert lines[2].top <= 150 and lines[2].bottom >= 170


def test_line_boxes_are_horizontally_tight() -> None:
    lines = detect_lines(_three_lines())
    for line in lines:
        assert line.left <= 30
        assert line.right >= 270


def test_ignores_specks_below_min_ink() -> None:
    img = _three_lines()
    img[110:112, 100:102] = C.INK      # bintik 2x2
    lines = detect_lines(img)
    assert len(lines) == 3


def test_empty_image_yields_no_lines() -> None:
    assert detect_lines(np.zeros((100, 100), dtype=np.uint8)) == []


def test_touching_lines_are_split_by_their_valley() -> None:
    # Dua baris yang nyaris menempel (jarak 1 px) harus tetap terpisah.
    img = np.zeros((120, 200), dtype=np.uint8)
    img[20:44, 20:180] = C.INK
    img[46:70, 20:180] = C.INK
    lines = detect_lines(img)
    assert len(lines) == 2


def test_split_tall_bands_splits_at_the_weakest_row() -> None:
    band = LineBox(top=0, bottom=60, left=0, right=100)
    profile = np.zeros(60, dtype=np.int64)
    profile[0:28] = 100
    profile[28:32] = 5          # lembah
    profile[32:60] = 100
    parts = split_tall_bands([band], profile, median_height=30.0)
    assert len(parts) == 2
    assert parts[0].bottom <= 32
    assert parts[1].top >= 28


def test_split_tall_bands_keeps_normal_bands() -> None:
    band = LineBox(top=0, bottom=30, left=0, right=100)
    profile = np.full(30, 100, dtype=np.int64)
    parts = split_tall_bands([band], profile, median_height=30.0)
    assert parts == [band]


def test_detect_lines_accepts_grayscale_like_noise() -> None:
    rng = np.random.default_rng(0)
    img = (rng.random((80, 80)) < 0.01).astype(np.uint8) * C.INK
    lines = detect_lines(img)
    # Tidak boleh meledak jadi ratusan baris karena noise.
    assert len(lines) <= 3


def test_reference_height_ignores_rule_lines() -> None:
    """Garis pemisah setinggi 8 px tidak boleh menurunkan tinggi acuan.

    Kalau ikut dihitung, semua baris teks dianggap terlalu tinggi lalu
    dipecah menjadi tiga bagian.
    """
    from ocr.lines import _reference_height

    heights = np.array([45, 45, 45, 8, 45, 8, 45], dtype=np.float64)
    assert _reference_height(heights) == pytest.approx(45.0, abs=2.0)


def test_reference_height_survives_merged_bands() -> None:
    from ocr.lines import _reference_height

    heights = np.array([45, 45, 90, 90, 45], dtype=np.float64)
    assert _reference_height(heights) == pytest.approx(45.0, abs=2.0)


def test_line_height_estimate_is_not_dragged_down_by_rules() -> None:
    from ocr.preprocess import estimate_line_height

    img = np.zeros((400, 600), dtype=np.uint8)
    # Tiga baris teks tinggi 40 px.
    for top in (20, 140, 260):
        img[top : top + 40, 40:560] = C.INK
    # Dua garis pemisah tipis.
    img[100:106, 40:560] = C.INK
    img[220:226, 40:560] = C.INK
    assert estimate_line_height(img) >= 20.0
