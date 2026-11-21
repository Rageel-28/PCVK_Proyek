"""Tes estimasi kemiringan (skew) dan deskew.

Struk difoto dengan tangan hampir selalu sedikit miring. Dampaknya besar:
satu baris teks yang seharusnya menempati ~50 baris piksel bisa tersebar
ke 100+ baris, sehingga proyeksi horizontal tidak lagi memisahkan baris.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C
from ocr.preprocess import deskew, estimate_skew


def _text_page(angle_deg: float = 0.0) -> np.ndarray:
    """Halaman biner berisi beberapa baris teks, diputar ``angle_deg``."""
    import cv2

    page = np.zeros((400, 600), dtype=np.uint8)
    for row in range(5):
        top = 40 + row * 60
        for x in range(40, 560, 24):
            page[top : top + 26, x : x + 14] = C.INK

    if angle_deg == 0.0:
        return page

    center = (page.shape[1] / 2 - 0.5, page.shape[0] / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        page, matrix, (page.shape[1], page.shape[0]),
        flags=cv2.INTER_NEAREST, borderValue=C.PAPER,
    )
    return rotated


def test_estimate_skew_returns_zero_for_straight_page() -> None:
    angle = estimate_skew(_text_page(0.0))
    assert abs(angle) <= 0.5


def test_estimate_skew_detects_positive_rotation() -> None:
    angle = estimate_skew(_text_page(3.0))
    # OpenCV memutar berlawanan arah jarum jam untuk sudut positif.
    assert angle == pytest.approx(-3.0, abs=0.6)


def test_estimate_skew_detects_negative_rotation() -> None:
    angle = estimate_skew(_text_page(-2.0))
    assert angle == pytest.approx(2.0, abs=0.6)


def test_estimate_skew_is_zero_for_blank_page() -> None:
    assert estimate_skew(np.zeros((200, 200), dtype=np.uint8)) == 0.0


def test_deskew_straightens_a_rotated_page() -> None:
    rotated = _text_page(3.0)
    fixed, angle = deskew(rotated)
    assert fixed.shape == rotated.shape
    assert abs(angle) > 1.0
    # Setelah dikoreksi, baris harus jauh lebih terpisah.
    after = estimate_skew(fixed)
    assert abs(after) <= 0.8


def test_deskew_leaves_straight_page_untouched() -> None:
    page = _text_page(0.0)
    fixed, angle = deskew(page)
    assert abs(angle) <= 0.5
    assert np.array_equal(fixed, page)


def test_deskew_rejects_non_binary_input() -> None:
    with pytest.raises(ValueError):
        deskew(np.zeros((10, 10, 3), dtype=np.uint8))
