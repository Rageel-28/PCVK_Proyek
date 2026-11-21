"""Tes konfigurasi pipeline OCR klasik.

Konfigurasi adalah kontrak: dimensi fitur, charset, dan nama kelas harus
stabil karena model JSON yang diekspor ke Dart bergantung padanya.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C


def test_charset_has_42_classes() -> None:
    # 10 digit + 26 huruf besar + 6 simbol
    assert len(C.CHARSET) == 42


def test_charset_mapping_is_bijective() -> None:
    assert len(set(C.CHARSET)) == len(C.CHARSET)
    for idx, ch in enumerate(C.CHARSET):
        assert C.CHAR_TO_IDX[ch] == idx
        assert C.IDX_TO_CHAR[idx] == ch


def test_hog_feature_dimension_is_1764() -> None:
    # 32x32 kanvas, sel 4 -> 8x8 sel; blok 2x2 stride 1 -> 7x7 blok
    # 7*7 blok * 2*2 sel * 9 bin = 1764
    assert C.HOG_DIM == 1764


def test_total_feature_dimension_is_1844() -> None:
    # HOG 1764 + zoning 48 + proyeksi 24 + topologi 8
    assert C.ZONING_DIM == 48
    assert C.PROJECTION_DIM == 24
    assert C.HOLE_DIM == 8
    assert C.FEATURE_DIM == 1844


def test_canvas_and_augmentation_ranges_are_sane() -> None:
    assert C.CHAR_SIZE == 32
    assert C.PAD == 2
    assert 0.0 < C.AUG_BLUR_SIGMA_MAX <= 2.0
    assert C.AUG_ROTATE_DEG_MAX > 0

def test_binarization_convention_is_ink_255() -> None:
    # Konvensi seluruh proyek: 255 = tinta, 0 = kertas.
    assert C.INK == 255
    assert C.PAPER == 0
