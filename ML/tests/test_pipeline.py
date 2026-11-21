"""Tes pipeline inferensi end-to-end (Python sebagai rujukan paritas).

Pipeline di sini adalah **rujukan** yang harus menghasilkan output identik
dengan port Dart. Kalau keduanya berbeda, salah satu rusak.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C
from ocr.pipeline import PageResult, line_crop, recognize_page, to_glyph_image


def _fake_model():
    """Model tiruan: memprediksi karakter dari rasio lebar/tinggi glyph."""

    class Model:
        charset = C.CHARSET
        feature_dim = C.FEATURE_DIM

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            probs = np.full((len(features), len(C.CHARSET)), 1e-6)
            for i, feat in enumerate(features):
                # 'I' sempit, 'W' lebar; sisanya 'A'.
                width_ratio = float(feat[-1])
                idx = C.CHAR_TO_IDX["I"] if width_ratio < 0.3 else C.CHAR_TO_IDX["W"]
                probs[i, idx] = 1.0
            return probs

    return Model()


def _synthetic_page() -> np.ndarray:
    """Halaman dengan dua baris teks yang jelas.

    Latar putih (255) dan tinta hitam (0), sama seperti foto struk sungguhan.
    Versi lama memakai latar hitam, sehingga Sauvola menganggap latar sebagai
    tinta dan pitanya menjadi blok padat selebar halaman.
    """
    img = np.full((200, 400), 255, dtype=np.uint8)
    for row in range(2):
        top = 40 + row * 80
        for x in range(30, 360, 30):
            img[top : top + 30, x : x + 12] = 0
    return img


def test_recognize_page_returns_lines() -> None:
    result = recognize_page(_synthetic_page(), model=_fake_model())
    assert isinstance(result, PageResult)
    assert len(result.lines) == 2


def test_lines_are_ordered_top_to_bottom() -> None:
    result = recognize_page(_synthetic_page(), model=_fake_model())
    tops = [line.top for line in result.lines]
    assert tops == sorted(tops)


def test_line_text_contains_only_charset_characters() -> None:
    result = recognize_page(_synthetic_page(), model=_fake_model())
    for line in result.lines:
        assert set(line.text) <= set(C.CHARSET) | {" "}


def test_line_boxes_are_reported_in_prepared_coordinates() -> None:
    # Koordinat baris memakai ruang citra **setelah** praproses (deskew +
    # upscale adaptif), bukan citra masukan. Pemakai hanya membandingkan
    # posisi relatif antar baris, jadi konsistensi internal ini cukup.
    from ocr.preprocess import prepare

    page = _synthetic_page()
    prepared = prepare(page)
    height, width = prepared.binary.shape
    result = recognize_page(page, model=_fake_model())
    assert result.lines, "halaman sintetis harus menghasilkan baris"
    for line in result.lines:
        assert 0 <= line.top < line.bottom <= height
        assert 0 <= line.left < line.right <= width


def test_empty_page_gives_no_lines() -> None:
    result = recognize_page(np.zeros((100, 100), dtype=np.uint8), model=_fake_model())
    assert result.lines == []


def test_to_glyph_image_produces_32x32() -> None:
    page = _synthetic_page()
    crop = page[40:70, 30:42]
    out = to_glyph_image(crop)
    assert out.shape == (C.CHAR_SIZE, C.CHAR_SIZE)
    assert out.dtype == np.uint8


def test_to_glyph_image_is_deterministic() -> None:
    page = _synthetic_page()
    crop = page[40:70, 30:42]
    assert np.array_equal(to_glyph_image(crop), to_glyph_image(crop))


def test_line_crop_returns_binary_band() -> None:
    page = _synthetic_page()
    result = recognize_page(page, model=_fake_model())
    band = line_crop(page, result.lines[0])
    assert band.dtype == np.uint8
    assert (band == C.INK).any()


def test_confidence_is_between_zero_and_one() -> None:
    result = recognize_page(_synthetic_page(), model=_fake_model())
    for line in result.lines:
        assert 0.0 <= line.confidence <= 1.0


def test_page_reports_readability() -> None:
    blank = recognize_page(np.zeros((100, 100), dtype=np.uint8), model=_fake_model())
    assert blank.readable is False

    page = recognize_page(_synthetic_page(), model=_fake_model())
    assert page.readable is True
