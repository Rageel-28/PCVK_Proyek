"""Tes praproses: grayscale, denoise, dan binarisasi Sauvola.

Konvensi proyek: keluaran biner selalu ``255 = tinta``, ``0 = kertas``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C
from ocr.preprocess import binarize, denoise, to_gray


def _synthetic_receipt() -> np.ndarray:
    """Kertas terang dengan teks gelap dan sedikit gradien pencahayaan."""
    img = np.full((200, 300), 235, dtype=np.uint8)
    img[40:52, 30:120] = 30
    img[80:92, 30:200] = 30
    # gradien pencahayaan: kiri lebih gelap
    ramp = np.linspace(-35, 10, 300).astype(np.int16)
    img = np.clip(img.astype(np.int16) + ramp[None, :], 0, 255).astype(np.uint8)
    return img


def test_to_gray_keeps_grayscale_input() -> None:
    img = _synthetic_receipt()
    assert np.array_equal(to_gray(img), img)


def test_to_gray_converts_bgr() -> None:
    import cv2

    gray = _synthetic_receipt()
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = to_gray(bgr)
    assert out.ndim == 2
    assert out.dtype == np.uint8
    assert out.shape == gray.shape


def test_to_gray_rejects_unsupported_shape() -> None:
    with pytest.raises(ValueError):
        to_gray(np.zeros((4, 4, 5), dtype=np.uint8))


def test_denoise_preserves_shape_and_range() -> None:
    img = _synthetic_receipt()
    out = denoise(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_binarize_uses_ink_255_convention() -> None:
    binary = binarize(_synthetic_receipt())
    assert binary.dtype == np.uint8
    assert set(np.unique(binary).tolist()) <= {0, 255}
    # Teks gelap harus jadi tinta (255), kertas jadi 0.
    assert binary[45, 60] == C.INK
    assert binary[10, 150] == C.PAPER


def test_binarize_survives_uneven_illumination() -> None:
    # Tanpa ambang adaptif, separuh kanan akan salah klasifikasi.
    binary = binarize(_synthetic_receipt())
    # Baris teks kedua membentang 30..200; harus utuh.
    assert binary[86, 40] == C.INK
    assert binary[86, 190] == C.INK
    # Daerah kertas di kanan dan kiri tetap kertas.
    assert binary[10, 20] == C.PAPER
    assert binary[10, 280] == C.PAPER


def test_binarize_is_mostly_ink_free_on_blank_paper() -> None:
    blank = np.full((120, 120), 240, dtype=np.uint8)
    binary = binarize(blank)
    ink_ratio = float((binary == C.INK).mean())
    assert ink_ratio < 0.02, f"kertas kosong menghasilkan {ink_ratio:.1%} tinta"


def test_binarize_keeps_solid_text_block() -> None:
    img = np.full((120, 240), 240, dtype=np.uint8)
    img[40:80, 40:200] = 20
    binary = binarize(img)
    block = binary[45:75, 50:190]
    assert float((block == C.INK).mean()) > 0.95


def test_binarize_handles_tiny_image() -> None:
    tiny = np.full((3, 3), 200, dtype=np.uint8)
    binary = binarize(tiny)
    assert binary.shape == tiny.shape


def test_binarize_is_deterministic() -> None:
    img = _synthetic_receipt()
    assert np.array_equal(binarize(img), binarize(img))


def test_binarize_upscales_when_text_is_small() -> None:
    # Struk dengan teks setinggi ~8 px harus di-upscale agar glyph >= 24 px.
    from ocr.preprocess import prepare

    small = np.full((300, 400), 240, dtype=np.uint8)
    for row in range(6):
        y = 40 + row * 12
        small[y : y + 7, 40:300] = 40
    prepared = prepare(small)
    assert prepared.scale > 1.0
    assert prepared.binary.shape[0] >= small.shape[0]


def test_prepare_reports_a_scale_factor() -> None:
    from ocr.preprocess import prepare

    out = prepare(_synthetic_receipt())
    assert out.scale >= 1.0
    assert out.binary.dtype == np.uint8
    assert out.gray.shape == _synthetic_receipt().shape


def _synthetic_page_with_edge_strip() -> np.ndarray:
    """Biner 400x600: strip penuh setinggi citra di tepi kiri plus teks."""
    page = np.zeros((600, 400), dtype=np.uint8)
    page[:, 0:30] = 255
    for i in range(5):
        y = 60 + i * 80
        page[y : y + 24, 60:260] = 255
    return page


def test_artifact_mask_drops_full_height_strip_but_keeps_text() -> None:
    from ocr.preprocess import artifact_mask

    page = _synthetic_page_with_edge_strip()
    mask = artifact_mask(page)

    # Strip artefak ditandai seluruhnya.
    assert np.all(mask[:, 5] == 255)
    # Batang teks tidak ditandai.
    assert np.all(mask[60:84, 60:260] == 0)


def test_strip_artifacts_removes_edge_strip_in_place() -> None:
    from ocr.preprocess import _strip_artifacts

    page = _synthetic_page_with_edge_strip()
    _strip_artifacts(page)

    assert np.count_nonzero(page[:, 5]) == 0
    assert np.count_nonzero(page[60:84, 60:260]) == 24 * 200


def test_artifact_mask_at_scale_matches_full_resolution_mask() -> None:
    from ocr.preprocess import artifact_mask_at_scale

    # Citra besar: pencarian dilakukan pada versi diperkecil.
    page = np.zeros((2400, 3200), dtype=np.uint8)
    page[:, 0:240] = 255
    page[200:260, 400:2000] = 255
    mask = artifact_mask_at_scale(page)

    assert mask.shape == page.shape
    assert np.all(mask[:, 20] == 255)
    assert np.all(mask[200:260, 400:2000] == 0)
