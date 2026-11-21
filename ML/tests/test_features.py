"""Tes ekstraksi fitur karakter.

Fitur adalah kontrak antara Python (latih) dan Dart (inferensi). Kalau
urutannya berubah, model JSON jadi tidak valid. Karena itu dimensi dan
sifat-sifat dasar fitur diuji ketat di sini.

Implementasi HOG **manual** — bukan ``skimage.hog`` — supaya rumusnya bisa
diport 1:1 ke Dart nanti.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C
from ocr.features import (
    extract_features,
    extract_features_batch,
    hole_features,
    hog_features,
    projection_features,
    zoning_features,
)


def _glyph(kind: str = "box") -> np.ndarray:
    """Glyph sintetis sederhana untuk tes."""
    img = np.zeros((C.CHAR_SIZE, C.CHAR_SIZE), dtype=np.uint8)
    if kind == "box":
        img[6:26, 8:24] = C.INK
    elif kind == "ring":
        img[6:26, 8:24] = C.INK
        img[10:22, 12:20] = C.PAPER
    elif kind == "blank":
        pass
    elif kind == "vbar":
        img[6:26, 14:18] = C.INK
    return img


@pytest.fixture(scope="module")
def fonts() -> list[Path]:
    from ocr.render_dataset import find_fonts

    found = find_fonts()
    if not found:
        pytest.skip("tidak ada font yang bisa dipakai di mesin ini")
    return found


# ---------------------------------------------------------------------------
# Dimensi
# ---------------------------------------------------------------------------
def test_hog_dimension_matches_config() -> None:
    assert hog_features(_glyph()).shape == (C.HOG_DIM,)


def test_zoning_dimension_matches_config() -> None:
    assert zoning_features(_glyph()).shape == (C.ZONING_DIM,)


def test_projection_dimension_matches_config() -> None:
    assert projection_features(_glyph()).shape == (C.PROJECTION_DIM,)


def test_total_feature_dimension_matches_config() -> None:
    assert extract_features(_glyph()).shape == (C.FEATURE_DIM,)


# ---------------------------------------------------------------------------
# Fitur topologi (lubang)
# ---------------------------------------------------------------------------
def test_hole_dimension_matches_config() -> None:
    assert hole_features(_glyph()).shape == (C.HOLE_DIM,)


def test_hole_features_count_lobang_dengan_benar() -> None:
    # 'ring' punya satu lubang, 'box' padat tidak punya.
    ring = hole_features(_glyph("ring"))
    box = hole_features(_glyph("box"))
    assert ring[0] == pytest.approx(1 / 3.0)
    assert box[0] == pytest.approx(0.0)


def test_hole_features_lihat_dua_lubang_pada_nol_bergaris() -> None:
    # Angka nol bertepi miring punya dua lubang; huruf O hanya satu.
    img = np.zeros((C.CHAR_SIZE, C.CHAR_SIZE), dtype=np.uint8)
    img[4:28, 8:24] = C.INK
    img[7:25, 11:21] = C.PAPER
    img[7:25, 14:18] = C.INK          # garis miring pemisah
    f = hole_features(img)
    assert f[0] == pytest.approx(2 / 3.0)


def test_hole_features_are_zero_for_blank_image() -> None:
    f = hole_features(_glyph("blank"))
    assert np.allclose(f, 0.0)


def test_batch_matches_single_extraction() -> None:
    imgs = np.stack([_glyph("box"), _glyph("ring"), _glyph("vbar")])
    batch = extract_features_batch(imgs)
    assert batch.shape == (3, C.FEATURE_DIM)
    for i in range(3):
        assert np.allclose(batch[i], extract_features(imgs[i]))


# ---------------------------------------------------------------------------
# Sifat numerik
# ---------------------------------------------------------------------------
def test_features_are_finite_float32() -> None:
    f = extract_features(_glyph())
    assert f.dtype == np.float32
    assert np.all(np.isfinite(f))


def test_features_are_deterministic() -> None:
    img = _glyph("ring")
    assert np.array_equal(extract_features(img), extract_features(img))


def test_blank_image_gives_zero_features() -> None:
    f = extract_features(_glyph("blank"))
    assert np.allclose(f, 0.0), f"citra kosong harus menghasilkan fitur nol"


def test_different_glyphs_give_different_features() -> None:
    a = extract_features(_glyph("box"))
    b = extract_features(_glyph("ring"))
    c = extract_features(_glyph("vbar"))
    assert not np.allclose(a, b)
    assert not np.allclose(b, c)
    assert not np.allclose(a, c)


def test_projection_is_normalized() -> None:
    f = projection_features(_glyph("box"))
    # Proyeksi dinormalisasi agar tidak bergantung pada jumlah tinta total.
    assert f.max() <= 1.0 + 1e-6
    assert f.min() >= 0.0


def test_zoning_density_is_a_fraction() -> None:
    f = zoning_features(_glyph("box"))
    density = f[0::C.ZONING_STATS]
    assert density.max() <= 1.0 + 1e-6
    assert density.min() >= 0.0


def test_hog_is_l2_block_normalized() -> None:
    # Setiap blok 36-d harus punya norma ~1 bila ada gradien, atau 0 bila
    # blok itu benar-benar kosong (tidak ada tinta sama sekali).
    f = hog_features(_glyph("box"))
    block_dim = C.HOG_BLOCK ** 2 * C.HOG_BINS  # 36
    for start in range(0, f.size, block_dim):
        norm = float(np.linalg.norm(f[start : start + block_dim]))
        assert norm == pytest.approx(1.0, abs=1e-4) or norm == 0.0, (
            f"blok di {start} norma {norm}"
        )


# ---------------------------------------------------------------------------
# Sensitivitas
# ---------------------------------------------------------------------------
def test_features_respond_to_stroke_direction() -> None:
    # Garis vertikal vs horizontal harus berbeda jelas di HOG.
    vert = np.zeros((C.CHAR_SIZE, C.CHAR_SIZE), dtype=np.uint8)
    vert[6:26, 15:17] = C.INK
    horz = np.zeros((C.CHAR_SIZE, C.CHAR_SIZE), dtype=np.uint8)
    horz[15:17, 6:26] = C.INK
    hv = hog_features(vert)
    hh = hog_features(horz)
    assert float(np.abs(hv - hh).sum()) > 1.0


def test_features_identify_glyphs_despite_bbox_jitter(fonts: list[Path]) -> None:
    # Uji yang bermakna: bukan kesamaan kosinus (HOG sengaja sensitif), tapi
    # apakah karakter masih dikenali benar saat bbox meleset 1 piksel.
    from ocr.render_dataset import crop_box, normalize_char, render_line

    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    image, boxes = render_line(text, fonts[0], 32)
    visible = [ch for ch in text if ch != " "]

    # Prototipe per kelas dari glyph bersih.
    protos: dict[str, np.ndarray] = {}
    for ch, box in zip(visible, boxes):
        protos[ch] = extract_features(normalize_char(crop_box(image, box)))
    proto_matrix = np.stack(list(protos.values()))
    proto_labels = list(protos.keys())

    correct = 0
    for ch, (x0, y0, x1, y1) in zip(visible, boxes):
        expanded = (max(0, x0 - 1), max(0, y0 - 1), x1 + 1, y1 + 1)
        feat = extract_features(normalize_char(crop_box(image, expanded)))
        sims = proto_matrix @ feat
        best = proto_labels[int(np.argmax(sims))]
        correct += int(best == ch)

    accuracy = correct / len(visible)
    assert accuracy >= 0.95, f"hanya {accuracy:.2%} dikenali benar setelah jitter bbox"


def test_features_alone_are_shift_sensitive_but_usable() -> None:
    # Tanpa normalisasi, fitur tetap cukup mirip untuk glyph yang sama.
    img = _glyph("box")
    shifted = np.zeros_like(img)
    shifted[:, 1:] = img[:, :-1]
    a = extract_features(img)
    b = extract_features(shifted)
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    assert cosine > 0.8, f"kemiripan terlalu rendah: {cosine}"


def test_batch_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        extract_features_batch(np.zeros((3, 16, 16), dtype=np.uint8))
