"""Tes renderer dataset karakter.

Renderer menghasilkan pasangan (gambar karakter, label) tanpa anotasi
manual: kita tahu teks yang dirender, jadi labelnya gratis. Syarat
utamanya adalah bbox **tight** dari piksel tinta — sama seperti yang
dihasilkan segmentasi saat inferensi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C
from ocr import render_dataset as R


@pytest.fixture(scope="module")
def fonts() -> list[Path]:
    found = R.find_fonts()
    if not found:
        pytest.skip("tidak ada font yang bisa dipakai di mesin ini")
    return found


@pytest.fixture(scope="module")
def font(fonts: list[Path]) -> Path:
    return fonts[0]


def test_find_fonts_returns_existing_files(fonts: list[Path]) -> None:
    assert fonts, "minimal satu font harus ditemukan"
    for path in fonts:
        assert path.exists()
        assert path.suffix.lower() in {".ttf", ".ttc", ".otf"}


def test_render_line_returns_one_box_per_visible_char(font: Path) -> None:
    text = "TOTAL 8.700"
    image, boxes = R.render_line(text, font, 26)
    visible = [ch for ch in text if ch != " "]
    assert len(boxes) == len(visible)
    assert image.dtype == np.uint8
    assert image.ndim == 2


def test_boxes_are_tight_against_ink(font: Path) -> None:
    image, boxes = R.render_line("ABCDE12345", font, 32)
    for box in boxes:
        crop = R.crop_box(image, box)
        assert crop.size > 0
        # bbox tight: keempat sisi harus menyentuh tinta
        assert crop[0].any(), f"sisi atas kosong untuk {box}"
        assert crop[-1].any(), f"sisi bawah kosong untuk {box}"
        assert crop[:, 0].any(), f"sisi kiri kosong untuk {box}"
        assert crop[:, -1].any(), f"sisi kanan kosong untuk {box}"


def test_boxes_are_ordered_left_to_right(font: Path) -> None:
    _, boxes = R.render_line("INDOMARET", font, 26)
    lefts = [b[0] for b in boxes]
    assert lefts == sorted(lefts)


def test_boxes_stay_inside_the_image(font: Path) -> None:
    image, boxes = R.render_line("POCARI SWEAT 500ML", font, 26)
    h, w = image.shape
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= w
        assert 0 <= y0 < y1 <= h


def test_normalize_char_produces_32x32_canvas() -> None:
    crop = np.zeros((10, 6), dtype=np.uint8)
    crop[2:8, 1:5] = C.INK
    out = R.normalize_char(crop)
    assert out.shape == (C.CHAR_SIZE, C.CHAR_SIZE)
    assert out.dtype == np.uint8


def test_normalize_char_preserves_aspect_ratio() -> None:
    wide = np.zeros((10, 40), dtype=np.uint8)
    wide[3:7, :] = C.INK
    out = R.normalize_char(wide)
    # Huruf lebar harus tetap lebar: tinta tidak boleh mengisi seluruh tinggi.
    rows_with_ink = np.where(out.any(axis=1))[0]
    assert len(rows_with_ink) < C.CHAR_SIZE * 0.5


def test_normalize_char_aligns_ink_centroid_to_canvas_center() -> None:
    # Glyph yang tidak simetris harus digeser sehingga pusat massanya di tengah
    # kanvas. Ini yang membuat fitur tahan terhadap jitter bbox 1-2 piksel.
    crop = np.zeros((20, 12), dtype=np.uint8)
    crop[2:6, 1:5] = C.INK          # tinta hanya di kiri atas
    out = R.normalize_char(crop)
    ys, xs = np.nonzero(out)
    assert ys.size > 0
    centre = (C.CHAR_SIZE - 1) / 2.0
    assert abs(float(ys.mean()) - centre) < 1.5, ys.mean()
    assert abs(float(xs.mean()) - centre) < 1.5, xs.mean()


def test_normalize_char_keeps_ink_inside_canvas() -> None:
    crop = np.zeros((14, 9), dtype=np.uint8)
    crop[1:13, 2:7] = C.INK
    out = R.normalize_char(crop)
    assert out[C.PAD : C.CHAR_SIZE - C.PAD, C.PAD : C.CHAR_SIZE - C.PAD].any()


def test_augment_is_deterministic_for_a_seed() -> None:
    crop = np.zeros((C.CHAR_SIZE, C.CHAR_SIZE), dtype=np.uint8)
    crop[8:24, 12:20] = C.INK
    a = R.augment(crop, seed=99)
    b = R.augment(crop, seed=99)
    assert np.array_equal(a, b)


def test_augment_keeps_ink_and_stays_binary() -> None:
    crop = np.zeros((C.CHAR_SIZE, C.CHAR_SIZE), dtype=np.uint8)
    crop[8:24, 12:20] = C.INK
    for seed in range(20):
        out = R.augment(crop, seed=seed)
        assert out.ndim == 2
        assert out.dtype == np.uint8
        assert out.any(), f"augmentasi menghapus seluruh tinta (seed={seed})"
        assert set(np.unique(out).tolist()) <= {0, 255}
        # Keluaran harus sudah dipotong ketat ke bbox tinta.
        assert out[0].any() and out[-1].any()
        assert out[:, 0].any() and out[:, -1].any()


def test_augment_result_normalizes_to_32x32() -> None:
    crop = np.zeros((C.CHAR_SIZE, C.CHAR_SIZE), dtype=np.uint8)
    crop[8:24, 12:20] = C.INK
    for seed in range(10):
        out = R.normalize_char(R.augment(crop, seed=seed))
        assert out.shape == (C.CHAR_SIZE, C.CHAR_SIZE)
        assert out.dtype == np.uint8


def test_build_dataset_labels_match_rendered_text(font: Path) -> None:
    X, y, text_index, _ = R.build_dataset(
        n_lines=6, fonts=[font], samples_per_line=1, seed=1, augment_rounds=0
    )
    assert X.shape[1:] == (C.CHAR_SIZE, C.CHAR_SIZE)
    assert X.dtype == np.uint8
    assert len(X) == len(y) == len(text_index)
    # text_index menandai baris sumber tiap karakter, untuk split bebas kebocoran.
    assert len(set(text_index.tolist())) == 6
    # Setiap label harus karakter yang ada di charset.
    for idx in y:
        assert C.IDX_TO_CHAR[int(idx)] in C.CHARSET


def test_build_dataset_is_deterministic(font: Path) -> None:
    a = R.build_dataset(
        n_lines=5, fonts=[font], samples_per_line=1, seed=3, augment_rounds=0
    )
    b = R.build_dataset(
        n_lines=5, fonts=[font], samples_per_line=1, seed=3, augment_rounds=0
    )
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_build_dataset_covers_all_classes_on_a_long_corpus(
    fonts: list[Path],
) -> None:
    X, y, _, _ = R.build_dataset(
        n_lines=1500, fonts=fonts[:4], samples_per_line=1, seed=11, augment_rounds=0
    )
    seen = set(np.unique(y).tolist())
    missing = set(range(len(C.CHARSET))) - seen
    assert not missing, f"kelas tidak terwakili: {sorted(C.IDX_TO_CHAR[i] for i in missing)}"


def test_boost_reaches_minimum_for_every_class(fonts: list[Path]) -> None:
    # Korpus pendek pasti kekurangan kelas langka; booster harus menutupinya.
    X, y, _, _ = R.build_dataset(
        n_lines=200,
        fonts=fonts[:2],
        samples_per_line=1,
        seed=21,
        augment_rounds=0,
        min_per_class=120,
    )
    counts = np.bincount(y, minlength=len(C.CHARSET))
    short = {
        C.IDX_TO_CHAR[i]: int(counts[i])
        for i in range(len(C.CHARSET))
        if counts[i] < 120
    }
    assert not short, f"kelas masih di bawah minimum: {short}"


def test_boosted_dataset_keeps_labels_consistent(fonts: list[Path]) -> None:
    X, y, text_index, _ = R.build_dataset(
        n_lines=200,
        fonts=fonts[:2],
        samples_per_line=1,
        seed=22,
        augment_rounds=0,
        min_per_class=60,
    )
    assert len(X) == len(y) == len(text_index)
    assert X.shape[1:] == (C.CHAR_SIZE, C.CHAR_SIZE)
    # Setiap sampel harus punya tinta.
    assert all(sample.any() for sample in X)


def test_first_round_is_always_the_clean_sample(
    fonts: list[Path], monkeypatch
) -> None:
    # Kalau semua varian diaugmentasi, classifier tidak pernah melihat glyph
    # bersih — padahal citra uji adalah render sintetis yang tajam.
    import numpy as np

    # Kondensasi horizontal dilewati di sini: ia sengaja membuang glyph yang
    # menyusut di bawah 5 px, sehingga jumlah sampel tidak lagi sama dengan
    # jumlah karakter terlihat.
    monkeypatch.setattr(C, "AUG_CONDENSE_PROB", 0.0)
    monkeypatch.setattr(C, "AUG_LOWERCASE_PROB", 0.0)

    rng = np.random.default_rng(0)
    produced = R._sample_line(
        "TOTAL 8.700", 0, fonts[:1], rng, augment_rounds=2
    )
    assert produced, "harus ada sampel"
    clean_ref = R._sample_line(
        "TOTAL 8.700", 0, fonts[:1], np.random.default_rng(0), augment_rounds=0
    )
    # Round pertama tiap karakter harus identik dengan versi tanpa augmentasi.
    n_visible = len([c for c in "TOTAL 8.700" if c != " "])
    assert len(clean_ref) == n_visible
    for i in range(n_visible):
        assert np.array_equal(produced[i * 3][0], clean_ref[i][0])


def test_augment_rounds_adds_that_many_extra_variants(fonts: list[Path]) -> None:
    rng = np.random.default_rng(1)
    one = R._sample_line("ABC", 0, fonts[:1], rng, augment_rounds=0)
    three = R._sample_line("ABC", 0, fonts[:1], rng, augment_rounds=2)
    assert len(one) == 3
    assert len(three) == 9


def test_cap_balances_the_class_distribution(fonts: list[Path]) -> None:
    # Garis putus-putus membuat '-' sangat dominan; cap harus memotongnya.
    X, y, _, _ = R.build_dataset(
        n_lines=400,
        fonts=fonts[:2],
        samples_per_line=1,
        seed=31,
        augment_rounds=0,
        min_per_class=40,
        max_per_class=90,
    )
    counts = np.bincount(y, minlength=len(C.CHARSET))
    over = {
        C.IDX_TO_CHAR[i]: int(counts[i])
        for i in range(len(C.CHARSET))
        if counts[i] > 90
    }
    short = {
        C.IDX_TO_CHAR[i]: int(counts[i])
        for i in range(len(C.CHARSET))
        if counts[i] < 40
    }
    assert not over, f"kelas melebihi cap: {over}"
    assert not short, f"kelas di bawah minimum: {short}"
    assert len(X) == len(y)


def test_cap_is_deterministic(fonts: list[Path]) -> None:
    kwargs = dict(
        n_lines=300,
        fonts=fonts[:2],
        samples_per_line=1,
        seed=33,
        augment_rounds=0,
        min_per_class=30,
        max_per_class=60,
    )
    a = R.build_dataset(**kwargs)
    b = R.build_dataset(**kwargs)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2])
    assert np.array_equal(a[3], b[3])


def test_variant_marks_clean_and_augmented_samples(fonts: list[Path]) -> None:
    X, y, _, variant = R.build_dataset(
        n_lines=50,
        fonts=fonts[:2],
        samples_per_line=1,
        seed=41,
        augment_rounds=2,
        min_per_class=None,
    )
    assert set(np.unique(variant).tolist()) == {0, 1, 2}
    # Setiap glyph bersih harus punya tepat dua pasangan augmentasi.
    n_clean = int((variant == 0).sum())
    assert n_clean == len(X) // 3


def test_cap_is_a_noop_when_no_class_exceeds_it(fonts: list[Path]) -> None:
    X_uncapped, y_uncapped, _, _ = R.build_dataset(
        n_lines=150,
        fonts=fonts[:1],
        samples_per_line=1,
        seed=35,
        augment_rounds=0,
        min_per_class=20,
    )
    X_capped, y_capped, _, _ = R.build_dataset(
        n_lines=150,
        fonts=fonts[:1],
        samples_per_line=1,
        seed=35,
        augment_rounds=0,
        min_per_class=20,
        max_per_class=10**9,
    )
    assert len(X_uncapped) == len(X_capped)
    assert np.array_equal(y_uncapped, y_capped)

def test_lowercase_skips_letters_that_collide_with_other_classes(
    fonts: list[Path], monkeypatch
) -> None:
    # ``l`` sama dengan ``I``. Kalau ikut dilatih sebagai ``L``, classifier
    # belajar bahwa garis tegak boleh berarti ``L`` dan ``TOTAL`` terbaca
    # ``TOTAI``. Baris yang memuat ``l`` harus tetap kapital.
    import numpy as np

    monkeypatch.setattr(C, "AUG_LOWERCASE_PROB", 1.0)
    monkeypatch.setattr(C, "AUG_CONDENSE_PROB", 0.0)
    for seed in range(6):
        produced = R._sample_line(
            "TOTAL BELANJA", 0, fonts[:1], np.random.default_rng(seed),
            augment_rounds=1,
        )
        assert produced, "harus ada sampel"
        # Semua crop berasal dari render kapital: tinggi glyph seragam.
        heights = {crop.shape[0] for crop, _x, _y, _w in produced}
        assert len(heights) == 1, f"baris ter-lowercase pada seed {seed}"
