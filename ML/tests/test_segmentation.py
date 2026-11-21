"""Tes segmentasi kata dan karakter.

Dua kegagalan klasik yang harus dicegah di sini:

1. **Kata terbelah** — celah di dalam huruf (mis. antara ``r`` dan ``n``
   pada font renggang) dianggap spasi.
2. **Karakter tergabung** — dua huruf yang saling menyentuh (``rn`` mirip
   ``m``) dianggap satu glyph.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C
from ocr.chars import Glyph, segment_glyphs, split_wide_glyph
from ocr.words import Word, segment_words


def _render_word_band(word: str, spacing: int = 4, char_w: int = 10) -> np.ndarray:
    """Band sintetis: satu huruf per blok, celah antar huruf = ``spacing``."""
    width = len(word) * (char_w + spacing) + 20
    img = np.zeros((40, width), dtype=np.uint8)
    x = 10
    for _ in word:
        img[8:32, x : x + char_w] = C.INK
        x += char_w + spacing
    return img


def test_words_split_on_wide_gaps() -> None:
    img = np.zeros((40, 200), dtype=np.uint8)
    # "AB CD" -> dua kata, celah dalam kata 3 px, celah antar kata 20 px
    for x in (10, 23):
        img[8:32, x : x + 10] = C.INK
    for x in (53, 66):
        img[8:32, x : x + 10] = C.INK

    words = segment_words(img)
    assert len(words) == 2
    assert words[0].left == 10
    assert words[1].left == 53


def test_single_word_without_spaces() -> None:
    img = _render_word_band("ABCD", spacing=3)
    words = segment_words(img)
    assert len(words) == 1


def test_word_boxes_are_ordered_left_to_right() -> None:
    img = np.zeros((40, 300), dtype=np.uint8)
    for x in (10, 60, 120, 200):
        img[8:32, x : x + 10] = C.INK
    words = segment_words(img)
    assert [w.left for w in words] == sorted(w.left for w in words)


def test_word_boxes_are_horizontally_tight() -> None:
    img = _render_word_band("AB", spacing=3)
    words = segment_words(img)
    crop = img[words[0].top : words[0].bottom, words[0].left : words[0].right]
    assert crop[:, 0].any()
    assert crop[:, -1].any()


def test_empty_band_gives_no_words() -> None:
    assert segment_words(np.zeros((30, 100), dtype=np.uint8)) == []


def test_glyphs_split_each_letter() -> None:
    img = _render_word_band("ABCD", spacing=4)
    glyphs = segment_glyphs(img)
    assert len(glyphs) == 4


def test_glyph_boxes_are_tight() -> None:
    img = _render_word_band("ABC", spacing=4)
    for glyph in segment_glyphs(img):
        crop = glyph.crop(img)
        assert crop[0].any() and crop[-1].any()
        assert crop[:, 0].any() and crop[:, -1].any()


def test_touching_letters_are_split_by_valley() -> None:
    # Dua huruf yang menyentuh lewat jembatan tipis harus tetap terpisah.
    img = np.zeros((40, 40), dtype=np.uint8)
    img[8:32, 8:20] = C.INK
    img[8:32, 20:32] = C.INK
    img[8:32, 19:21] = C.PAPER      # jembatan tipis: hanya tersisa di 2 baris
    img[20:22, 19:21] = C.INK
    glyphs = segment_glyphs(img)
    assert len(glyphs) == 2


def test_small_dots_are_not_dropped() -> None:
    # Titik pada huruf 'i' dan tanda titik harus ikut jadi glyph.
    img = np.zeros((40, 60), dtype=np.uint8)
    img[8:30, 10:18] = C.INK        # badan huruf
    img[32:36, 12:16] = C.INK       # titik di bawah (seperti koma)
    glyphs = segment_glyphs(img)
    assert len(glyphs) == 2


def test_split_wide_glyph_returns_two_halves() -> None:
    img = np.zeros((40, 60), dtype=np.uint8)
    img[8:32, 5:25] = C.INK
    img[8:32, 35:55] = C.INK
    glyph = Glyph(left=5, top=8, right=55, bottom=32)
    parts = split_wide_glyph(img, glyph)
    assert len(parts) == 2
    assert parts[0].right <= parts[1].left


def test_split_wide_glyph_keeps_normal_glyph() -> None:
    img = np.zeros((40, 30), dtype=np.uint8)
    img[8:32, 10:18] = C.INK
    glyph = Glyph(left=10, top=8, right=18, bottom=32)
    assert split_wide_glyph(img, glyph) == [glyph]


def test_glyph_boxes_stay_inside_the_band() -> None:
    img = _render_word_band("WIDE", spacing=5)
    h, w = img.shape
    for glyph in segment_glyphs(img):
        assert 0 <= glyph.left < glyph.right <= w
        assert 0 <= glyph.top < glyph.bottom <= h


def test_segmentation_handles_low_contrast_edges() -> None:
    # Huruf dengan tepi tipis tetap harus terpisah.
    img = np.zeros((40, 80), dtype=np.uint8)
    img[10:30, 10:18] = C.INK
    img[10:30, 24:32] = C.INK
    img[10:30, 40:48] = C.INK
    assert len(segment_glyphs(img)) == 3


# ---------------------------------------------------------------------------
# Penggabungan komponen bertumpuk (titik pada i/j/:, huruf terbelah)
# ---------------------------------------------------------------------------
def test_stacked_fragments_are_merged_into_one_glyph() -> None:
    # Huruf yang terbelah baris tipis oleh resampling: dua komponen dengan
    # sumbu-x sama, bertumpuk, berjarak 2 px.
    img = np.zeros((50, 40), dtype=np.uint8)
    img[0:34, 5:35] = C.INK
    img[36:47, 5:35] = C.INK
    glyphs = segment_glyphs(img, reference_width=30.0)
    assert len(glyphs) == 1
    assert glyphs[0].top == 0
    assert glyphs[0].bottom == 47


def test_dot_above_letter_is_merged_with_its_stem() -> None:
    # ``i``: titik kecil di atas batang, celah 2 px, sumbu-x hampir sama.
    img = np.zeros((50, 40), dtype=np.uint8)
    img[4:12, 12:24] = C.INK
    img[14:46, 12:24] = C.INK
    glyphs = segment_glyphs(img, reference_width=30.0)
    assert len(glyphs) == 1


def test_side_by_side_letters_are_not_merged_vertically() -> None:
    # Dua huruf bersebelahan tidak boleh melebur hanya karena bertumpuk
    # sebagian secara horizontal.
    img = np.zeros((50, 80), dtype=np.uint8)
    img[4:46, 5:35] = C.INK
    img[4:46, 40:70] = C.INK
    glyphs = segment_glyphs(img, reference_width=30.0)
    assert len(glyphs) == 2


def test_word_gap_threshold_tolerates_wide_letter_spacing() -> None:
    # ``2 X 9.800``: celah huruf melebar sampai 23 px, celah kata 44 px.
    # Ambang berbasis median celah akan memisahkan ``2`` dari ``X``.
    from ocr.words import word_gap_threshold

    runs = [(0, 26), (70, 101), (148, 171), (189, 199), (218, 242), (253, 279)]
    threshold = word_gap_threshold(runs)
    assert 23 < threshold < 44
