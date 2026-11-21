"""Pipeline inferensi OCR klasik — rujukan Python untuk port Dart.

Urutan tahap:

    citra
      -> praproses (grayscale, denoise, Sauvola, upscale adaptif)
      -> deteksi baris (profil horizontal)
      -> buang garis & barcode
      -> segmentasi kata (profil vertikal)
      -> segmentasi glyph (connected components + merge + split)
      -> normalisasi ke 32x32
      -> fitur 1844-d
      -> classifier linear (softmax bersuhu)
      -> decode: beam search + model bahasa + leksikon
      -> rakit teks per baris

Modul ini adalah **rujukan paritas**: port Dart harus menghasilkan teks
yang sama. Setiap fungsi di sini punya padanan langsung di Dart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import ocr.config as C
from ocr.chars import Glyph, segment_glyphs
from ocr.decode import (
    correct_words,
    decode_line,
    normalize_spacing,
    strip_rule_artifacts,
)
from ocr.features import extract_features_batch
from ocr.lexicon import Lexicon
from ocr.lines import LineBox, detect_lines
from ocr.lm import CharLM
from ocr.preprocess import Prepared, prepare
from ocr.render_dataset import normalize_char
from ocr.rules import filter_bands
from ocr.templates import Templates
from ocr.words import Word, segment_words

_PUNCTUATION_ONLY = set(".,:;/-+%")


@dataclass
class RecognizedLine:
    """Satu baris hasil OCR beserta kotaknya di citra penuh."""

    text: str
    left: int
    top: int
    right: int
    bottom: int
    confidence: float


@dataclass
class PageResult:
    """Hasil OCR satu halaman."""

    lines: list[RecognizedLine] = field(default_factory=list)
    scale: float = 1.0
    readable: bool = False

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


def to_glyph_image(crop: np.ndarray) -> np.ndarray:
    """Normalisasi crop glyph biner ke kanvas 32x32 (255 = tinta)."""
    return normalize_char(crop)


def line_crop(binary: np.ndarray, line: RecognizedLine) -> np.ndarray:
    """Potong pita baris dari citra biner."""
    return binary[line.top : line.bottom, line.left : line.right]


def _word_probs(
    band: np.ndarray,
    band_top: int,
    band_left: int,
    word: Word,
    model,
    reference_width: float = 0.0,
    median_height: float = 0.0,
) -> np.ndarray | None:
    """Probabilitas classifier untuk seluruh glyph dalam satu kata.

    Glyph yang jauh lebih pendek dari huruf biasa dibatasi ke kelas tanda
    baca. Alasannya: :func:`normalize_char` meregangkan setiap crop ke kanvas
    32x32, sehingga titik (11x11 px) dan huruf ``O`` (28x44 px) sama-sama
    menjadi blob besar — model tidak punya informasi ukuran lagi dan
    menebak ``O``. Membatasi kandidat adalah cara standar memakai informasi
    yang model tidak lihat, tanpa mengubah model.
    """
    word_band = band[
        word.top - band_top : word.bottom - band_top,
        word.left - band_left : word.right - band_left,
    ]
    glyphs: list[Glyph] = segment_glyphs(
        word_band, word.top, word.left, reference_width=reference_width
    )
    if not glyphs:
        return None

    images = []
    tiny: list[bool] = []
    for glyph in glyphs:
        crop = band[
            glyph.top - band_top : glyph.bottom - band_top,
            glyph.left - band_left : glyph.right - band_left,
        ]
        if crop.size == 0:
            continue
        images.append(to_glyph_image(crop))
        tiny.append(
            median_height > 0
            and glyph.height < median_height * C.TINY_GLYPH_RATIO
        )
    if not images:
        return None

    probs = model.predict_proba(extract_features_batch(np.stack(images)))
    if any(tiny):
        probs = np.array(probs, dtype=np.float64, copy=True)
        allowed = [C.CHAR_TO_IDX[ch] for ch in C.PUNCTUATION_CLASSES]
        for index, is_tiny in enumerate(tiny):
            if not is_tiny:
                continue
            mask = np.zeros(probs.shape[1], dtype=bool)
            mask[allowed] = True
            probs[index, ~mask] = 0.0
            total = probs[index].sum()
            if total > 0:
                probs[index] /= total
    return probs


def _line_metrics(
    band: np.ndarray, band_top: int, band_left: int
) -> tuple[float, float]:
    """Lebar dan tinggi glyph acuan dari **seluruh** baris.

    Diambil dari semua kata di baris, bukan per kata: pada kata tebal
    (``TUNAI``, ``KEMBALI``) seluruh huruf menyatu sehingga median lebar kata
    itu sendiri tidak bermakna.

    Returns:
        ``(lebar_acuan, tinggi_acuan)``. ``(0.0, 0.0)`` bila baris kosong.
    """
    from ocr.chars import _components

    words = segment_words(band, band_top, band_left)
    widths: list[int] = []
    heights: list[int] = []
    for word in words:
        word_band = band[
            word.top - band_top : word.bottom - band_top,
            word.left - band_left : word.right - band_left,
        ]
        for glyph in _components(word_band):
            widths.append(glyph.width)
            heights.append(glyph.height)
    if not widths:
        return 0.0, 0.0
    return float(np.median(widths)), float(np.median(heights))


def _decode_line(
    band: np.ndarray,
    band_top: int,
    band_left: int,
    model,
    lm: CharLM | None,
    lexicon: Lexicon | None,
) -> tuple[str, float]:
    """Rakit teks satu baris dari glyph-glyphnya."""
    words: list[Word] = segment_words(band, band_top, band_left)
    if not words:
        return "", 0.0

    reference_width, median_height = _line_metrics(band, band_top, band_left)

    pieces: list[str] = []
    confidences: list[float] = []
    prefix = ""

    for word in words:
        probs = _word_probs(
            band,
            band_top,
            band_left,
            word,
            model,
            reference_width,
            median_height,
        )
        if probs is None:
            continue

        if lm is not None:
            word_text, confidence = decode_line(probs, lm, lexicon, prefix=prefix)
        else:
            indices = probs.argmax(axis=1)
            word_text = "".join(C.IDX_TO_CHAR[int(i)] for i in indices)
            confidence = float(probs.max(axis=1).mean())

        if not word_text:
            continue

        # Tanda baca berdiri sendiri: tempelkan ke kata sebelumnya supaya
        # "8 . 700" tidak menjadi tiga kata terpisah.
        if pieces and set(word_text) <= _PUNCTUATION_ONLY:
            pieces[-1] = pieces[-1] + word_text
        else:
            pieces.append(word_text)

        prefix = (prefix + " " + word_text).strip()
        confidences.append(confidence)

    if not pieces:
        return "", 0.0

    text = normalize_spacing(" ".join(pieces))
    if lexicon is not None:
        # Koreksi kolom nominal butuh konteks antarkata (``DISKON 50G`` ->
        # ``DISKON 500``), sedangkan decoder bekerja per kata. Jalankan ulang
        # pada baris utuh tanpa koreksi leksikon: hanya normalisasi angka,
        # satuan, dan nominal yang aktif.
        text = correct_words(text, lexicon, use_lexicon=False)
    text = strip_rule_artifacts(text)
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return text, confidence


def recognize_page(
    image: np.ndarray,
    model,
    lm: CharLM | None = None,
    lexicon: Lexicon | None = None,
    templates: Templates | None = None,
    min_confidence: float = 0.30,
    prepared: Prepared | None = None,
) -> PageResult:
    """Jalankan pipeline OCR lengkap pada satu citra struk.

    Args:
        image: citra grayscale atau BGR.
        model: objek dengan ``predict_proba(features) -> (N, 42)``.
        lm: model bahasa karakter; bila ``None``, decode memakai argmax murni.
        lexicon: leksikon kata; bila ``None``, tidak ada koreksi ejaan.
        templates: templat baris teks tetap (tagline, footer, alamat); bila
            ``None``, tidak ada pencocokan templat.
        min_confidence: ambang rata-rata keyakinan per baris; baris di bawah
            ambang dibuang karena biasanya noise.
        prepared: hasil praproses yang sudah ada (untuk menghindari kerja
            ulang pada evaluasi batch).

    Returns:
        ``PageResult`` berisi baris-baris teks dan kotaknya.
    """
    prep = prepared if prepared is not None else prepare(image)
    binary = prep.binary

    bands = detect_lines(binary)
    if not bands:
        return PageResult(lines=[], scale=prep.scale, readable=False)
    bands = filter_bands(binary, bands)
    if not bands:
        return PageResult(lines=[], scale=prep.scale, readable=False)

    results: list[RecognizedLine] = []
    for band in bands:
        crop = binary[band.top : band.bottom, band.left : band.right]
        text, confidence = _decode_line(
            crop, band.top, band.left, model, lm, lexicon
        )
        text = text.strip()
        if templates is not None and text:
            text = templates.snap(text)
        if not text:
            continue
        if confidence < min_confidence:
            continue
        results.append(
            RecognizedLine(
                text=text,
                left=band.left,
                top=band.top,
                right=band.right,
                bottom=band.bottom,
                confidence=confidence,
            )
        )

    readable = len(results) >= 2
    return PageResult(lines=results, scale=prep.scale, readable=readable)
