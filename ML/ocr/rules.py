"""Klasifikasi dan penyaringan pita baris: teks vs garis vs barcode.

Aturan praktis dari pengamatan struk thermal:

- **Garis** (``-----``) tinggi pitanya sangat pendek dibanding baris teks,
  tintanya membentuk segmen horizontal panjang, dan hampir semua piksel
  tinta berada di 2-3 baris yang sama.
- **Barcode** tinggi pitanya jauh lebih besar dari median, tintanya
  membentuk banyak transisi vertikal hitam-putih.
- **Teks** tidak memenuhi keduanya.

Semua ambang relatif terhadap median tinggi baris halaman, bukan angka
absolut, supaya tetap bekerja pada foto resolusi berbeda.
"""

from __future__ import annotations

import numpy as np

import ocr.config as C
from ocr.lines import LineBox


def _crop(binary: np.ndarray, band: LineBox) -> np.ndarray:
    return binary[band.top : band.bottom, band.left : band.right]


def _run_lengths(row: np.ndarray) -> list[int]:
    """Panjang setiap rentetan True di satu baris."""
    lengths: list[int] = []
    current = 0
    for value in row:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def is_rule(
    binary: np.ndarray, band: LineBox, median_height: float = 0.0
) -> bool:
    """True bila pita ini adalah garis horizontal (putus-putus atau padat)."""
    crop = _crop(binary, band) == C.INK
    if crop.size == 0:
        return False

    height, width = crop.shape
    rows_with_ink = np.flatnonzero(crop.any(axis=1))
    if rows_with_ink.size == 0:
        return False

    columns = np.flatnonzero(crop.any(axis=0))
    if columns.size == 0:
        return False
    span = (columns[-1] - columns[0] + 1) / max(1, width)
    if span <= 0.6:
        return False

    # Analisis baris yang paling banyak tintanya.
    best_row = int(np.argmax(crop.sum(axis=1)))
    runs = _run_lengths(crop[best_row])
    if not runs:
        return False

    dashed = len(runs) >= 3 and float(np.median(runs)) < 0.25 * width
    solid = max(runs) > 0.5 * width

    if not (dashed or solid):
        return False

    # Kriteria tinggi: garis jauh lebih pendek daripada baris teks. Ini yang
    # dipakai saat median halaman diketahui (kasus nyata).
    if median_height > 0:
        return height <= 0.55 * median_height

    # Tanpa median, pakai konsentrasi tinta: garis menempati sedikit baris.
    return rows_with_ink.size <= max(6, int(height * 0.35))


def is_graphic(binary: np.ndarray, band: LineBox) -> bool:
    """True bila pita berisi gambar/logo, bukan teks.

    Logo toko (wordmark, blok abu-abu, kotak promo) menghasilkan rentetan
    tinta horizontal yang sangat panjang: satu baris piksel bisa menyala
    sampai setengah lebar pita. Huruf tidak pernah seperti itu — pada baris
    teks rentetan terpanjang biasanya di bawah 10% lebar, bahkan untuk huruf
    tebal yang menyatu.
    """
    crop = _crop(binary, band) == C.INK
    if crop.size == 0:
        return False

    height, width = crop.shape
    if width < 40:
        return False

    longest = 0
    for row in crop:
        runs = _run_lengths(row)
        if runs:
            longest = max(longest, max(runs))
    if longest > 0.25 * width:
        return True
    return False

def is_barcode(binary: np.ndarray, band: LineBox, median_height: float = 0.0) -> bool:
    """True bila pita ini barcode: bar vertikal seragam dan pita sangat tinggi."""
    crop = _crop(binary, band) == C.INK
    if crop.size == 0:
        return False

    height, width = crop.shape
    if width < 40:
        return False

    # Barcode jauh lebih tinggi dari baris teks biasa.
    if median_height > 0 and height < 1.6 * median_height:
        return False

    # Bar barcode vertikal penuh: jumlah tinta per baris hampir konstan.
    # Teks biasa punya profil baris yang sangat bervariasi. Hanya baris
    # berisi tinta yang dihitung, supaya pita yang sedikit lebih longgar
    # dari barcode tetap dikenali.
    per_row_all = crop.sum(axis=1).astype(np.float64)
    per_row = per_row_all[per_row_all > 0]
    if per_row.size < 10:
        return False
    mean = float(per_row.mean())
    if mean <= 0:
        return False
    coefficient_of_variation = float(per_row.std() / mean)
    if coefficient_of_variation > 0.25:
        return False

    mid = crop[height // 4 : max(height // 4 + 1, 3 * height // 4)]
    transitions = np.abs(np.diff(mid.astype(np.int8), axis=1)).sum(axis=1)
    mean_transitions = float(transitions.mean()) if transitions.size else 0.0
    return mean_transitions > 0.05 * width


def classify_band(
    binary: np.ndarray, band: LineBox, median_height: float = 0.0
) -> str:
    """Kembalikan ``"rule"``, ``"barcode"``, ``"graphic"``, atau ``"text"``."""
    if is_rule(binary, band, median_height):
        return "rule"
    if is_graphic(binary, band):
        return "graphic"
    if is_barcode(binary, band, median_height):
        return "barcode"
    return "text"


def filter_bands(
    binary: np.ndarray, bands: list[LineBox], median_height: float = 0.0
) -> list[LineBox]:
    """Buang pita yang bukan teks, pertahankan urutan."""
    if not bands:
        return []
    if median_height <= 0:
        median_height = float(np.median([b.height for b in bands]))
    return [
        band
        for band in bands
        if classify_band(binary, band, median_height) == "text"
    ]
