"""Segmentasi kata di dalam satu pita baris.

Metode: proyeksi vertikal jumlah tinta per kolom, lalu celah (kolom tanpa
tinta) yang lebih lebar dari ambang dianggap batas kata.

Ambangnya **relatif terhadap median celah** di baris itu, bukan angka
absolut. Font monospace punya celah antar huruf yang seragam; celah antar
kata biasanya 2-3 kali lebih lebar. Dengan ambang relatif, segmentasi
tetap bekerja pada resolusi berapa pun.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import ocr.config as C


@dataclass(frozen=True)
class Word:
    """Kotak kata di dalam koordinat citra penuh."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def _column_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Rentetan True (kolom berisi tinta) sebagai (awal, akhir-eksklusif)."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _gaps(mask: np.ndarray) -> list[tuple[int, int]]:
    """Celah (rentetan False) sebagai (awal, akhir-eksklusif)."""
    return _column_runs(~mask)


def word_gap_threshold(runs: list[tuple[int, int]]) -> float:
    """Tentukan ambang celah batas kata dari lebar glyph baris itu.

    Batas kata pada struk thermal kira-kira selebar **satu huruf kosong**.
    Jadi ambangnya dinyatakan relatif terhadap lebar glyph:

        ambang = WORD_GAP_TO_WIDTH x median_lebar_glyph

    Pendekatan ini dipilih setelah dua alternatif gagal pada data nyata:

    - *Pengali median celah* — gagal pada ``2 X 9.800`` karena celah antar
      huruf di baris itu melebar sampai 23 px sedangkan celah kata 44 px;
      rasionya hanya ~2x sehingga ambang ikut memisahkan ``2`` dan ``X``.
    - *k-means/Otsu pada celah* — gagal karena sebaran celah punya **tiga**
      kelompok: celah huruf (~5-23 px), celah kata (~40-50 px), dan celah
      kolom antara nama dan harga (~1000 px). Dengan k=2, kelompok celah
      kata ikut tergabung ke kelompok celah huruf, sehingga seluruh nama
      barang menjadi satu kata.

    Args:
        runs: rentetan kolom bertinta ``(awal, akhir-eksklusif)``.

    Returns:
        Ambang celah (piksel). Celah yang **lebih besar** dari nilai ini
        dianggap batas kata.
    """
    if not runs:
        return 2.0
    widths = [end - start for start, end in runs]
    median_width = float(np.median(widths))
    if median_width <= 0:
        return 2.0

    # "Lebar satu huruf" adalah lebar glyph **ditambah** celah khasnya, karena
    # spasi antar kata pada struk kira-kira selebar satu sel huruf kosong.
    gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
    median_gap = float(np.median(gaps)) if gaps else 0.0
    advance = median_width + max(0.0, median_gap)
    return max(2.0, advance * C.WORD_GAP_TO_WIDTH)


def segment_words(
    band: np.ndarray,
    band_top: int = 0,
    band_left: int = 0,
    gap_ratio: float | None = None,
) -> list[Word]:
    """Pecah pita baris menjadi kata.

    Ambang batas ditentukan oleh :func:`word_gap_threshold` dari lebar glyph
    baris itu sendiri, bukan dari sebaran celah.

    Args:
        band: citra biner pita baris (255 = tinta), sudah dipotong ke baris.
        band_top: offset vertikal pita di citra penuh.
        band_left: offset horizontal pita di citra penuh.
        gap_ratio: tidak dipakai lagi; dipertahankan agar pemanggil lama
            tetap jalan.

    Returns:
        Daftar ``Word`` terurut kiri ke kanan.
    """
    ink = np.asarray(band) == C.INK
    if ink.ndim != 2 or ink.size == 0 or not ink.any():
        return []

    columns = ink.any(axis=0)
    runs = _column_runs(columns)
    if not runs:
        return []

    threshold = word_gap_threshold(runs)

    rows = np.flatnonzero(ink.any(axis=1))
    top = int(rows[0])
    bottom = int(rows[-1]) + 1

    words: list[Word] = []
    current_left = runs[0][0]
    for (start, end), (next_start, _) in zip(runs, runs[1:]):
        if next_start - end > threshold:
            words.append(
                Word(
                    left=band_left + current_left,
                    top=band_top + top,
                    right=band_left + end,
                    bottom=band_top + bottom,
                )
            )
            current_left = next_start
    words.append(
        Word(
            left=band_left + current_left,
            top=band_top + top,
            right=band_left + runs[-1][1],
            bottom=band_top + bottom,
        )
    )
    return words
