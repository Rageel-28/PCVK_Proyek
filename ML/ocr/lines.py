"""Deteksi baris teks dari citra biner struk.

Pendekatan klasik: proyeksi horizontal jumlah tinta per baris, lalu cari
pita (band) yang nilainya di atas ambang. Ambang dipilih dengan Otsu 1-d
pada profil itu sendiri, jadi tidak perlu angka ajaib per foto.

Dua kasus yang harus ditangani:

1. **Baris menempel** — huruf turun (``g``, ``y``) dan naik (``t``, ``l``)
   bisa membuat dua baris bersambung. Pita yang terlalu tinggi dipecah di
   lembah terdalam profilnya.
2. **Bintik noise** — komponen 2-3 piksel tidak boleh jadi "baris".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import ocr.config as C


@dataclass(frozen=True)
class LineBox:
    """Kotak baris: ``top``/``bottom`` eksklusif di bawah, ``right`` eksklusif."""

    top: int
    bottom: int
    left: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left


def horizontal_profile(binary: np.ndarray) -> np.ndarray:
    """Jumlah piksel tinta per baris."""
    return (binary == C.INK).sum(axis=1).astype(np.int64)


def _otsu_threshold(profile: np.ndarray) -> float:
    """Otsu 1-d pada profil baris (bukan pada citra)."""
    values = profile.astype(np.float64)
    if values.size == 0 or values.max() <= 0:
        return 0.0
    hist, edges = np.histogram(values, bins=64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return 0.0

    weight_bg = np.cumsum(hist).astype(np.float64)
    weight_fg = total - weight_bg
    sum_total = float((hist * centers).sum())
    cum_mean = np.cumsum(hist * centers)

    valid = (weight_bg > 0) & (weight_fg > 0)
    if not valid.any():
        return float(values.mean())

    mean_bg = cum_mean[valid] / weight_bg[valid]
    mean_fg = (sum_total - cum_mean[valid]) / weight_fg[valid]
    between = weight_bg[valid] * weight_fg[valid] * (mean_bg - mean_fg) ** 2
    best = int(np.argmax(between))
    return float(centers[valid][best])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Indeks awal/akhir (eksklusif) dari setiap rentetan True."""
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


def _reference_height(heights: np.ndarray) -> float:
    """Tinggi baris teks acuan, tahan terhadap garis dan pita tergabung.

    Dua pengganggu harus dibuang bersamaan:

    - **Garis pemisah** (``-----``) tingginya hanya 8-9 px. Kalau ikut
      dihitung, median turun drastis (terukur 9 px padahal teksnya 45 px),
      lalu semua baris teks dianggap "terlalu tinggi" dan dipecah menjadi
      tiga bagian.
    - **Pita yang sudah tergabung** (dua baris menyatu) tingginya dua kali
      lipat. Kalau ikut dihitung, tidak ada yang dianggap terlalu tinggi.

    Karena itu penyaringan dilakukan dua tahap: buang dulu pencilan jauh
    memakai persentil ke-75 sebagai patokan, lalu ambil median separuh bawah
    dari sisanya.
    """
    if heights.size == 0:
        return 0.0
    high = float(np.percentile(heights, 75))
    if high <= 0:
        return 0.0
    kept = heights[(heights >= 0.5 * high) & (heights <= 1.5 * high)]
    if kept.size == 0:
        return high
    return float(np.median(kept[kept <= np.median(kept)]))


def split_tall_bands(
    bands: list[LineBox], profile: np.ndarray, median_height: float
) -> list[LineBox]:
    """Pecah pita yang jauh lebih tinggi dari median di lembah terdalamnya."""
    if median_height <= 0:
        return bands

    limit = median_height * C.LINE_SPLIT_RATIO

    out: list[LineBox] = []
    for band in bands:
        if band.height <= limit:
            out.append(band)
            continue

        segment = profile[band.top : band.bottom]
        if segment.size < 6:
            out.append(band)
            continue

        # Cari titik terendah di bagian tengah (hindari tepi pita).
        margin = max(1, segment.size // 5)
        inner = segment[margin : segment.size - margin]
        if inner.size == 0:
            out.append(band)
            continue
        cut = int(np.argmin(inner)) + margin

        # Kalau lembahnya tidak jelas (masih tinggi), jangan dipaksa pecah.
        if segment[cut] > 0.75 * float(np.median(segment)):
            out.append(band)
            continue

        top = LineBox(band.top, band.top + cut, band.left, band.right)
        bottom = LineBox(band.top + cut, band.bottom, band.left, band.right)
        out.extend(split_tall_bands([top], profile, median_height))
        out.extend(split_tall_bands([bottom], profile, median_height))
    return out


def detect_lines(
    binary: np.ndarray,
    min_ink: int | None = None,
    min_height: int | None = None,
) -> list[LineBox]:
    """Deteksi pita baris teks.

    Args:
        binary: citra biner, 255 = tinta.
        min_ink: jumlah piksel tinta minimum per baris agar dianggap isi.
        min_height: tinggi pita minimum (menyaring garis tipis dan noise).

    Returns:
        Daftar ``LineBox`` terurut dari atas ke bawah.
    """
    if binary.ndim != 2:
        raise ValueError(f"citra harus 2-d, dapat {binary.shape}")
    if binary.size == 0:
        return []

    ink = binary == C.INK
    if not ink.any():
        return []

    if min_height is None:
        min_height = max(4, int(round(binary.shape[0] * 0.004)))

    profile = ink.sum(axis=1).astype(np.int64)
    if min_ink is None:
        # Ambang relatif terhadap lebar citra, bukan Otsu pada profil.
        # Otsu gagal di sini: barcode punya tinta ribuan piksel per baris,
        # sedangkan baris teks hanya ratusan, sehingga Otsu memilih ambang
        # di antara keduanya dan membuang semua teks.
        min_ink = max(2, int(round(binary.shape[1] * 0.008)))

    threshold = float(min_ink)
    active = profile >= threshold

    # Isi celah vertikal kecil di dalam satu baris (mis. pada huruf ':').
    active = _fill_small_gaps(active, max_gap=max(1, min_height // 3))

    raw: list[LineBox] = []
    for start, end in _runs(active):
        if end - start < min_height:
            continue
        columns = ink[start:end].any(axis=0)
        cols = np.flatnonzero(columns)
        if cols.size == 0:
            continue
        raw.append(
            LineBox(
                top=int(start),
                bottom=int(end),
                left=int(cols[0]),
                right=int(cols[-1]) + 1,
            )
        )

    if not raw:
        return []

    heights_arr = np.asarray([b.height for b in raw], dtype=np.float64)
    median_height = float(np.median(heights_arr))
    reference_height = _reference_height(heights_arr)
    bands = split_tall_bands(raw, profile, reference_height)

    # Gabung pita yang terpisah sangat dekat (masih satu baris).
    bands = _merge_close(bands, max_gap=max(1, int(median_height * 0.15)))
    return [b for b in bands if b.height >= min_height]


def _fill_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Tutup celah False pendek di antara dua True."""
    out = mask.copy()
    for start, end in _runs(~mask):
        if 0 < start and end < len(mask) and (end - start) <= max_gap:
            out[start:end] = True
    return out


def _merge_close(bands: list[LineBox], max_gap: int) -> list[LineBox]:
    """Gabung fragmen baris yang terbelah celah kecil.

    Hanya fragmen pendek yang digabung. Dua baris penuh yang kebetulan
    berdekatan tidak boleh dilebur — itu justru kasus yang paling merusak.
    """
    if not bands:
        return []
    bands = sorted(bands, key=lambda b: b.top)
    median = float(np.median([b.height for b in bands]))
    merged = [bands[0]]
    for band in bands[1:]:
        prev = merged[-1]
        gap = band.top - prev.bottom
        is_fragment = min(prev.height, band.height) < 0.6 * median
        if gap <= max_gap and is_fragment:
            merged[-1] = LineBox(
                top=prev.top,
                bottom=max(prev.bottom, band.bottom),
                left=min(prev.left, band.left),
                right=max(prev.right, band.right),
            )
        else:
            merged.append(band)
    return merged
