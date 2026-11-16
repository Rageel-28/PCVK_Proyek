"""Ekstraksi fitur karakter: HOG + Zoning/Grid Features.

Dua kelompok fitur, keduanya ditulis manual (tanpa scikit-image / OpenCV)
karena versi Dart harus meniru perhitungan yang sama persis:

  A. HOG 324 dimensi — ARAH tepi (bentuk goresan)
     1. Inversi citra: tinta menjadi terang (255 - piksel)
     2. Gradien: gx = I[:,x+1] - I[:,x-1], gy = I[y+1,:] - I[y-1,:]
     3. Magnitudo m = sqrt(gx^2 + gy^2)
        Orientasi  theta = atan2(gy, gx) dalam derajat, dimodulo 180
     4. Setiap sel 8x8 mengakumulasi m ke 9 bin orientasi
        (interpolasi bilinear ke dua bin terdekat)
     5. Blok 2x2 sel (bertumpuk, stride 1) dinormalisasi L2-Hys
     6. Hasil: 3 x 3 blok x 2 x 2 sel x 9 bin = 324 dimensi

  B. Zoning 16 dimensi — SEBARAN tinta
     Kanvas dibagi grid 4x4; tiap zona menyumbang rata-rata intensitas tinta
     (0..1). Menangkap "huruf ini padat di bagian mana", yang tidak direkam
     HOG karena HOG membuang posisi demi arah.

Vektor akhir = konkatenasi keduanya, 324 + 16 = 340 dimensi. Penyamaan skala
antar kelompok dilakukan StandardScaler saat pelatihan (lihat `train_char.py`).

Normalisasi L2-Hys: bagi dengan norma L2, potong nilai > 0.2, bagi lagi.
Ini membuat fitur tahan terhadap perbedaan kontras antar citra.
"""

from __future__ import annotations

import numpy as np

import config as C


def _gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = img.astype(np.float32) / 255.0
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return gx, gy


def hog(
    img: np.ndarray,
    cell: int = C.HOG_CELL,
    bins: int = C.HOG_BINS,
    block: int = C.HOG_BLOCK,
) -> np.ndarray:
    """Hitung vektor HOG untuk satu karakter (kanvas persegi)."""
    if img.ndim == 3:
        img = img.mean(axis=2)
    if img.shape[0] != C.CHAR_SIZE or img.shape[1] != C.CHAR_SIZE:
        raise ValueError(f"Dimensi harus {C.CHAR_SIZE}x{C.CHAR_SIZE}, dapat {img.shape}")

    # Tinta harus terang agar gradien searah dengan bentuk huruf.
    inv = 255.0 - img.astype(np.float32)
    gx, gy = _gradients(inv)

    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0

    bin_w = 180.0 / bins
    pos = ang / bin_w
    lo = np.floor(pos).astype(np.int32) % bins
    frac = (pos - np.floor(pos)).astype(np.float32)
    hi = (lo + 1) % bins

    n_cells = C.CHAR_SIZE // cell
    hist = np.zeros((n_cells, n_cells, bins), dtype=np.float32)

    for cy in range(n_cells):
        for cx in range(n_cells):
            ys = slice(cy * cell, (cy + 1) * cell)
            xs = slice(cx * cell, (cx + 1) * cell)
            m = mag[ys, xs].ravel()
            l = lo[ys, xs].ravel()
            h = hi[ys, xs].ravel()
            f = frac[ys, xs].ravel()
            np.add.at(hist[cy, cx], l, m * (1.0 - f))
            np.add.at(hist[cy, cx], h, m * f)

    parts: list[np.ndarray] = []
    for by in range(n_cells - block + 1):
        for bx in range(n_cells - block + 1):
            blk = hist[by : by + block, bx : bx + block, :].ravel()
            blk = blk / (np.linalg.norm(blk) + 1e-6)
            blk = np.minimum(blk, 0.2)
            blk = blk / (np.linalg.norm(blk) + 1e-6)
            parts.append(blk)

    return np.concatenate(parts).astype(np.float32)


def hog_batch(images: np.ndarray) -> np.ndarray:
    """Hitung HOG untuk banyak karakter sekaligus (N, 32, 32) -> (N, 324)."""
    n = images.shape[0]
    out = np.zeros((n, C.HOG_DIM), dtype=np.float32)
    for i in range(n):
        out[i] = hog(images[i])
    return out


def hog_dims() -> int:
    return C.HOG_DIM


# ---------------------------------------------------------------------------
# Zoning / Grid Features
# ---------------------------------------------------------------------------
def zoning(img: np.ndarray, grid: int = C.ZONING_GRID) -> np.ndarray:
    """Kepadatan tinta per zona pada grid x grid.

    Kanvas karakter dibagi menjadi `grid * grid` sel, lalu tiap sel
    menyumbang satu nilai: rata-rata intensitas piksel tinta, 0..1.

    Konvensi sama dengan [hog]: citra diinversi lebih dulu sehingga tinta
    menjadi terang. Nilai 0 berarti zona kosong, 1 berarti zona tertutup
    tinta penuh.

    Beda peran dengan HOG:
      - HOG  -> ARAH tepi (bentuk goresan); tidak peduli di mana tinta berada
      - Zoning -> SEBARAN tinta; tidak peduli arah, tetapi tahu zona mana
        yang padat dan mana yang kosong

    Contoh: `L` dan `7` bisa punya histogram gradien yang mirip, tetapi
    zonanya berbeda jelas — `L` padat di kolom kiri dan baris bawah,
    `7` padat di baris atas dan kolom kanan.
    """
    if img.ndim == 3:
        img = img.mean(axis=2)
    if img.shape[0] != C.CHAR_SIZE or img.shape[1] != C.CHAR_SIZE:
        raise ValueError(f"Dimensi harus {C.CHAR_SIZE}x{C.CHAR_SIZE}, dapat {img.shape}")

    inv = 255.0 - img.astype(np.float32)
    h, w = inv.shape
    out = np.zeros(grid * grid, dtype=np.float32)

    for r in range(grid):
        y0 = r * h // grid
        y1 = (r + 1) * h // grid
        for c in range(grid):
            x0 = c * w // grid
            x1 = (c + 1) * w // grid
            cell = inv[y0:y1, x0:x1]
            out[r * grid + c] = float(cell.mean()) / 255.0 if cell.size else 0.0

    return out


def zoning_batch(images: np.ndarray, grid: int = C.ZONING_GRID) -> np.ndarray:
    """Hitung zoning untuk banyak karakter (N, 32, 32) -> (N, grid*grid)."""
    n = images.shape[0]
    out = np.zeros((n, grid * grid), dtype=np.float32)
    for i in range(n):
        out[i] = zoning(images[i], grid)
    return out


def combined(img: np.ndarray) -> np.ndarray:
    """Vektor gabungan: HOG ++ Zoning (324 + 16 = 340 dimensi).

    Hasil konkatenasi ini BELUM diskalakan. Penskalaan dilakukan oleh
    StandardScaler yang dilatih di `train_char.py`, karena mean dan
    simpangan baku harus dihitung dari data latih, bukan per citra.
    """
    return np.concatenate([hog(img), zoning(img)]).astype(np.float32)


def combined_batch(images: np.ndarray) -> np.ndarray:
    """Hitung vektor gabungan untuk banyak karakter (N, 32, 32) -> (N, 340)."""
    n = images.shape[0]
    out = np.zeros((n, C.FEATURE_DIM), dtype=np.float32)
    for i in range(n):
        out[i] = combined(images[i])
    return out


def combined_dims() -> int:
    return C.FEATURE_DIM
