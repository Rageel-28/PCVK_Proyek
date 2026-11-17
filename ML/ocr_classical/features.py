"""Ekstraksi fitur HOG (Histogram of Oriented Gradients).

Implementasi ditulis manual -- tanpa scikit-image -- karena versi Dart
nanti harus meniru perhitungan yang sama persis. Rumusnya:

    1. Inversi citra: tinta menjadi terang (255 - piksel)
    2. Gradien: gx = I[:,x+1] - I[:,x-1], gy = I[y+1,:] - I[y-1,:]
    3. Magnitudo m = sqrt(gx^2 + gy^2)
       Orientasi  theta = atan2(gy, gx) dalam derajat, dimodulo 180
    4. Setiap sel 8x8 mengakumulasi m ke 9 bin orientasi
       (interpolasi bilinear ke dua bin terdekat)
    5. Blok 2x2 sel (bertumpuk, stride 1) dinormalisasi L2-Hys
    6. Hasil: 3 x 3 blok x 2 x 2 sel x 9 bin = 324 dimensi

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

    # Fitur spasial tambahan (20 dimensi): kerapatan tinta 4x4 + geometri kotak tinta.
    inv_norm = inv / 255.0
    grid = np.zeros(16, dtype=np.float32)
    idx = 0
    for cy in range(4):
        for cx in range(4):
            grid[idx] = float(inv_norm[cy * 8 : (cy + 1) * 8, cx * 8 : (cx + 1) * 8].mean())
            idx += 1

    ys, xs = np.nonzero(inv_norm > 0.25)
    n_ink = len(xs)
    ink_frac = float(n_ink) / float(C.CHAR_SIZE * C.CHAR_SIZE)
    if n_ink > 0:
        w_frac = float(xs.max() - xs.min() + 1) / float(C.CHAR_SIZE)
        h_frac = float(ys.max() - ys.min() + 1) / float(C.CHAR_SIZE)
        y_center = (float(ys.min() + ys.max()) * 0.5) / float(C.CHAR_SIZE - 1)
    else:
        w_frac = 0.0
        h_frac = 0.0
        y_center = 0.0
    geom = np.array([ink_frac, w_frac, h_frac, y_center], dtype=np.float32)

    parts.append(grid)
    parts.append(geom)
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
