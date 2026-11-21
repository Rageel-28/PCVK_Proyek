"""Ekstraksi fitur karakter: HOG 324-d + zoning 48-d + proyeksi 24-d.

Semua diimplementasikan manual dengan numpy — bukan ``skimage.hog`` —
supaya rumusnya bisa diport 1:1 ke Dart saat inferensi di perangkat.

Total 396 dimensi:

============  ====  ==================================================
Blok          Dim   Isi
============  ====  ==================================================
HOG           324   gradien 9 bin, sel 8x8, blok 2x2 stride 1, L2-norm
Zoning         48   grid 4x4 x (densitas, sentroid-x, sentroid-y)
Proyeksi       24   distribusi tinta 16 kolom + 8 baris
============  ====  ==================================================
"""

from __future__ import annotations

import cv2
import numpy as np

import ocr.config as C

_EPS = 1e-6

# Bin tengah: 10, 30, 50, ... derajat (gradien tak bertanda, 0-180).
_ORI_BIN_WIDTH = 180.0 / C.HOG_BINS


def _gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradien gx, gy dengan beda tengah (portabel ke Dart).

    Di tepi citra dipakai beda satu sisi, sama seperti ``np.gradient``.
    """
    f = img.astype(np.float32) / 255.0
    gy, gx = np.gradient(f)
    return gx.astype(np.float32), gy.astype(np.float32)


def hog_features(char: np.ndarray) -> np.ndarray:
    """Histogram of Oriented Gradients manual, 324 dimensi.

    Langkah:
        1. gradien gx, gy (beda tengah)
        2. magnitudo + orientasi tak bertanda 0-180 derajat
        3. histogram 9 bin per sel 8x8 dengan interpolasi bilinear orientasi
        4. normalisasi L2 per blok 2x2 sel, stride 1 -> 3x3 blok
    """
    img = np.asarray(char, dtype=np.uint8)
    if img.shape != (C.CHAR_SIZE, C.CHAR_SIZE):
        raise ValueError(
            f"dimensi karakter harus {C.CHAR_SIZE}x{C.CHAR_SIZE}, dapat {img.shape}"
        )

    gx, gy = _gradients(img)
    magnitude = np.sqrt(gx * gx + gy * gy)

    # Orientasi tak bertanda: 0-180 derajat.
    orientation = np.degrees(np.arctan2(gy, gx)) % 180.0

    # Posisi bin pecahan, lalu bobot interpolasi ke dua bin tetangga.
    position = orientation / _ORI_BIN_WIDTH
    lower = np.floor(position)
    upper_weight = (position - lower).astype(np.float32)
    lower_weight = 1.0 - upper_weight
    bin0 = (lower.astype(np.int32)) % C.HOG_BINS
    bin1 = (bin0 + 1) % C.HOG_BINS

    # Indeks sel datar (0..15) untuk tiap piksel.
    cells_per_side = C.CHAR_SIZE // C.HOG_CELL           # 4
    yy, xx = np.indices(img.shape, dtype=np.int32)
    cell_flat = (yy // C.HOG_CELL) * cells_per_side + (xx // C.HOG_CELL)
    cell_flat = cell_flat.ravel()

    n_cells = cells_per_side * cells_per_side
    idx0 = cell_flat * C.HOG_BINS + bin0.ravel()
    idx1 = cell_flat * C.HOG_BINS + bin1.ravel()
    w0 = (magnitude * lower_weight).ravel()
    w1 = (magnitude * upper_weight).ravel()

    hist = np.bincount(
        idx0, weights=w0, minlength=n_cells * C.HOG_BINS
    ) + np.bincount(idx1, weights=w1, minlength=n_cells * C.HOG_BINS)
    hist = hist.reshape(cells_per_side, cells_per_side, C.HOG_BINS).astype(np.float32)

    # Normalisasi L2 per blok 2x2 sel, stride 1.
    blocks_per_side = cells_per_side - C.HOG_BLOCK + 1     # 3
    out = np.empty(
        blocks_per_side * blocks_per_side * C.HOG_BLOCK ** 2 * C.HOG_BINS,
        dtype=np.float32,
    )
    pos = 0
    for by in range(blocks_per_side):
        for bx in range(blocks_per_side):
            block = hist[by : by + C.HOG_BLOCK, bx : bx + C.HOG_BLOCK, :].ravel()
            norm = np.sqrt(np.sum(block * block) + _EPS * _EPS)
            out[pos : pos + block.size] = block / norm
            pos += block.size
    return out


def zoning_features(char: np.ndarray) -> np.ndarray:
    """Fitur zoning 48-d: grid 4x4 x (densitas, sentroid-x, sentroid-y).

    Sentroid diukur relatif di dalam selnya (0..1), jadi invarian terhadap
    posisi sel di kanvas.
    """
    img = np.asarray(char, dtype=np.uint8).astype(np.float32) / 255.0
    if img.shape != (C.CHAR_SIZE, C.CHAR_SIZE):
        raise ValueError(
            f"dimensi karakter harus {C.CHAR_SIZE}x{C.CHAR_SIZE}, dapat {img.shape}"
        )

    grid = C.ZONING_GRID
    cell = C.CHAR_SIZE // grid
    # (baris_grid, baris_sel, kolom_grid, kolom_sel)
    blocks = img.reshape(grid, cell, grid, cell)

    density = blocks.mean(axis=(1, 3))

    # Sentroid lokal: koordinat 0..1 di dalam sel.
    coords = np.arange(cell, dtype=np.float32) / max(1, cell - 1)
    ink = blocks.sum(axis=(1, 3)) + _EPS

    # cx: rata-rata koordinat-x berbobot tinta, per (baris_grid, kolom_grid).
    cx = np.einsum("ijkl,l->ik", blocks, coords) / ink
    # cy: rata-rata koordinat-y berbobot tinta.
    cy = np.einsum("ijkl,j->ik", blocks, coords) / ink

    out = np.empty((grid, grid, C.ZONING_STATS), dtype=np.float32)
    out[:, :, 0] = density
    out[:, :, 1] = cx
    out[:, :, 2] = cy
    return out.ravel()


def projection_features(char: np.ndarray) -> np.ndarray:
    """Distribusi tinta 16 kolom + 8 baris, dinormalisasi agar berjumlah 1."""
    img = np.asarray(char, dtype=np.uint8).astype(np.float32) / 255.0
    if img.shape != (C.CHAR_SIZE, C.CHAR_SIZE):
        raise ValueError(
            f"dimensi karakter harus {C.CHAR_SIZE}x{C.CHAR_SIZE}, dapat {img.shape}"
        )

    col_width = C.CHAR_SIZE // C.PROJECTION_COLS     # 2
    row_height = C.CHAR_SIZE // C.PROJECTION_ROWS    # 4

    cols = img.reshape(C.CHAR_SIZE, C.PROJECTION_COLS, col_width).sum(axis=(0, 2))
    rows = img.reshape(C.PROJECTION_ROWS, row_height, C.CHAR_SIZE).sum(axis=(1, 2))

    cols = cols / (cols.sum() + _EPS)
    rows = rows / (rows.sum() + _EPS)
    return np.concatenate([cols, rows]).astype(np.float32)


def hole_features(char: np.ndarray) -> np.ndarray:
    """Fitur topologi 8-d: jumlah lubang, luas relatif, dan posisinya.

    Lubang didefinisikan sebagai daerah kertas yang tertutup tinta. Caranya:
    binerkan, beri bingkai kertas, lalu flood-fill dari luar; daerah kertas
    yang tidak terjangkau flood-fill adalah lubang.

    Fitur ini yang membedakan ``0`` (dua lubang, terbelah garis miring) dari
    ``O`` (satu lubang), dan ``8`` dari ``B``. Tanpa fitur topologi, kedua
    pasangan itu nyaris mustahil dipisahkan oleh HOG saja.
    """
    img = np.asarray(char, dtype=np.uint8)
    if img.shape != (C.CHAR_SIZE, C.CHAR_SIZE):
        raise ValueError(
            f"dimensi karakter harus {C.CHAR_SIZE}x{C.CHAR_SIZE}, dapat {img.shape}"
        )

    ink = (img > 127).astype(np.uint8)
    ink_total = int(ink.sum())
    if ink_total == 0:
        return np.zeros(C.HOLE_DIM, dtype=np.float32)

    padded = np.pad(ink, 1, constant_values=0)
    # Beri label daerah KERTAS (bukan tinta): 1 - padded membuat kertas = 1.
    # Lubang = daerah kertas yang tidak menyentuh bingkai luar.
    n_labels, labels = cv2.connectedComponents(1 - padded, connectivity=4)

    border_labels = np.unique(
        np.concatenate(
            [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
        )
    )
    is_outside = np.zeros(n_labels, dtype=bool)
    is_outside[border_labels] = True

    holes: list[tuple[int, float, float]] = []
    for label in range(1, n_labels):
        if is_outside[label]:
            continue
        mask = labels == label
        area = int(mask.sum())
        if area <= 0:
            continue
        ys, xs = np.nonzero(mask)
        holes.append((area, float(xs.mean()), float(ys.mean())))

    holes.sort(key=lambda h: h[0], reverse=True)

    feats = np.zeros(C.HOLE_DIM, dtype=np.float32)
    feats[0] = min(len(holes), C.HOLE_MAX) / float(C.HOLE_MAX)
    for i in range(C.HOLE_MAX):
        if i < len(holes):
            feats[1 + i] = holes[i][0] / float(ink_total)
    for i in range(2):
        base = 1 + C.HOLE_MAX + 2 * i
        if i < len(holes):
            feats[base] = holes[i][1] / (C.CHAR_SIZE - 1)
            feats[base + 1] = holes[i][2] / (C.CHAR_SIZE - 1)
    return feats


def extract_features(char: np.ndarray) -> np.ndarray:
    """Vektor fitur 1844-d untuk satu karakter 32x32."""
    return np.concatenate(
        [
            hog_features(char),
            zoning_features(char),
            projection_features(char),
            hole_features(char),
        ]
    ).astype(np.float32)


def extract_features_batch(images: np.ndarray) -> np.ndarray:
    """Vektor fitur untuk banyak karakter sekaligus.

    Args:
        images: ``(N, 32, 32)`` uint8.

    Returns:
        ``(N, 1844)`` float32.
    """
    arr = np.asarray(images)
    if arr.ndim != 3 or arr.shape[1:] != (C.CHAR_SIZE, C.CHAR_SIZE):
        raise ValueError(
            f"citra harus (N, {C.CHAR_SIZE}, {C.CHAR_SIZE}), dapat {arr.shape}"
        )
    out = np.empty((len(arr), C.FEATURE_DIM), dtype=np.float32)
    for i in range(len(arr)):
        out[i] = extract_features(arr[i])
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Ekstrak fitur dari dataset karakter.")
    ap.add_argument("--npz", type=str, default=str(C.DATA_DIR / "chars.npz"))
    ap.add_argument("--out", type=str, default=str(C.OUTPUT_DIR / "features.npz"))
    args = ap.parse_args()

    data = np.load(args.npz)
    X, y, text_index = data["X"], data["y"], data["text_index"]
    variant = data["variant"] if "variant" in data else None
    print(f"{len(X)} sampel -> fitur {C.FEATURE_DIM}-d ...")

    F = extract_features_batch(X)
    print(f"bentuk fitur: {F.shape}  memori: {F.nbytes / 1e6:.1f} MB")

    payload = {"F": F, "y": y, "text_index": text_index}
    if variant is not None:
        payload["variant"] = variant
    np.savez_compressed(args.out, **payload)
    print(f"disimpan ke {args.out}")


if __name__ == "__main__":
    main()
