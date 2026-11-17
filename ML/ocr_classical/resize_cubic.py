"""Geometri citra yang identik dengan versi Dart.

Kenapa tidak memakai `cv2`
--------------------------
Paritas Dart vs Python adalah kontrak proyek ini: model yang sama harus
menghasilkan teks yang sama di kedua sisi. Dua operasi OpenCV ternyata
tidak bisa ditiru persis oleh Dart:

- `cv2.INTER_CUBIC` memakai aritmetika fixed-point internal yang berbeda
  di tepi citra.
- `cv2.warpAffine` + `getRotationMatrix2D` memakai pusat `(w/2, h/2)`,
  `cvRound` (setengah ke genap), dan matriks yang mudah salah tanda saat
  ditiru.

Modul ini menjadi **satu-satunya** implementasi resize dan rotasi untuk
pipeline: `ocr_pipeline`/`deskew` memakainya, dan `gray_image.dart`
meniru rumus yang sama baris demi baris.

Resize (sama dengan `resizedBicubic` di Dart):

    fx      = (x + 0.5) * scale - 0.5
    base    = floor(fx), frac = fx - base
    base    dijepit ke [0, n-2]   (frac tetap)
    tap ke-k (k=0..3) membaca piksel base-1+k dengan batas BORDER_REFLECT_101
    bobot   = round(cubic(frac - (k - 1)) * 2048)
    hasil   = (sum + 2^21) >> 22, dijepit ke [0, 255]

Rotasi (sama dengan `rotated` di Dart):

    pusat   = (w/2, h/2)
    src     = R^-1 * (dst - pusat) + pusat
    sampel  = piksel terdekat, pembulatan setengah ke genap
    di luar kanvas -> nilai border
"""

from __future__ import annotations

import math

import numpy as np

COEF_SCALE = 1 << 11
COEF_SHIFT = 22
ROUND_SHIFT = 1 << (COEF_SHIFT - 1)
_CUBIC_A = -0.75


def _cubic_weight(t: float) -> float:
    x = abs(t)
    a = _CUBIC_A
    if x <= 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


def _reflect101(i: int, n: int) -> int:
    if n <= 1:
        return 0
    idx = i
    while idx < 0 or idx >= n:
        idx = -idx if idx < 0 else 2 * (n - 1) - idx
    return idx


def resize_bicubic(img: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """Resize grayscale uint8 dengan bicubic fixed-point (paritas Dart)."""
    src = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = src.shape[:2]
    if new_width == w and new_height == h:
        return src.copy()
    if new_width <= 0 or new_height <= 0:
        return np.zeros((max(new_height, 0), max(new_width, 0)), np.uint8)

    sx_scale = w / new_width
    sy_scale = h / new_height

    # Koefisien horizontal di-cache: tiap kolom tujuan punya 4 tap tetap.
    x_idx = np.zeros((new_width, 4), dtype=np.int64)
    x_coef = np.zeros((new_width, 4), dtype=np.int64)
    for x in range(new_width):
        fx = (x + 0.5) * sx_scale - 0.5
        base = int(np.floor(fx))
        frac = fx - base
        if base < 0:
            base = 0
        if base > w - 2:
            base = max(0, w - 2)
        for k in range(4):
            x_idx[x, k] = _reflect101(base - 1 + k, w)
            x_coef[x, k] = int(round(_cubic_weight(frac - (k - 1)) * COEF_SCALE))

    out = np.zeros((new_height, new_width), np.uint8)
    src_i = src.astype(np.int64)

    for y in range(new_height):
        fy = (y + 0.5) * sy_scale - 0.5
        base_y = int(np.floor(fy))
        frac_y = fy - base_y
        if base_y < 0:
            base_y = 0
        if base_y > h - 2:
            base_y = max(0, h - 2)

        y_idx = np.zeros(4, dtype=np.int64)
        y_coef = np.zeros(4, dtype=np.int64)
        for k in range(4):
            y_idx[k] = _reflect101(base_y - 1 + k, h)
            y_coef[k] = int(round(_cubic_weight(frac_y - (k - 1)) * COEF_SCALE))

        acc = np.zeros(new_width, dtype=np.int64)
        for j in range(4):
            row = src_i[y_idx[j]]
            hsum = (
                row[x_idx[:, 0]] * x_coef[:, 0]
                + row[x_idx[:, 1]] * x_coef[:, 1]
                + row[x_idx[:, 2]] * x_coef[:, 2]
                + row[x_idx[:, 3]] * x_coef[:, 3]
            )
            acc += hsum * y_coef[j]

        vals = (acc + ROUND_SHIFT) >> COEF_SHIFT
        out[y] = np.clip(vals, 0, 255).astype(np.uint8)

    return out

def _py_round_half_even(v: float) -> int:
    """Pembulatan setengah ke genap, sama seperti `round()` di Dart."""
    floor_v = math.floor(v)
    diff = v - floor_v
    if diff < 0.5:
        return int(floor_v)
    if diff > 0.5:
        return int(floor_v) + 1
    return int(floor_v) if int(floor_v) % 2 == 0 else int(floor_v) + 1


def rotate_nearest(img: np.ndarray, degrees: float, border_value: int = 0) -> np.ndarray:
    """Putar citra dengan piksel terdekat; identik dengan `rotated` di Dart."""
    if abs(degrees) < 1e-9:
        return img.copy()
    rad = math.radians(degrees)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    h, w = img.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    dy = (np.arange(h, dtype=np.float64) - cy)[:, None]
    dx = (np.arange(w, dtype=np.float64) - cx)[None, :]
    src_x = cos_a * dx - sin_a * dy + cx
    src_y = sin_a * dx + cos_a * dy + cy

    sx = np.where(src_x >= 0, np.floor(src_x + 0.5), -np.floor(-src_x + 0.5)).astype(np.int64)
    sy = np.where(src_y >= 0, np.floor(src_y + 0.5), -np.floor(-src_y + 0.5)).astype(np.int64)
    valid = (sx >= 0) & (sy >= 0) & (sx < w) & (sy < h)

    out = np.full((h, w), border_value, np.uint8)
    out[valid] = img[sy[valid], sx[valid]]
    return out

def _round_half_away(v: float) -> int:
    """Pembulatan setengah menjauhi nol, sama seperti `double.round()` di Dart."""
    return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))


def resize_bilinear(img: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """Resize bilinear, identik dengan `resized` di Dart.

    Dipakai hanya untuk memperkecil citra saat estimasi sudut deskew, agar
    skor profil yang dibandingkan sama persis di kedua implementasi.
    """
    src = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = src.shape[:2]
    if new_width == w and new_height == h:
        return src.copy()
    if new_width <= 0 or new_height <= 0:
        return np.zeros((max(new_height, 0), max(new_width, 0)), np.uint8)

    sx = w / new_width
    sy = h / new_height
    src_f = src.astype(np.float64)

    fy = (np.arange(new_height, dtype=np.float64) + 0.5) * sy - 0.5
    y0 = np.clip(np.floor(fy).astype(np.int64), 0, h - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wy = np.clip(fy - y0, 0.0, 1.0)[:, None]

    fx = (np.arange(new_width, dtype=np.float64) + 0.5) * sx - 0.5
    x0 = np.clip(np.floor(fx).astype(np.int64), 0, w - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wx = np.clip(fx - x0, 0.0, 1.0)[None, :]

    v00 = src_f[y0[:, None], x0[None, :]]
    v01 = src_f[y0[:, None], x1[None, :]]
    v10 = src_f[y1[:, None], x0[None, :]]
    v11 = src_f[y1[:, None], x1[None, :]]
    top = v00 + (v01 - v00) * wx
    bottom = v10 + (v11 - v10) * wx
    val = np.clip(top + (bottom - top) * wy, 0.0, 255.0)
    return np.floor(val + 0.5).astype(np.uint8)
