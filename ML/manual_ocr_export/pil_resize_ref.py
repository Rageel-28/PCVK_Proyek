"""Implementasi ulang resampler PIL (BILINEAR & LANCZOS) untuk diverifikasi.

Pipeline OCR manual memakai `PIL.Image.resize` saat:
  - menormalkan glyph ke 64x48 (BILINEAR) sebelum ekstraksi fitur 172 dim
  - memperbesar citra kecil (LANCZOS) pada upscale adaptif

Port Dart harus meniru PIL, bukan interpolasi bilinear generik. Skrip ini
membuktikan bahwa algoritma di bawah identik dengan PIL (selisih 0) untuk
berbagai rasio skala, sehingga aman diporting ke `lib/ocr/manual/pil_resize.dart`.

Pemakaian:
    python pil_resize_ref.py
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image


def _bilinear(x: float) -> float:
    x = abs(x)
    if x < 1.0:
        return 1.0 - x
    return 0.0


def _lanczos(x: float) -> float:
    x = abs(x)
    if x < 3.0:
        if x < 1e-12:
            return 1.0
        return (math.sin(math.pi * x) * math.sin(math.pi * x / 3.0)
                / (math.pi * math.pi * x * x / 3.0))
    return 0.0


FILTERS = {"bilinear": (_bilinear, 1.0), "lanczos": (_lanczos, 3.0)}


def _coefficients(in_size: int, out_size: int, filt, support: float):
    """Koefisien seperti `precompute_coeffs` di Pillow Resample.c."""
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support_s = support * filterscale
    ss = 1.0 / filterscale

    out = []
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = int(center - support_s + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + support_s + 0.5)
        if xmax > in_size:
            xmax = in_size

        weights = []
        total = 0.0
        for x in range(xmin, xmax):
            w = filt((x - center + 0.5) * ss)
            weights.append(w)
            total += w
        if total != 0.0:
            # Pillow menormalkan dengan mengalikan 1/ww (bukan membagi),
            # dan pembulatan float inilah yang memengaruhi hasil 1 LSB.
            inv = 1.0 / total
            weights = [w * inv for w in weights]
        out.append((xmin, weights))
    return out


def _clip8(value: int) -> int:
    if value >= 255:
        return 255
    if value < 0:
        return 0
    return value


# Pillow menyimpan koefisien sebagai fixed-point: PRECISION_BITS = 32 - 8 - 2.
PRECISION_BITS = 22


def _quantize(weights: list[float]) -> list[int]:
    scale = 1 << PRECISION_BITS
    return [int(round(w * scale)) for w in weights]


def resize(arr: np.ndarray, out_w: int, out_h: int, name: str = "bilinear") -> np.ndarray:
    """Resize uint8 2D dua lintasan (horizontal lalu vertikal), seperti PIL.

    Perbedaan penting terhadap interpolasi bilinear biasa: koefisien
    dikuantisasi ke fixed-point 22 bit, akumulasi memakai integer, lalu
    dibulatkan dengan `+ (1 << 21)` sebelum digeser. Tanpa ini hasil
    berbeda 1 LSB pada sebagian piksel, dan fitur grid 12x12 ikut bergeser.
    """
    filt, support = FILTERS[name]
    h, w = arr.shape
    half = 1 << (PRECISION_BITS - 1)

    # Lintasan horizontal: hasil antara disimpan sebagai uint8 (PIL juga).
    tmp = np.zeros((h, out_w), np.uint8)
    for xx, (xmin, weights) in enumerate(_coefficients(w, out_w, filt, support)):
        ki = _quantize(weights)
        for y in range(h):
            acc = 0
            for i, k in enumerate(ki):
                acc += k * int(arr[y, xmin + i])
            tmp[y, xx] = _clip8((acc + half) >> PRECISION_BITS)

    out = np.zeros((out_h, out_w), np.uint8)
    for yy, (ymin, weights) in enumerate(_coefficients(h, out_h, filt, support)):
        ki = _quantize(weights)
        for x in range(out_w):
            acc = 0
            for i, k in enumerate(ki):
                acc += k * int(tmp[ymin + i, x])
            out[yy, x] = _clip8((acc + half) >> PRECISION_BITS)

    return out


def compare() -> None:
    rng = np.random.default_rng(0)
    cases = [
        ((60, 40), (48, 64), "bilinear"),    # glyph 60x40 -> 64x48 (upscale)
        ((120, 80), (48, 64), "bilinear"),   # downscale
        ((17, 9), (48, 64), "bilinear"),     # upscale besar
        ((41, 27), (48, 64), "bilinear"),    # rasio ganjil
        ((300, 200), (48, 64), "bilinear"),  # downscale besar
        ((20, 14), (120, 60), "lanczos"),    # upscale adaptif
        ((640, 120), (2600, 488), "lanczos"),
        ((33, 21), (64, 48), "lanczos"),
    ]

    print(f"{'input':>10} {'output':>10} {'filter':>9} {'max diff':>9}")
    print("-" * 44)
    worst = 0
    for (in_w, in_h), (out_w, out_h), name in cases:
        arr = rng.integers(0, 256, size=(in_h, in_w), dtype=np.uint8)
        mine = resize(arr, out_w, out_h, name)
        pil = np.array(
            Image.fromarray(arr).resize((out_w, out_h), getattr(Image, name.upper()))
        )
        diff = int(np.abs(mine.astype(np.int16) - pil.astype(np.int16)).max())
        worst = max(worst, diff)
        print(f"{in_w}x{in_h:<5} {out_w}x{out_h:<5} {name:>9} {diff:>9}")

    print("-" * 44)
    print("selisih maksimum:", worst)
    if worst == 0:
        print("IDENTIK dengan PIL untuk semua kasus.")
    else:
        print("BELUM identik — jangan diporting apa adanya.")


if __name__ == "__main__":
    compare()
