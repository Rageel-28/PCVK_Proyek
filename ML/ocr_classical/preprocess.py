"""Pra-pengolahan citra struk: grayscale, denoise, dan binarisasi.

Konvensi keluaran seluruh modul:
    mask uint8, 255 = TINTA, 0 = KERTAS
Konvensi ini dipakai konsisten supaya "kepadatan tinta" intuitif dan
proyeksi baris/karakter tinggal menjumlahkan piksel.

Semua algoritma ditulis dengan operasi dasar (blur kotak, ambang) agar
mudah dipindahkan ke Dart tanpa library pengolahan citra.
"""

from __future__ import annotations

import cv2
import numpy as np

import config as C


def to_gray(image: np.ndarray) -> np.ndarray:
    """Pastikan citra menjadi grayscale uint8."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise(gray: np.ndarray, strength: int = 3) -> np.ndarray:
    """Median blur: menghapus noise titik tanpa mengaburkan tepi huruf."""
    if strength <= 1:
        return gray
    if strength % 2 == 0:
        strength += 1
    return cv2.medianBlur(gray, strength)


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """Otsu global: cepat, baik untuk citra dengan pencahayaan rata."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def binarize_adaptive_mean(gray: np.ndarray, block: int = 31, c: int = 10) -> np.ndarray:
    """Adaptive mean threshold: tahan terhadap bayangan bertahap."""
    if block % 2 == 0:
        block += 1
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, c
    )
    return binary


def binarize_sauvola(
    gray: np.ndarray,
    window: int = C.SAUVOLA_WINDOW,
    k: float = C.SAUVOLA_K,
    r: float = C.SAUVOLA_R,
) -> np.ndarray:
    """Sauvola: ambang lokal berbasis mean dan simpangan baku.

    Rumus:
        T(x,y) = m(x,y) * (1 + k * (s(x,y)/R - 1))
    dengan m = rata-rata lokal, s = simpangan baku lokal.

    Lebih baik daripada Otsu pada struk karena kertas struk sering
    memiliki bayangan dan noda dengan intensitas berbeda-beda.
    """
    if window % 2 == 0:
        window += 1

    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    sq = cv2.boxFilter(g * g, -1, (window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    var = np.maximum(sq - mean * mean, 0.0)
    std = np.sqrt(var)

    threshold = mean * (1.0 + k * (std / r - 1.0))
    ink = (g < threshold).astype(np.uint8) * 255
    return ink


def binarize(gray: np.ndarray, method: str = "sauvola") -> np.ndarray:
    """Pintu masuk tunggal binarisasi."""
    method = method.lower()
    if method == "otsu":
        return binarize_otsu(gray)
    if method in ("adaptive", "mean"):
        return binarize_adaptive_mean(gray)
    if method == "sauvola":
        return binarize_sauvola(gray)
    raise ValueError(f"Metode binarisasi tidak dikenal: {method}")


def remove_small_components(ink: np.ndarray, min_area: int = 4) -> np.ndarray:
    """Buang bintik tinta yang terlalu kecil (sisa noise)."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = np.zeros_like(ink)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def prepare(
    image: np.ndarray,
    method: str = "sauvola",
    median: int = 3,
    clean: bool = True,
) -> np.ndarray:
    """Rangkaian lengkap: grayscale -> denoise -> binarisasi -> bersihkan."""
    gray = to_gray(image)
    gray = denoise(gray, median)
    ink = binarize(gray, method)
    if clean:
        ink = remove_small_components(ink, min_area=C.CHAR_MIN_INK)
    return ink
