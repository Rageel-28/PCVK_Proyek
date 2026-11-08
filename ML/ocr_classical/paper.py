"""Deteksi dan pemotongan area kertas struk.

Foto struk biasanya memuat latar gelap (meja, keyboard, tangan). Bila
latar ikut diproses, bingkai kertas meninggalkan tinta tipis di setiap
baris sehingga segmentasi baris gagal.

Pendekatan (CV klasik, tanpa deep learning):
  1. Ambang Otsu pada kanal kecerahan -> kertas menjadi putih
  2. Connected components -> ambil komponen terbesar
  3. Periksa kelayakan: luas cukup, rasio aspek menyerupai struk
  4. Potong dengan margin kecil
"""

from __future__ import annotations

import cv2
import numpy as np

import config as C


def find_paper_region(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Kembalikan bbox (x, y, w, h) area kertas, atau None bila tak yakin."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Bersihkan bintik kecil sebelum mencari komponen.
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )

    num, _labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if num <= 1:
        return None

    areas = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, num)]
    areas.sort(reverse=True)
    area, idx = areas[0]

    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    bw = int(stats[idx, cv2.CC_STAT_WIDTH])
    bh = int(stats[idx, cv2.CC_STAT_HEIGHT])

    # Kertas harus menutupi sebagian besar gambar.
    if area < 0.20 * h * w:
        return None

    # Bentuk kertas struk: tinggi >= lebar dan tidak terlalu ekstrem.
    aspect = bh / max(bw, 1)
    if aspect < C.PAPER_MIN_ASPECT or aspect > C.PAPER_MAX_ASPECT:
        return None

    fill = area / max(bw * bh, 1)
    if fill < 0.55:
        return None

    # Margin kecil agar tepi kertas tidak terpotong.
    m = int(0.01 * max(bw, bh))
    x0 = max(0, x - m)
    y0 = max(0, y - m)
    x1 = min(w, x + bw + m)
    y1 = min(h, y + bh + m)
    return (x0, y0, x1 - x0, y1 - y0)


def crop_paper(gray: np.ndarray, enabled: bool = True) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Potong area kertas bila terdeteksi; jika tidak, kembalikan aslinya."""
    if not enabled:
        return gray, None
    box = find_paper_region(gray)
    if box is None:
        return gray, None
    x, y, w, h = box
    # Jangan potong bila hasilnya nyaris sama dengan gambar asli.
    if w * h > 0.98 * gray.shape[0] * gray.shape[1]:
        return gray, None
    return gray[y : y + h, x : x + w], box
