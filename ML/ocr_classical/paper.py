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
    t_otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Jika Otsu menyatukan kertas dengan latar terang (>90% bingkai), coba
    # ambang lebih tinggi untuk memisahkan kertas putih dari latar abu-abu.
    thresholds = [int(t_otsu), 195, 205, 185]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for idx_t, t in enumerate(thresholds):
        bright = (blur > t).astype(np.uint8) * 255
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

        num, _labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        if num <= 1:
            continue

        areas = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, num)]
        areas.sort(reverse=True)
        area, idx = areas[0]

        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])

        if idx_t == 0 and bw * bh > 0.90 * h * w:
            continue
        if bw * bh > 0.92 * h * w:
            continue
        if area < 0.15 * h * w:
            continue

        aspect = bh / max(bw, 1)
        if aspect < C.PAPER_MIN_ASPECT or aspect > C.PAPER_MAX_ASPECT:
            continue

        fill = area / max(bw * bh, 1)
        if fill < 0.50:
            continue

        m = int(0.01 * max(bw, bh))
        x0 = max(0, x - m)
        y0 = max(0, y - m)
        x1 = min(w, x + bw + m)
        y1 = min(h, y + bh + m)
        return (x0, y0, x1 - x0, y1 - y0)

    return None


def _clean_paper_borders(crop: np.ndarray) -> np.ndarray:
    """Hilangkan segitiga latar gelap di tepi kertas miring dengan banjir tepi.

    Menjaga keseragaman kertas putih agar tepi kertas yang miring tidak ikut
    terbinarisasi menjadi deretan huruf palsu ('L', 'I', 'U') di pinggir baris.
    """
    h, w = crop.shape
    if h < 8 or w < 8:
        return crop
    hist = np.bincount(crop.ravel(), minlength=256)
    total = h * w
    sum_all = float(np.dot(np.arange(256, dtype=np.float64), hist))
    w_bg = 0.0
    sum_bg = 0.0
    best_var = -1.0
    best_t = 0
    for t in range(256):
        w_bg += float(hist[t])
        if w_bg == 0.0:
            continue
        w_fg = float(total) - w_bg
        if w_fg == 0.0:
            break
        sum_bg += t * float(hist[t])
        m_bg = sum_bg / w_bg
        m_fg = (sum_all - sum_bg) / w_fg
        var = w_bg * w_fg * (m_bg - m_fg) * (m_bg - m_fg)
        if var > best_var:
            best_var = var
            best_t = t

    dark = (crop <= best_t).astype(np.uint8)
    num, labels = cv2.connectedComponents(dark, connectivity=4)
    if num <= 1:
        return crop
    border = set(
        int(v)
        for v in np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    ) - {0}
    if not border:
        return crop

    mask = np.isin(labels, list(border)).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    bg_count = int(mask.sum())
    if bg_count <= 0 or bg_count >= total * 0.40:
        return crop

    rem_hist = np.bincount(crop[mask == 0], minlength=256)
    rem_total = total - bg_count
    target = (rem_total + 1) // 2
    acc = 0
    med = 255
    for v in range(256):
        acc += int(rem_hist[v])
        if acc >= target:
            med = v
            break

    out = crop.copy()
    out[mask > 0] = med
    return out


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
    return _clean_paper_borders(gray[y : y + h, x : x + w]), box
