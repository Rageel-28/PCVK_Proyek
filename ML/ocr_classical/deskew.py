"""Estimasi dan koreksi kemiringan (deskew) berbasis profil proyeksi.

Ide: bila teks tegak, jumlah piksel tinta per baris membentuk pola
"puncak dan lembah" yang tajam. Semakin miring, pola itu semakin kabur.
Maka sudut terbaik adalah sudut yang memaksimalkan variansi profil
horizontal.

Alternatif yang disediakan untuk pembanding: minAreaRect dari piksel tinta.
"""

from __future__ import annotations

import cv2
import numpy as np


def _rotate(ink: np.ndarray, angle: float, border: int = 0) -> np.ndarray:
    h, w = ink.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        ink, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=border
    )


def _profile_score(ink: np.ndarray, angle: float) -> float:
    rotated = _rotate(ink, angle)
    profile = (rotated > 0).sum(axis=1).astype(np.float32)
    return float(profile.var())


def _downscale(ink: np.ndarray, longest: int = 700) -> tuple[np.ndarray, float]:
    h, w = ink.shape
    m = max(h, w)
    if m <= longest:
        return ink, 1.0
    scale = longest / m
    return (
        cv2.resize(ink, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA),
        scale,
    )


def estimate_skew(
    ink: np.ndarray,
    max_angle: float = 10.0,
    coarse_step: float = 1.0,
    fine_step: float = 0.1,
) -> float:
    """Kembalikan sudut (derajat) yang harus diterapkan agar teks tegak."""
    if ink.size == 0 or not ink.any():
        return 0.0

    small, _ = _downscale(ink)
    best_angle = 0.0
    best_score = -1.0

    for a in np.arange(-max_angle, max_angle + 1e-9, coarse_step):
        s = _profile_score(small, float(a))
        if s > best_score:
            best_score, best_angle = s, float(a)

    lo = best_angle - coarse_step
    hi = best_angle + coarse_step
    for a in np.arange(lo, hi + 1e-9, fine_step):
        s = _profile_score(small, float(a))
        if s > best_score:
            best_score, best_angle = s, float(a)

    return best_angle


def estimate_skew_minarea(ink: np.ndarray) -> float:
    """Pembanding: sudut dari kotak terkecil yang membungkus piksel tinta."""
    ys, xs = np.nonzero(ink)
    if len(xs) < 50:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return float(angle)


def deskew(ink: np.ndarray, angle: float) -> np.ndarray:
    """Terapkan rotasi koreksi; tidak memperbesar kanvas (sudut kecil)."""
    if abs(angle) < 1e-6:
        return ink
    return _rotate(ink, angle)


def rotate_like(image: np.ndarray, angle: float, border: int = 255) -> np.ndarray:
    """Putar citra grayscale dengan sudut yang sama seperti mask tinta."""
    if abs(angle) < 1e-6:
        return image
    return _rotate(image, angle, border=border)


def auto_deskew(
    ink: np.ndarray,
    max_angle: float = 10.0,
    method: str = "profile",
) -> tuple[np.ndarray, float]:
    """Estimasi lalu koreksi. Kembalikan (citra, sudut terpakai)."""
    if method == "minarea":
        angle = estimate_skew_minarea(ink)
    else:
        angle = estimate_skew(ink, max_angle=max_angle)
    if abs(angle) > max_angle:
        angle = 0.0
    return deskew(ink, angle), angle


def auto_deskew_pair(
    gray: np.ndarray,
    ink: np.ndarray,
    max_angle: float = 10.0,
    method: str = "profile",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Koreksi sekaligus pada grayscale dan mask tinta.

    Keduanya harus memakai sudut yang sama supaya bbox hasil segmentasi
    pada mask tetap valid ketika dipakai memotong grayscale.
    """
    if method == "minarea":
        angle = estimate_skew_minarea(ink)
    else:
        angle = estimate_skew(ink, max_angle=max_angle)
    if abs(angle) > max_angle:
        angle = 0.0
    if abs(angle) < 1e-6:
        return gray, ink, 0.0
    return rotate_like(gray, angle), deskew(ink, angle), angle
