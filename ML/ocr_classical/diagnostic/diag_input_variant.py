"""Uji cepat: varian masukan glyph mana yang paling terbaca model?

Diagnosis sebelumnya menunjukkan:
  - binarisasi Sauvola JELAS terbaca mata
  - glyph thermal dinormalisasi -> conf ~20%, akurasi karakter ~40%
  - teks sama dengan font latih -> akurasi 100%
Jadi masalahnya di glyph. Uji ini mencari bentuk masukan terbaik:
  A. gray   (sekarang)          -> stroke pudar
  B. biner  (mask tinta)        -> stroke 1px tegas
  C. biner ditebalkan 2px       -> stroke setebal data latih
  D. biner ditebalkan + blur    -> meniru anti-alias font latih

Pemakaian:
    python diag_input_variant.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import features as F  # noqa: E402
import preprocess as PP  # noqa: E402
import segment as SG  # noqa: E402
from ocr_pipeline import CharClassifier  # noqa: E402
from render_synthetic import normalize_char  # noqa: E402

DIAG = HERE / "data" / "tmp_diag"
TESTS = [("JAVANATEHMLATI350", (622, 668)),
         ("CIDATARKABGARUT", (341, 393))]


def norm_from(arr_img, bbox) -> np.ndarray:
    return normalize_char(arr_img, bbox)


def variant(gray, ink, bb, mode: str) -> np.ndarray:
    l, t, r, b = bb
    if mode == "A_gray":
        return norm_from(gray, bb)
    if mode == "B_biner":
        return norm_from(ink, bb)
    sub = ink[t:b + 1, l:r + 1]
    if mode == "C_tebal2":
        up = cv2.dilate(sub, np.ones((2, 2), np.uint8))
        canvas = np.full_like(ink, 0)
        canvas[t:b + 1, l:r + 1] = up
        return norm_from(canvas, bb)
    if mode == "D_tebal_blur":
        up = cv2.dilate(sub, np.ones((2, 2), np.uint8))
        up = cv2.GaussianBlur(up, (3, 3), 0.8)
        canvas = np.full_like(ink, 0)
        canvas[t:b + 1, l:r + 1] = up
        return norm_from(canvas, bb)
    raise ValueError(mode)


def char_accuracy(pred: str, truth: str) -> float:
    n = min(len(pred), len(truth))
    return sum(1 for i in range(n) if pred[i] == truth[i]) / max(len(truth), 1)


def main() -> None:
    clf = CharClassifier()
    gray = cv2.imread(str(DIAG / "r1_scan_datar.jpg"), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ink = PP.prepare(gray, "sauvola")

    modes = ["A_gray", "B_biner", "C_tebal2", "D_tebal_blur"]
    print("=" * 92)
    print(f"{'teks acuan':22s} {'varian':14s} {'prediksi':32s} {'conf':>6s} {'akurasi':>8s}")
    print("=" * 92)

    agg: dict[str, list] = {m: [] for m in modes}
    for truth, box in TESTS:
        boxes = SG.segment_characters(ink, box)
        # Ambil sejumlah glyph pertama sesuai panjang acuan (hindari kolom harga)
        boxes = boxes[:len(truth)]
        for m in modes:
            chars, confs = [], []
            for bb in boxes:
                patch = variant(gray, ink, bb, m)
                ch, cf, _ = clf.predict(F.combined(patch))
                chars.append(ch)
                confs.append(cf)
            pred = "".join(chars)
            acc = char_accuracy(pred, truth)
            agg[m].append((np.mean(confs), acc))
            print(f"{truth:22s} {m:14s} {pred[:32]:32s} "
                  f"{np.mean(confs) * 100:5.1f}% {acc * 100:7.1f}%")
        print("-" * 92)

    print("\nRATA-RATA:")
    for m, rows in agg.items():
        c = np.mean([r[0] for r in rows]) * 100
        a = np.mean([r[1] for r in rows]) * 100
        print(f"  {m:14s} conf={c:5.1f}%  akurasi karakter={a:5.1f}%")
    print("=" * 92)


if __name__ == "__main__":
    main()
