"""Uji terkontrol: apakah masalahnya RESOLUSI, KONTRAS, atau FONT?

Tiga eksperimen pada citra struk yang sama:
  A. Varies upscale  : 1x, 2x, 4x, 6x  -> apakah teks terbaca bila diperbesar?
  B. Kontras         : CLAHE + normalisasi -> apakah tinta tipis jadi terbaca?
  C. Satu baris saja : crop satu baris, perbesar 8x -> apakah font thermal
                       memang di luar distribusi model?

Bila (A) dan (C) tetap gagal walau resolusi tinggi, penyebabnya bukan
resolusi melainkan DOMAIN FONT: model dilatih pada render font sistem,
sedangkan struk memakai font thermal dot-matrix yang jauh lebih sempit.

Pemakaian:
    python experiment_failure_cause.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import features as F  # noqa: E402
import preprocess as PP  # noqa: E402
import segment as SG  # noqa: E402
from ocr_pipeline import CharClassifier  # noqa: E402
from render_synthetic import normalize_char  # noqa: E402

DIAG = HERE / "data" / "tmp_diag"


def read_ink(gray: np.ndarray, method: str = "sauvola") -> np.ndarray:
    return PP.prepare(gray, method=method)


def ocr_metrics(gray: np.ndarray, clf: CharClassifier, method: str = "sauvola") -> dict:
    """Jalankan hanya tahap fitur+klasifikasi (tanpa crop/deskew) pada array."""
    ink = read_ink(gray, method)
    lines = SG.segment_lines(ink)
    texts, confs = [], []
    for lb in lines:
        chars = []
        for (l, t, r, b) in SG.segment_characters(ink, lb):
            patch = normalize_char(gray, (l, t, r, b))
            ch, cf, _ = clf.predict(F.combined(patch))
            chars.append(ch)
            confs.append(cf)
        if chars:
            texts.append("".join(chars))
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return {"lines": len(lines), "chars": len(confs), "conf": mean_conf,
            "preview": " / ".join(texts[:2])[:64]}


def exp_a_upscale(gray: np.ndarray, clf: CharClassifier, name: str) -> None:
    print(f"\n[A] pengaruh resolusi — {name} (asli {gray.shape[1]}x{gray.shape[0]})")
    for f in (1, 2, 4, 6):
        if f == 1:
            g = gray
        else:
            g = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        r = ocr_metrics(g, clf)
        print(f"  {f}x ({g.shape[1]:5d}px lebar): {r['lines']:3d} baris, "
              f"{r['chars']:4d} char, conf={r['conf'] * 100:5.1f}%  \"{r['preview']}\"")


def exp_b_contrast(gray: np.ndarray, clf: CharClassifier, name: str) -> None:
    print(f"\n[B] pengaruh kontras — {name}")
    variants = {
        "asli": gray,
        "CLAHE": cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray),
        "stretch 1-99%": cv2.normalize(
            gray, None, 0, 255, cv2.NORM_MINMAX,
            dtype=cv2.CV_8U),
    }
    for label, g in variants.items():
        r = ocr_metrics(g, clf)
        print(f"  {label:14s}: {r['lines']:3d} baris, {r['chars']:4d} char, "
              f"conf={r['conf'] * 100:5.1f}%  \"{r['preview']}\"")


def exp_c_single_line(gray: np.ndarray, clf: CharClassifier, name: str) -> None:
    print(f"\n[C] satu baris diperbesar 8x — {name}")
    ink = read_ink(gray, "sauvola")
    lines = SG.segment_lines(ink)
    if not lines:
        print("  tidak ada baris terdeteksi")
        return
    # Ambil baris yang paling banyak glyph
    best = max(lines[:12], key=lambda lb: len(SG.segment_characters(ink, lb)))
    tb = best
    t, b = tb[0], tb[1]
    cols = np.nonzero((ink[t:b + 1, :] > 0).sum(axis=0) > 0)[0]
    if len(cols) == 0:
        print("  baris kosong")
        return
    l, r = int(cols[0]), int(cols[-1])
    crop = gray[t:b + 1, l:r + 1]
    if crop.size == 0:
        print("  crop kosong")
        return
    big = cv2.resize(crop, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    res = ocr_metrics(big, clf)
    print(f"  baris y={t}-{b} diperbesar 8x ({big.shape[1]}x{big.shape[0]}px)")
    print(f"  {res['lines']} baris, {res['chars']} char, conf={res['conf'] * 100:.1f}%")
    print(f"  hasil: \"{res['preview']}\"")


def aspect_report(gray: np.ndarray, name: str) -> None:
    """Rasio aspek glyph thermal vs font latih (Consolas ~0.55)."""
    ink = read_ink(gray, "sauvola")
    ratios = []
    for lb in SG.segment_lines(ink)[:8]:
        for (l, t, r, b) in SG.segment_characters(ink, lb):
            w, hgt = r - l + 1, b - t + 1
            if hgt >= 8:
                ratios.append(w / hgt)
    if ratios:
        print(f"  rasio aspek glyph {name}: median={np.median(ratios):.2f} "
              f"(font latih Consolas ~0.55, Courier ~0.60)")


def main() -> None:
    clf = CharClassifier()
    cases = [("r1_scan_datar.jpg", "scan datar"),
             ("r2_foto_tangan.jpg", "foto di tangan"),
             ("r4_foto_latar.jpg", "foto latar ramai")]

    print("=" * 80)
    for fname, label in cases:
        p = DIAG / fname
        if not p.exists():
            continue
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        aspect_report(gray, fname)
        if fname == "r1_scan_datar.jpg":
            exp_a_upscale(gray, clf, label)
            exp_b_contrast(gray, clf, label)
            exp_c_single_line(gray, clf, label)
        else:
            r = ocr_metrics(gray, clf)
            print(f"  hasil langsung: {r['lines']} baris, {r['chars']} char, "
                  f"conf={r['conf'] * 100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
