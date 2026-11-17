"""Uji hipotesis: rasio aspek glyph thermal (0.33) vs font latih (0.55).

`normalize_char` memakai skala "contain": mempertahankan rasio aspek lalu
memberi padding putih. Akibatnya glyph thermal yang SEMPIT (0.33) menjadi
huruf tipis di tengah kanvas 32x32 — bentuk yang tidak pernah dilihat model
saat latih.

Uji ini membandingkan tiga cara normalisasi pada baris struk NYATA:
  (a) contain  : perilaku sekarang (dipakai pipeline)
  (b) stretch  : dipaksa mengisi kanvas (rasio aspek diabaikan)
  (c) stretch pada kanvas sempit : kompromi

Bila (b)/(c) jauh lebih baik, solusinya adalah mengganti normalisasi.

Pemakaian:
    python diag_aspect_fix.py
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


def norm_stretch(gray: np.ndarray, bbox, margin: int = 2) -> np.ndarray:
    """Isi kanvas penuh, rasio aspek diabaikan."""
    l, t, r, b = bbox
    crop = gray[t:b + 1, l:r + 1]
    if crop.size == 0:
        return np.full((C.CHAR_SIZE, C.CHAR_SIZE), 255, np.uint8)
    inner = C.CHAR_SIZE - 2 * margin
    resized = np.array(Image.fromarray(crop).resize((inner, inner), Image.BILINEAR))
    canvas = np.full((C.CHAR_SIZE, C.CHAR_SIZE), 255, np.uint8)
    canvas[margin:margin + inner, margin:margin + inner] = resized
    return canvas


def norm_stretch_keep_h(gray: np.ndarray, bbox) -> np.ndarray:
    """Tinggi mengisi kanvas, lebar ditekan ke rasio target 0.55."""
    l, t, r, b = bbox
    crop = gray[t:b + 1, l:r + 1]
    if crop.size == 0:
        return np.full((C.CHAR_SIZE, C.CHAR_SIZE), 255, np.uint8)
    target_w = max(1, int((C.CHAR_SIZE - 4) * 0.55))
    resized = np.array(Image.fromarray(crop).resize(
        (target_w, C.CHAR_SIZE - 4), Image.BILINEAR))
    canvas = np.full((C.CHAR_SIZE, C.CHAR_SIZE), 255, np.uint8)
    ox = (C.CHAR_SIZE - target_w) // 2
    canvas[2:2 + C.CHAR_SIZE - 4, ox:ox + target_w] = resized
    return canvas


def eval_line(clf: CharClassifier, gray, ink, line_box, fn, truth: str):
    chars, confs = [], []
    for bb in SG.segment_characters(ink, line_box):
        patch = fn(gray, bb)
        ch, cf, _ = clf.predict(F.combined(patch))
        chars.append(ch)
        confs.append(cf)
    text = "".join(chars)
    t = truth.replace(" ", "")
    n = min(len(text), len(t))
    sim = sum(1 for i in range(n) if text[i] == t[i]) / max(len(text), len(t), 1)
    return text, float(np.mean(confs)) if confs else 0.0, sim


def main() -> None:
    clf = CharClassifier()
    gray = cv2.imread(str(DIAG / "r1_scan_datar.jpg"), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ink = PP.prepare(gray, "sauvola")

    tests = [
        ("JAVANA TEH MLATI 350", (622, 668)),
        ("CIDATAR - KAB GARUT", (341, 393)),
        ("HARGA JUAL", (915, 958)),
        ("TOTAL", (1036, 1064)),
    ]
    methods = [
        ("contain (sekarang)", normalize_char),
        ("stretch penuh", norm_stretch),
        ("stretch rasio 0.55", norm_stretch_keep_h),
    ]

    print("=" * 96)
    print("%-22s %-4s %-34s %-7s %s" % ("teks acuan", "cara", "prediksi", "conf", "kemiripan"))
    print("=" * 96)
    summary: dict[str, list] = {m[0]: [] for m in methods}
    for truth, box in tests:
        for mname, fn in methods:
            text, conf, sim = eval_line(clf, gray, ink, box, fn, truth)
            summary[mname].append((conf, sim))
            print("%-22s %-4s %-34s %5.1f%% %6.0f%%"
                  % (truth, "", text[:34], conf * 100, sim * 100))
        print("-" * 96)

    print("\nRINGKASAN rata-rata:")
    for mname, rows in summary.items():
        conf = np.mean([r[0] for r in rows]) * 100
        sim = np.mean([r[1] for r in rows]) * 100
        print(f"  {mname:24s} conf={conf:5.1f}%  kemiripan={sim:5.1f}%")
    print("=" * 96)


if __name__ == "__main__":
    main()
