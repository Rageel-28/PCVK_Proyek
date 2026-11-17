"""Uji solusi: menambah font bergaya thermal ke data latih.

Diagnosis: glyph thermal tidak cocok dengan font latih (korelasi <= 0.54).
Font yang PALING mendekati tetapi belum dipakai: tahomabd, msgothic,
verdanab, arialbd.

Uji ini membandingkan tiga model pada baris struk NYATA:
  1. baseline  : font latih sekarang (6 font)
  2. +bold     : tambah tahomabd, msgothic, verdanab, arialbd
  3. +bold+aug : tambahan di atas + degradasi thermal (kompresi, dot-matrix)

Pemakaian:
    python experiment_add_fonts.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import features as F  # noqa: E402
import preprocess as PP  # noqa: E402
import segment as SG  # noqa: E402
from render_synthetic import load_font, normalize_char  # noqa: E402

DIAG = HERE / "data" / "tmp_diag"
NEW_FONTS = ["tahomabd.ttf", "msgothic.ttc", "verdanab.ttf", "arialbd.ttf"]
TRAIN_FONTS = [f for f in C.FONTS if (C.FONT_DIR / f).exists()]

# Baris struk nyata dengan teks yang sudah dibaca manual dari citra
TESTS = [
    ("JAVANATEHMLATI350", (622, 668)),
    ("CIDATARKABGARUT", (291, 333)),
    ("HARGAJUAL", (915, 958)),
    ("TOTALBELANJA", (1036, 1064)),
]


def render_char(ch: str, fname: str, size: int) -> np.ndarray | None:
    try:
        f = load_font(fname, size)
    except Exception:  # noqa: BLE001
        return None
    probe = Image.new("L", (10, 10), 255)
    w = ImageDraw.Draw(probe).textlength(ch, font=f)
    ascent, descent = f.getmetrics()
    if w <= 0 or ascent <= 0:
        return None
    img = Image.new("L", (int(w) + 16, ascent + descent + 8), 255)
    ImageDraw.Draw(img).text((8, 4), ch, font=f, fill=0)
    a = np.asarray(img)
    ys, xs = np.nonzero(a < 128)
    if len(xs) == 0:
        return None
    return normalize_char(a, (int(xs.min()), int(ys.min()),
                              int(xs.max()), int(ys.max())))


def build_set(fonts: list[str], per_char: int = 60, seed: int = 0) -> tuple:
    """Bangkitkan glyph sintetis untuk semua karakter."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ch in C.CHARSET:
        got = 0
        for _ in range(per_char * 3):
            if got >= per_char:
                break
            fname = fonts[int(rng.integers(0, len(fonts)))]
            size = int(rng.choice(C.RENDER_SIZES))
            g = render_char(ch, fname, size)
            if g is None:
                continue
            X.append(g)
            y.append(C.CHAR_TO_IDX[ch])
            got += 1
    return np.stack(X).astype(np.uint8), np.array(y, np.int32)


def thermal_augment(img: np.ndarray, rng) -> np.ndarray:
    out = img.copy()
    sq = rng.uniform(0.55, 0.82)
    nw = max(1, int(out.shape[1] * sq))
    small = cv2.resize(out, (nw, out.shape[0]), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(out, 255)
    x0 = max(0, (out.shape[1] - nw) // 2)
    canvas[:, x0:x0 + nw] = small
    out = canvas
    if rng.random() < 0.85:
        b = int(rng.integers(2, 4))
        h, w = out.shape
        s = cv2.resize(out, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)
        out = cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST)
    if rng.random() < 0.7:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.3, 0.8))
    return np.clip(out.astype(np.float32) * rng.uniform(0.8, 1.0),
                   0, 255).astype(np.uint8)


def train(X_img, y):
    X = F.combined_batch(X_img)
    sc = StandardScaler().fit(X)
    m = LinearSVC(C=C.SVM_C, max_iter=5000, dual=True, random_state=42)
    m.fit(sc.transform(X), y)
    return m, sc


def eval_line(model, sc, gray, ink, box, truth) -> tuple[str, float, float]:
    boxes = SG.segment_characters(ink, box)[:len(truth)]
    chars, confs = [], []
    for bb in boxes:
        vec = F.combined(normalize_char(gray, bb))
        vec = (vec - sc.mean_) / sc.scale_
        scores = model.coef_ @ vec + model.intercept_
        best = int(np.argmax(scores))
        chars.append(C.IDX_TO_CHAR[int(model.classes_[best])])
        z = scores - scores.max()
        e = np.exp(z)
        confs.append(float(e[best] / e.sum()))
    pred = "".join(chars)
    n = min(len(pred), len(truth))
    acc = sum(1 for i in range(n) if pred[i] == truth[i]) / max(len(truth), 1)
    return pred, float(np.mean(confs)) if confs else 0.0, acc


def main() -> None:
    gray = cv2.imread(str(DIAG / "r1_scan_datar.jpg"), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ink = PP.prepare(gray, "sauvola")

    print("[1/2] membangun data latih")
    # Baseline: gunakan data npz yang sudah ada (subset agar sebanding)
    d = np.load(C.DATA_DIR / "chars_train.npz")
    idx = np.random.default_rng(0).choice(len(d["y"]), 9000, replace=False)
    X_base, y_base = d["X"][idx], d["y"][idx]

    X_new, y_new = build_set(TRAIN_FONTS + NEW_FONTS, per_char=60)
    print(f"  baseline {len(y_base)} sampel (font lama)")
    print(f"  font baru {len(y_new)} sampel ({len(TRAIN_FONTS) + len(NEW_FONTS)} font)")

    print("[2/2] melatih & mengevaluasi")
    models = {}
    t0 = time.time()
    models["baseline"] = train(X_base, y_base)
    print(f"  baseline selesai {time.time() - t0:.0f}s")

    t0 = time.time()
    models["+bold fonts"] = train(X_new, y_new)
    print(f"  +bold selesai {time.time() - t0:.0f}s")

    rng = np.random.default_rng(1)
    Xa = np.concatenate([X_new, np.stack([thermal_augment(i, rng) for i in X_new])])
    ya = np.concatenate([y_new, y_new])
    t0 = time.time()
    models["+bold+aug"] = train(Xa, ya)
    print(f"  +bold+aug selesai {time.time() - t0:.0f}s ({len(ya)} sampel)\n")

    print("=" * 100)
    print("%-18s %-14s %-28s %-7s %s" % ("teks acuan", "model", "prediksi", "conf", "akurasi"))
    print("=" * 100)
    agg = {k: [] for k in models}
    for truth, box in TESTS:
        for name, (m, sc) in models.items():
            pred, conf, acc = eval_line(m, sc, gray, ink, box, truth)
            agg[name].append((conf, acc))
            print("%-18s %-14s %-28s %5.1f%% %6.1f%%"
                  % (truth, name, pred[:28], conf * 100, acc * 100))
        print("-" * 100)

    print("\nRATA-RATA:")
    for name, rows in agg.items():
        print("  %-14s conf=%5.1f%%  akurasi karakter=%5.1f%%"
              % (name, np.mean([r[0] for r in rows]) * 100,
                 np.mean([r[1] for r in rows]) * 100))
    print("=" * 100)


if __name__ == "__main__":
    main()
