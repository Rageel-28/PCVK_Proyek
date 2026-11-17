"""Uji solusi: apakah augmentasi "thermal-like" menyelamatkan klasifikasi?

Hipotesis dari diagnosis:
  - Glyph thermal punya rasio aspek median 0.33, sedangkan font latih 0.55-0.62
    -> thermal ~40% lebih SEMPIT (dot-matrix, karakter dirapatkan)
  - Model hanya pernah melihat font sistem, jadi bentuk thermal = out-of-domain

Uji ini melatih model BARU pada data sintetis yang diberi degradasi meniru
thermal:
  1. kompresi horizontal (0.5-0.75x)  -> meniru glyph sempit
  2. kuantisasi dot-matrix            -> meniru grid titik printer
  3. blur + penurunan kontras         -> meniru tinta pudar
  4. thinning/penebalan acak          -> meniru stroke tidak seragam

Lalu diuji pada baris struk NYATA yang teksnya diketahui, dibandingkan
dengan model lama.

Pemakaian:
    python experiment_thermal_augment.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import features as F  # noqa: E402
import preprocess as PP  # noqa: E402
import segment as SG  # noqa: E402
from ocr_pipeline import CharClassifier  # noqa: E402
from render_synthetic import normalize_char  # noqa: E402

DIAG = HERE / "data" / "tmp_diag"


def dot_matrix(img: np.ndarray, block: int = 3) -> np.ndarray:
    """Kuantisasi ke grid titik, meniru printer thermal."""
    h, w = img.shape
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def thermal_augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()

    # 1. Kompresi horizontal
    squeeze = rng.uniform(0.50, 0.78)
    new_w = max(1, int(out.shape[1] * squeeze))
    squeezed = cv2.resize(out, (new_w, out.shape[0]), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(out, 255)
    x0 = max(0, (out.shape[1] - new_w) // 2)
    canvas[:, x0:x0 + new_w] = squeezed
    out = canvas

    # 2. Dot-matrix
    if rng.random() < 0.8:
        out = dot_matrix(out, block=int(rng.integers(2, 4)))

    # 3. Blur ringan + kontras turun
    if rng.random() < 0.7:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.4, 0.9))
    lo = rng.uniform(0, 40)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.75, 1.0) + lo, 0, 255).astype(np.uint8)

    # 4. Penebalan / penipisan stroke
    if rng.random() < 0.5:
        k = int(rng.integers(1, 3))
        binm = (out < 128).astype(np.uint8) * 255
        binm = cv2.dilate(binm, np.ones((k + 1, k + 1), np.uint8))
        out = 255 - binm

    return out


def load_training(limit: int | None = None):
    d = np.load(C.DATA_DIR / "chars_train.npz")
    X, y = d["X"], d["y"]
    if limit and len(y) > limit:
        idx = np.random.default_rng(0).choice(len(y), limit, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def train(Xtr_img, ytr, seed: int = 42):
    Xtr = F.combined_batch(Xtr_img)
    scaler = StandardScaler().fit(Xtr)
    model = LinearSVC(C=C.SVM_C, max_iter=C.SVM_MAX_ITER, dual=True,
                      random_state=seed)
    model.fit(scaler.transform(Xtr), ytr)
    return model, scaler


def predict_line(model, scaler, gray, line_box, ink) -> tuple[str, float]:
    chars, confs = [], []
    for (l, t, r, b) in SG.segment_characters(ink, line_box):
        norm = normalize_char(gray, (l, t, r, b))
        vec = F.combined(norm)
        vec = (vec - scaler.mean_) / scaler.scale_
        scores = model.coef_ @ vec + model.intercept_
        best = int(np.argmax(scores))
        chars.append(C.IDX_TO_CHAR[int(model.classes_[best])])
        z = scores - scores.max()
        e = np.exp(z)
        confs.append(float(e[best] / e.sum()))
    return "".join(chars), float(np.mean(confs)) if confs else 0.0


def main() -> None:
    print("[1/4] memuat data latih sintetis")
    # Subsampel agar eksperimen cepat; kedua model memakai data yang SAMA.
    X, y = load_training(limit=6000)
    print(f"  {len(y)} sampel, {len(np.unique(y))} kelas")

    print("[2/4] menyiapkan baris uji NYATA dari struk")
    gray = cv2.imread(str(DIAG / "r1_scan_datar.jpg"), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ink = PP.prepare(gray, "sauvola")

    # Baris yang teksnya diketahui (dari gambar struk asli).
    TEST_LINES = [
        ("JAVANA TEH MLATI 350", (622, 668)),
        ("CIDATAR - KAB GARUT", (341, 393)),
        ("HARGA JUAL", (915, 958)),
    ]

    print("[3/4] melatih dua model")
    model_old, sc_old = train(X, y)
    print("  model lama : HOG+Zoning pada data sintetis asli")

    rng = np.random.default_rng(0)
    aug = np.stack([thermal_augment(im, rng) for im in X])
    X_aug = np.concatenate([X, aug])
    y_aug = np.concatenate([y, y])
    model_new, sc_new = train(X_aug, y_aug)
    print(f"  model baru : + augmentasi thermal ({len(y_aug)} sampel)")

    print("[4/4] evaluasi pada baris struk NYATA")
    print("\n%-24s %-34s %s" % ("teks acuan", "model lama", "model +augmentasi"))
    print("-" * 100)
    for truth, (y0, y1) in TEST_LINES:
        t_old, c_old = predict_line(model_old, sc_old, gray, (y0, y1), ink)
        t_new, c_new = predict_line(model_new, sc_new, gray, (y0, y1), ink)

        def sim(a: str, b: str) -> float:
            x = a.replace(" ", "")
            z = b.replace(" ", "")
            n = min(len(x), len(z))
            same = sum(1 for i in range(n) if x[i] == z[i])
            return same / max(len(x), len(z), 1) * 100

        print("%-24s %-34s %s" % (truth, t_old[:32], t_new[:32]))
        print("%-24s conf %4.1f%% sim %3.0f%%            conf %4.1f%% sim %3.0f%%"
              % ("", c_old * 100, sim(truth, t_old),
                 c_new * 100, sim(truth, t_new)))
    print("-" * 100)


if __name__ == "__main__":
    main()
