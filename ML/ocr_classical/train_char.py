"""Latih pengklasifikasi karakter (HOG + Linear SVM).

Model disimpan dalam dua bentuk:
  models/char_svm.joblib  -> untuk eksperimen Python
  models/char_svm.json    -> untuk dipakai aplikasi Flutter

Ekspor JSON mengikuti pola yang sudah dipakai model kategori:
bobot linear (coef + intercept) ditulis apa adanya, lalu aplikasi
cukup menghitung argmax(coef @ x + intercept).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import config as C
import features as F


def _augment_patch(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augmentasi citra klasik ringan pada patch asli 32x32 (geser & gamma)."""
    out = patch.astype(np.float32)
    dx = int(rng.integers(-1, 2))
    dy = int(rng.integers(-1, 2))
    if dx != 0 or dy != 0:
        shifted = np.full_like(out, 255.0)
        sx0 = max(0, -dx)
        sx1 = C.CHAR_SIZE - max(0, dx)
        sy0 = max(0, -dy)
        sy1 = C.CHAR_SIZE - max(0, dy)
        tx0 = max(0, dx)
        tx1 = C.CHAR_SIZE - max(0, -dx)
        ty0 = max(0, dy)
        ty1 = C.CHAR_SIZE - max(0, -dy)
        shifted[ty0:ty1, tx0:tx1] = out[sy0:sy1, sx0:sx1]
        out = shifted

    if rng.random() < 0.7:
        gamma = float(rng.uniform(0.75, 1.30))
        out = 255.0 * (np.clip(out, 0.0, 255.0) / 255.0) ** gamma

    return np.clip(out, 0, 255).astype(np.uint8)


def load_split(val_ratio: float, seed: int, with_synthetic: bool = False):
    """Muat data latih dari potongan karakter struk asli (`chars_real_gold.npz`).

    Secara default TIDAK memakai data sintetis; seluruh sampel berasal dari
    potongan karakter struk asli di `ML/Dataset Citra/`. Augmentasi klasik
    ringan diterapkan HANYA pada split latih (`Xtr_img`) untuk menyeimbangkan
    kelas yang jumlah sampel aslinya sedikit.
    """
    real_path = C.DATA_DIR / "chars_real_gold.npz"
    if not real_path.exists():
        raise FileNotFoundError(
            f"{real_path} belum ada. Jalankan dulu: python build_real_chars.py"
        )
    dr = np.load(real_path)
    X_real, y_real = dr["X"], dr["y"]
    if len(y_real) == 0:
        raise RuntimeError("chars_real_gold.npz kosong.")

    rng = np.random.default_rng(seed)
    counts = np.bincount(y_real, minlength=len(C.CHARSET))

    # Kelas dengan >= 4 sampel di-split ke latih + validasi; kelas sangat langka
    # (1..3 sampel) masuk ke latih agar tetap dikenal tanpa membuangnya.
    splittable = np.isin(y_real, np.nonzero(counts >= 4)[0])
    if splittable.any():
        X_sp, y_sp = X_real[splittable], y_real[splittable]
        # Pastikan cukup sampel validasi per kelas (>= 1 per kelas).
        Xtr_img, Xva_img, ytr, yva = train_test_split(
            X_sp, y_sp, test_size=val_ratio, random_state=seed, stratify=y_sp
        )
    else:
        Xtr_img, Xva_img = X_real[:0], X_real[:0]
        ytr, yva = y_real[:0], y_real[:0]

    rare_mask = ~splittable
    if rare_mask.any():
        Xtr_img = np.concatenate([Xtr_img, X_real[rare_mask]], axis=0)
        ytr = np.concatenate([ytr, y_real[rare_mask]], axis=0)

    # Augmentasi HANYA pada data latih asli untuk menyeimbangkan kelas langka
    # dan memperkaya variasi ketebalan tinta / pergeseran 1 px. Tanda baca
    # berlebih (`.` dan `,`) dibatasi agar blok buram resolusi rendah tidak
    # mendominasi huruf kecil; simbol sangat langka (`%`, `+`, dll.) tidak
    # digandakan paksa hingga 80-an agar tidak mencuri huruf.
    aug_X: list[np.ndarray] = []
    aug_y: list[np.ndarray] = []
    for cls_id in np.unique(ytr):
        cls_idx = np.nonzero(ytr == cls_id)[0]
        n_cls = len(cls_idx)
        ch = C.CHARSET[int(cls_id)]
        if ch in ".,":
            keep = cls_idx if n_cls <= 180 else rng.choice(cls_idx, 180, replace=False)
            aug_X.append(Xtr_img[keep])
            aug_y.append(np.full(len(keep), cls_id, dtype=ytr.dtype))
            aug_n = min(40, len(keep))
            gen_imgs = [_augment_patch(Xtr_img[int(rng.choice(keep))], rng) for _ in range(aug_n)]
            if gen_imgs:
                aug_X.append(np.stack(gen_imgs))
                aug_y.append(np.full(aug_n, cls_id, dtype=ytr.dtype))
        elif ch in "%&+:=()@/-":
            aug_X.append(Xtr_img[cls_idx])
            aug_y.append(np.full(n_cls, cls_id, dtype=ytr.dtype))
            needed = min(20, n_cls * 2) if n_cls < 20 else min(40, n_cls)
            if needed > 0:
                gen_imgs = [_augment_patch(Xtr_img[int(rng.choice(cls_idx))], rng) for _ in range(needed)]
                aug_X.append(np.stack(gen_imgs))
                aug_y.append(np.full(needed, cls_id, dtype=ytr.dtype))
        else:
            aug_X.append(Xtr_img[cls_idx])
            aug_y.append(np.full(n_cls, cls_id, dtype=ytr.dtype))
            target_n = max(80, min(240, n_cls * 3))
            needed = max(0, target_n - n_cls)
            if needed > 0:
                gen_imgs = [_augment_patch(Xtr_img[int(rng.choice(cls_idx))], rng) for _ in range(needed)]
                aug_X.append(np.stack(gen_imgs))
                aug_y.append(np.full(needed, cls_id, dtype=ytr.dtype))

    Xtr_img = np.concatenate(aug_X, axis=0)
    ytr = np.concatenate(aug_y, axis=0)

    n_synth = 0
    if with_synthetic:
        path = C.DATA_DIR / "chars_train.npz"
        if path.exists():
            d = np.load(path)
            Xs, ys = d["X"], d["y"]
            n_synth = len(ys)
            Xtr_img = np.concatenate([Xtr_img, Xs], axis=0)
            ytr = np.concatenate([ytr, ys], axis=0)

    print(
        f"  asli={len(y_real)}  sintetis={n_synth}  "
        f"latih(aug)={len(ytr)}  validasi(asli)={len(yva)}  kelas={len(np.unique(ytr))}"
    )
    return Xtr_img, ytr, Xva_img, yva


def export_json(model: LinearSVC, path: Path, threshold: float) -> None:
    coef = model.coef_.astype(np.float64)
    payload = {
        "charset": C.CHARSET,
        "char_size": C.CHAR_SIZE,
        "hog": {"cell": C.HOG_CELL, "bins": C.HOG_BINS, "block": C.HOG_BLOCK},
        "svm": {
            "classes": [int(c) for c in model.classes_],
            "coef": [[round(float(v), 6) for v in row] for row in coef],
            "intercept": [round(float(v), 6) for v in model.intercept_],
        },
        "confidence_threshold": threshold,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"  JSON  : {path.name}  ({path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c", type=float, default=C.SVM_C)
    ap.add_argument("--max-iter", type=int, default=C.SVM_MAX_ITER)
    ap.add_argument("--with-synthetic", action="store_true",
                    help="gabungkan data sintetis sebagai pelengkap (default: hanya data asli)")
    args = ap.parse_args()

    print("[1/4] Memuat data karakter")
    Xtr_img, ytr, Xva_img, yva = load_split(args.val_ratio, args.seed, args.with_synthetic)

    print("[2/4] Ekstraksi fitur HOG")
    Xtr = F.hog_batch(Xtr_img)
    Xva = F.hog_batch(Xva_img)
    print(f"  dimensi fitur: {Xtr.shape[1]}")

    print("[3/4] Melatih Linear SVM")
    model = LinearSVC(C=args.c, max_iter=args.max_iter, dual=True, class_weight=None, random_state=args.seed)
    model.fit(Xtr, ytr)

    pred = model.predict(Xva)
    acc = accuracy_score(yva, pred)
    print(f"  akurasi karakter validasi: {acc * 100:.2f}%")

    labels = sorted(np.unique(np.concatenate([yva, pred])))
    names = [C.IDX_TO_CHAR[int(i)] for i in labels]
    print("\n  laporan per kelas:")
    print(classification_report(yva, pred, labels=labels, target_names=names, digits=3, zero_division=0))

    cm = confusion_matrix(yva, pred, labels=labels)
    np.save(C.OUTPUT_DIR / "char_confusion.npy", cm)
    (C.OUTPUT_DIR / "char_confusion_labels.json").write_text(
        json.dumps(names), encoding="utf-8"
    )

    print("[4/4] Menyimpan model")
    joblib.dump(model, C.MODEL_DIR / "char_svm.joblib")
    print(f"  joblib: char_svm.joblib ({(C.MODEL_DIR / 'char_svm.joblib').stat().st_size / 1024:.0f} KB)")
    export_json(model, C.MODEL_DIR / "char_svm.json", C.CONFIDENCE_THRESHOLD)

    summary = {
        "val_accuracy": float(acc),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_features": int(Xtr.shape[1]),
        "n_classes": int(len(np.unique(ytr))),
        "C": args.c,
    }
    (C.OUTPUT_DIR / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nRingkasan: {summary}")


if __name__ == "__main__":
    main()
