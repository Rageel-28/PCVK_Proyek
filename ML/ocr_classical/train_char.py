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


def load_split(val_ratio: float, seed: int):
    path = C.DATA_DIR / "chars_train.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} belum ada. Jalankan dulu: python render_synthetic.py"
        )
    d = np.load(path)
    X_img, y = d["X"], d["y"]

    # Buang kelas yang sampelnya terlalu sedikit agar tidak mengganggu.
    counts = np.bincount(y, minlength=len(C.CHARSET))
    keep = np.isin(y, np.nonzero(counts >= 5)[0])
    dropped = int((~keep).sum())
    X_img, y = X_img[keep], y[keep]
    if dropped:
        print(f"  {dropped} sampel dari kelas langka dibuang")

    Xtr_img, Xva_img, ytr, yva = train_test_split(
        X_img, y, test_size=val_ratio, random_state=seed, stratify=y
    )
    print(f"  latih={len(ytr)}  validasi={len(yva)}  kelas={len(np.unique(y))}")
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
    args = ap.parse_args()

    print("[1/4] Memuat data karakter")
    Xtr_img, ytr, Xva_img, yva = load_split(args.val_ratio, args.seed)

    print("[2/4] Ekstraksi fitur HOG")
    Xtr = F.hog_batch(Xtr_img)
    Xva = F.hog_batch(Xva_img)
    print(f"  dimensi fitur: {Xtr.shape[1]}")

    print("[3/4] Melatih Linear SVM")
    model = LinearSVC(C=args.c, max_iter=args.max_iter, dual=True, random_state=args.seed)
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
