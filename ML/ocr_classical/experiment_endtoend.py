"""Bandingkan HOG saja vs HOG+Zoning pada data uji yang SAMA.

Akurasi validasi per karakter hanya mengukur pengenalan glyph yang sudah
dipotong rapi. Yang menentukan mutu produk adalah CER/WER end-to-end, yang
juga menyerap kesalahan segmentasi. Skrip ini melatih dua model pada data
latih identik, mengevaluasi keduanya pada baris uji identik, lalu
melaporkan selisihnya.

Catatan metodologi: perbandingan ini TIDAK memakai angka README lama, karena
data latih dan uji di repo ini diregenerasi (seed 42) sehingga angka lama
berasal dari himpunan berbeda. Yang dibandingkan di sini adalah dua model
pada satu himpunan yang sama.

Pemakaian:
    python experiment_endtoend.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import config as C
import features as F
from evaluate import cer, load_lines, wer
from ocr_pipeline import ClassicalOCR
from train_char import load_split


def export_model(use_zoning: bool, path: Path, seed: int = 42) -> dict:
    """Latih satu varian lalu tulis JSON dengan skema yang sama.

    Agar kedua model dapat dijalankan pipeline yang sama, model HOG-saja
    diekspor sebagai 340 dimensi dengan 16 kolom zona bernilai nol:
      - scaler zona: mean 0, scale 1  -> zona lewat apa adanya
      - bobot zona : 0                -> zona tidak memengaruhi skor

    Dengan begitu perbedaan hasil murni berasal dari fitur zoning, bukan dari
    perbedaan bentuk berkas.
    """
    Xtr_img, ytr, _Xva, _yva = load_split(0.12, seed)
    Xtr = F.combined_batch(Xtr_img) if use_zoning else F.hog_batch(Xtr_img)

    scaler = StandardScaler().fit(Xtr)
    model = LinearSVC(C=C.SVM_C, max_iter=C.SVM_MAX_ITER, dual=True, random_state=seed)
    model.fit(scaler.transform(Xtr), ytr)

    mean = [float(v) for v in scaler.mean_]
    scale = [float(v) for v in scaler.scale_]
    coef = [[float(v) for v in row] for row in model.coef_]

    if not use_zoning:
        mean += [0.0] * C.ZONING_DIM
        scale += [1.0] * C.ZONING_DIM
        coef = [row + [0.0] * C.ZONING_DIM for row in coef]

    payload = {
        "charset": C.CHARSET,
        "char_size": C.CHAR_SIZE,
        "hog": {"cell": C.HOG_CELL, "bins": C.HOG_BINS, "block": C.HOG_BLOCK},
        "zoning": {"grid": C.ZONING_GRID, "dim": C.ZONING_DIM},
        "features": {
            "hog_dim": C.HOG_DIM,
            "zoning_dim": C.ZONING_DIM,
            "combined_dim": C.FEATURE_DIM,
            "svm_input_dim": int(model.coef_.shape[1]) + (0 if use_zoning else C.ZONING_DIM),
        },
        "scaler": {"mean": mean, "scale": scale},
        "pca": None,
        "svm": {
            "classes": [int(c) for c in model.classes_],
            "coef": coef,
            "intercept": [float(v) for v in model.intercept_],
        },
        "confidence_threshold": C.CONFIDENCE_THRESHOLD,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def evaluate_model(model_path: Path, images, labels) -> dict:
    ocr = ClassicalOCR(
        method="otsu", use_deskew=True, use_lexicon=True, model_path=model_path
    )
    cers, wers, exact = [], [], 0
    for img, truth in zip(images, labels):
        res = ocr.run(img)
        cers.append(cer(res.text, truth))
        wers.append(wer(res.text, truth))
        if res.text.strip() == truth.strip():
            exact += 1
    return {
        "cer": float(np.mean(cers)),
        "wer": float(np.mean(wers)),
        "exact": exact / max(len(images), 1),
    }


def main() -> None:
    out_dir = C.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    images, labels = load_lines()
    print(f"baris uji: {len(images)}\n")

    rows = []
    for name, use_zoning in (("HOG saja", False), ("HOG+Zoning+Scaler", True)):
        model_path = out_dir / ("cmp_%s.json" % ("zoning" if use_zoning else "hog"))
        print(f"[latih] {name}")
        export_model(use_zoning, model_path)
        metrics = evaluate_model(model_path, images, labels)
        metrics["name"] = name
        rows.append(metrics)
        print("  CER=%.2f%%  WER=%.2f%%  exact=%.2f%%\n"
              % (metrics["cer"] * 100, metrics["wer"] * 100, metrics["exact"] * 100))

    print("%-22s %8s %8s %8s" % ("model", "CER", "WER", "exact"))
    print("-" * 50)
    for r in rows:
        print("%-22s %7.2f%% %7.2f%% %7.2f%%"
              % (r["name"], r["cer"] * 100, r["wer"] * 100, r["exact"] * 100))
    print("-" * 50)
    d_cer = (rows[0]["cer"] - rows[1]["cer"]) * 100
    d_wer = (rows[0]["wer"] - rows[1]["wer"]) * 100
    print("selisih (positif = zoning lebih baik): CER %+.2f poin, WER %+.2f poin"
          % (d_cer, d_wer))

    (out_dir / "endtoend_compare.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
