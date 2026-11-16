"""Latih pengklasifikasi karakter: HOG + Zoning -> StandardScaler -> (PCA) -> SVM.

Model disimpan dalam dua bentuk:
  models/char_svm.joblib  -> untuk eksperimen Python
  models/char_svm.json    -> untuk dipakai aplikasi Flutter

Alur latih:

    1. Citra 32x32 -> HOG 324 dim + Zoning 16 dim  = 340 dim  (concatenate)
    2. StandardScaler   -> mean 0, variansi 1 per dimensi (WAJIB)
    3. PCA (opsional)   -> buang redundansi HOG/Zoning bila diminta
    4. LinearSVC        -> C dipilih lewat GridSearchCV

Kenapa StandardScaler wajib
---------------------------
HOG dinormalisasi L2-Hys sehingga nilainya berkisar 0..0.6, sedangkan zoning
adalah kepadatan 0..1 dengan rata-rata sekitar 0.2. Tanpa penyamaan skala,
masalah optimisasi SVM memberi penalti tidak seimbang: zona dengan nilai
absolut lebih besar mendominasi margin. Setelah discalakan, setiap dimensi
menyumbang setara, dan C menjadi bermakna sama untuk semua fitur.

Ekspor JSON memuat scaler dan PCA apa adanya (bukan dilipat ke bobot SVM),
supaya tiap tahap dapat diperiksa dan ditiru Dart langkah demi langkah.
Melipat memang mungkin untuk model linear, tetapi menyembunyikan tahap yang
justru ingin dijelaskan di laporan.

Pemakaian
---------
    python train_char.py                      # konfigurasi config.py
    python train_char.py --pca 120            # pakai PCA 120 komponen
    python train_char.py --grid-search        # cari C terbaik
    python train_char.py --compare            # bandingkan varian fitur
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import config as C
import features as F


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Ekspor
# ---------------------------------------------------------------------------
def export_json(
    model: LinearSVC,
    path: Path,
    threshold: float,
    scaler: StandardScaler | None,
    pca: PCA | None,
) -> None:
    """Tulis model ke JSON yang dapat dieksekusi Dart tanpa scikit-learn.

    Struktur:
        features : dimensi tiap kelompok
        scaler   : mean + scale (StandardScaler), null bila tidak dipakai
        pca      : components + mean, null bila tidak dipakai
        svm      : coef + intercept (model linear)

    Urutan penerapan di Dart:
        x -> (x - mean) / scale -> (p - pca_mean) @ components.T -> coef @ p + b
    """
    coef = model.coef_.astype(np.float64)

    payload = {
        "charset": C.CHARSET,
        "char_size": C.CHAR_SIZE,
        "hog": {"cell": C.HOG_CELL, "bins": C.HOG_BINS, "block": C.HOG_BLOCK},
        "zoning": {"grid": C.ZONING_GRID, "dim": C.ZONING_DIM},
        "features": {
            "hog_dim": C.HOG_DIM,
            "zoning_dim": C.ZONING_DIM,
            "combined_dim": C.FEATURE_DIM,
            "svm_input_dim": int(coef.shape[1]),
        },
        "scaler": None
        if scaler is None
        else {
            "mean": [round(float(v), 8) for v in scaler.mean_],
            "scale": [round(float(v), 8) for v in scaler.scale_],
        },
        "pca": None
        if pca is None
        else {
            "n_components": int(pca.components_.shape[0]),
            "mean": [round(float(v), 8) for v in pca.mean_],
            "components": [
                [round(float(v), 8) for v in row] for row in pca.components_
            ],
        },
        "svm": {
            "classes": [int(c) for c in model.classes_],
            "coef": [[round(float(v), 8) for v in row] for row in coef],
            "intercept": [round(float(v), 8) for v in model.intercept_],
        },
        "confidence_threshold": threshold,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"  JSON  : {path.name}  ({path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Varian fitur
# ---------------------------------------------------------------------------
def build_features(images: np.ndarray, use_zoning: bool) -> np.ndarray:
    if use_zoning:
        return F.combined_batch(images)
    return F.hog_batch(images)


def evaluate(name: str, Xtr, ytr, Xva, yva, c: float, seed: int) -> dict:
    t0 = time.time()
    model = LinearSVC(C=c, max_iter=C.SVM_MAX_ITER, dual=True, random_state=seed)
    model.fit(Xtr, ytr)
    acc = accuracy_score(yva, model.predict(Xva))
    dt = time.time() - t0
    print(f"  {name:38} dim={Xtr.shape[1]:4d}  akurasi={acc * 100:6.2f}%  ({dt:.1f}s)")
    return {"name": name, "dim": int(Xtr.shape[1]), "accuracy": float(acc)}


def compare_variants(Xtr_img, ytr, Xva_img, yva, seed: int) -> list[dict]:
    """Bandingkan empat konfigurasi untuk laporan (angka jujur, bukan klaim)."""
    print("\n[perbandingan] pengaruh zoning & penskalaan")
    results = []

    hog_tr = F.hog_batch(Xtr_img)
    hog_va = F.hog_batch(Xva_img)
    comb_tr = F.combined_batch(Xtr_img)
    comb_va = F.combined_batch(Xva_img)

    results.append(evaluate("HOG saja, tanpa scaler", hog_tr, ytr, hog_va, yva, C.SVM_C, seed))

    sc_hog = StandardScaler().fit(hog_tr)
    results.append(evaluate(
        "HOG saja, + StandardScaler",
        sc_hog.transform(hog_tr), ytr, sc_hog.transform(hog_va), yva, C.SVM_C, seed))

    results.append(evaluate("HOG+Zoning, tanpa scaler", comb_tr, ytr, comb_va, yva, C.SVM_C, seed))

    sc_comb = StandardScaler().fit(comb_tr)
    results.append(evaluate(
        "HOG+Zoning, + StandardScaler",
        sc_comb.transform(comb_tr), ytr, sc_comb.transform(comb_va), yva, C.SVM_C, seed))

    return results


def grid_search_c(Xtr, ytr, seed: int, sample: int = 12000) -> float:
    """Cari C terbaik dengan GridSearchCV (3-fold) pada subsampel.

    Gamma tidak disertakan karena kernel yang dipakai linear — tidak punya
    parameter gamma. RBF diuji terpisah sebagai pembanding di laporan, tetapi
    tidak dipakai aplikasi: bobotnya tidak bisa diekspor sebagai satu matriks
    `coef` sehingga inferensi Dart akan jauh lebih berat.
    """
    rng = np.random.default_rng(seed)
    if len(ytr) > sample:
        idx = rng.choice(len(ytr), sample, replace=False)
        Xs, ys = Xtr[idx], ytr[idx]
    else:
        Xs, ys = Xtr, ytr

    grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    print(f"\n[grid-search] LinearSVC C={grid['C']}  sampel={len(ys)}  3-fold")
    search = GridSearchCV(
        LinearSVC(max_iter=C.SVM_MAX_ITER, dual=True, random_state=seed),
        grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    t0 = time.time()
    search.fit(Xs, ys)
    print(f"  C terbaik = {search.best_params_['C']}  "
          f"(cv akurasi {search.best_score_ * 100:.2f}%, {time.time() - t0:.0f}s)")
    for c, mean in zip(search.cv_results_["param_C"], search.cv_results_["mean_test_score"]):
        print(f"    C={c:<6} akurasi={mean * 100:.2f}%")
    return float(search.best_params_["C"])


# ---------------------------------------------------------------------------
# Utama
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c", type=float, default=C.SVM_C)
    ap.add_argument("--max-iter", type=int, default=C.SVM_MAX_ITER)
    ap.add_argument("--pca", type=int, default=C.PCA_COMPONENTS,
                    help="jumlah komponen PCA; 0 = tanpa PCA")
    ap.add_argument("--no-scaler", action="store_true",
                    help="lewati StandardScaler (hanya untuk eksperimen)")
    ap.add_argument("--grid-search", action="store_true", help="cari C terbaik")
    ap.add_argument("--compare", action="store_true",
                    help="bandingkan varian fitur sebelum melatih model final")
    args = ap.parse_args()

    print("[1/5] Memuat data karakter")
    Xtr_img, ytr, Xva_img, yva = load_split(args.val_ratio, args.seed)

    comparison = compare_variants(Xtr_img, ytr, Xva_img, yva, args.seed) if args.compare else []

    print("\n[2/5] Ekstraksi fitur HOG + Zoning")
    Xtr = F.combined_batch(Xtr_img)
    Xva = F.combined_batch(Xva_img)
    print(f"  HOG {C.HOG_DIM} + Zoning {C.ZONING_DIM} = {Xtr.shape[1]} dimensi")

    scaler = None
    if not args.no_scaler:
        print("\n[3/5] StandardScaler")
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xva = scaler.transform(Xva)
        print(f"  mean  rentang: {scaler.mean_.min():.4f} .. {scaler.mean_.max():.4f}")
        print(f"  scale rentang: {scaler.scale_.min():.4f} .. {scaler.scale_.max():.4f}")
    else:
        print("\n[3/5] StandardScaler dilewati (--no-scaler)")

    pca = None
    if args.pca > 0:
        print(f"\n[3b/5] PCA -> {args.pca} komponen")
        pca = PCA(n_components=args.pca, random_state=args.seed).fit(Xtr)
        Xtr = pca.transform(Xtr)
        Xva = pca.transform(Xva)
        ratio = float(pca.explained_variance_ratio_.sum())
        print(f"  variansi terjelaskan: {ratio * 100:.2f}%")

    c_value = args.c
    if args.grid_search:
        c_value = grid_search_c(Xtr, ytr, args.seed)

    print(f"\n[4/5] Melatih LinearSVC (C={c_value})")
    t0 = time.time()
    model = LinearSVC(C=c_value, max_iter=args.max_iter, dual=True, random_state=args.seed)
    model.fit(Xtr, ytr)
    print(f"  selesai dalam {time.time() - t0:.1f}s")

    pred = model.predict(Xva)
    acc = accuracy_score(yva, pred)
    print(f"  akurasi karakter validasi: {acc * 100:.2f}%")

    labels = sorted(np.unique(np.concatenate([yva, pred])))
    names = [C.IDX_TO_CHAR[int(i)] for i in labels]
    print("\n  laporan per kelas:")
    print(classification_report(yva, pred, labels=labels, target_names=names,
                                digits=3, zero_division=0))

    cm = confusion_matrix(yva, pred, labels=labels)
    np.save(C.OUTPUT_DIR / "char_confusion.npy", cm)
    (C.OUTPUT_DIR / "char_confusion_labels.json").write_text(
        json.dumps(names), encoding="utf-8"
    )

    print("\n[5/5] Menyimpan model")
    joblib.dump({"svm": model, "scaler": scaler, "pca": pca},
                C.MODEL_DIR / "char_svm.joblib")
    print(f"  joblib: char_svm.joblib "
          f"({(C.MODEL_DIR / 'char_svm.joblib').stat().st_size / 1024:.0f} KB)")
    export_json(model, C.MODEL_DIR / "char_svm.json", C.CONFIDENCE_THRESHOLD, scaler, pca)

    summary = {
        "val_accuracy": float(acc),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "hog_dim": C.HOG_DIM,
        "zoning_dim": C.ZONING_DIM,
        "combined_dim": C.FEATURE_DIM,
        "svm_input_dim": int(model.coef_.shape[1]),
        "scaler": scaler is not None,
        "pca_components": int(args.pca),
        "pca_variance": float(pca.explained_variance_ratio_.sum()) if pca else None,
        "n_classes": int(len(np.unique(ytr))),
        "C": c_value,
        "comparison": comparison,
    }
    (C.OUTPUT_DIR / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nRingkasan: akurasi={acc * 100:.2f}%  "
          f"fitur={C.FEATURE_DIM}->{model.coef_.shape[1]}  "
          f"scaler={scaler is not None}  pca={args.pca}")


if __name__ == "__main__":
    main()
