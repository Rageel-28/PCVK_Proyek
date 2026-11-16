"""Uji apakah PCA membantu pada vektor gabungan HOG+Zoning.

PCA diuji pada beberapa jumlah komponen. Yang dibandingkan:
  - akurasi validasi
  - waktu latih
  - jumlah dimensi yang akhirnya diekspor (menentukan ukuran model)

Hasilnya dipakai untuk memutuskan `PCA_COMPONENTS` di `config.py`, bukan
ditebak. Bila PCA tidak menaikkan akurasi sama sekali, ia tidak dipakai:
setiap tahap tambahan harus dibayar dengan kejelasan dan risiko paritas.

Pemakaian:
    python experiment_pca.py
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import config as C
import features as F
from train_char import load_split


def main() -> None:
    print("Memuat data & mengekstraksi fitur...")
    Xtr_img, ytr, Xva_img, yva = load_split(0.12, 42)
    Xtr = F.combined_batch(Xtr_img)
    Xva = F.combined_batch(Xva_img)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)

    print(f"dimensi gabungan: {Xtr.shape[1]}\n")
    print("%-14s %8s %10s %10s" % ("konfigurasi", "dimensi", "akurasi", "waktu"))
    print("-" * 46)

    rows = []
    configs = [("tanpa PCA", 0), ("PCA 200", 200), ("PCA 120", 120),
               ("PCA 64", 64), ("PCA 32", 32)]

    for name, n in configs:
        t0 = time.time()
        if n:
            pca = PCA(n_components=n, random_state=42).fit(Xtr_s)
            a_tr, a_va = pca.transform(Xtr_s), pca.transform(Xva_s)
            var = float(pca.explained_variance_ratio_.sum()) * 100
        else:
            a_tr, a_va = Xtr_s, Xva_s
            var = 100.0

        model = LinearSVC(C=C.SVM_C, max_iter=C.SVM_MAX_ITER, dual=True, random_state=42)
        model.fit(a_tr, ytr)
        acc = accuracy_score(yva, model.predict(a_va))
        dt = time.time() - t0

        print("%-14s %8d %9.2f%% %8.0fs   (variansi %.1f%%)"
              % (name, a_tr.shape[1], acc * 100, dt, var))
        rows.append((name, a_tr.shape[1], acc * 100, var))

    print("-" * 46)
    best = max(rows, key=lambda r: r[2])
    print(f"akurasi tertinggi: {best[0]} ({best[2]:.2f}%)")
    print("\nCatatan: PCA hanya dipakai bila menaikkan akurasi secara berarti.")
    print("Bila selisihnya di bawah 0,05 poin, dimensi penuh lebih dipilih")
    print("karena lebih sederhana dan tidak menambah tahap yang bisa menyimpang.")


if __name__ == "__main__":
    main()
