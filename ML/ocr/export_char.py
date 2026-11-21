"""Ekspor classifier karakter ke JSON yang bisa dibaca Dart.

Berkas keluaran adalah **kontrak** antara pelatihan Python dan inferensi
Dart. Isinya harus cukup lengkap sehingga Dart tidak perlu menebak apa pun:
charset, ukuran kanvas, parameter tiap blok fitur, koefisien, dan suhu
kalibrasi.

Selain model, skrip ini juga bisa menulis ``golden_vectors.json``: beberapa
glyph contoh beserta fitur dan prediksi versi Python. Tes Dart memakainya
untuk memastikan port-nya menghasilkan angka yang sama.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ocr.config as C


def _features_block() -> dict:
    return {
        "hog": {
            "cell": C.HOG_CELL,
            "bins": C.HOG_BINS,
            "block": C.HOG_BLOCK,
            "dim": C.HOG_DIM,
        },
        "zoning": {"grid": C.ZONING_GRID, "dim": C.ZONING_DIM},
        "projection": {
            "cols": C.PROJECTION_COLS,
            "rows": C.PROJECTION_ROWS,
            "dim": C.PROJECTION_DIM,
        },
        "hole": {"max": C.HOLE_MAX, "dim": C.HOLE_DIM},
    }


def build_payload(clf, temperature: float, decimals: int = 6) -> dict:
    """Susun dict model dari classifier sklearn.

    Koefisien dibulatkan ke ``decimals`` angka di belakang koma. Pembulatan
    ini memangkas ukuran JSON drastis tanpa mengubah prediksi (galat relatif
    ~1e-6, jauh di bawah jarak antar kelas).
    """
    coef = np.asarray(clf.coef_, dtype=np.float64)
    intercept = np.asarray(clf.intercept_, dtype=np.float64)

    n_classes, dim = coef.shape
    if dim != C.FEATURE_DIM:
        raise ValueError(
            f"dimensi fitur model {dim} != config {C.FEATURE_DIM}"
        )
    if n_classes != len(C.CHARSET):
        raise ValueError(
            f"jumlah kelas model {n_classes} != charset {len(C.CHARSET)}"
        )

    return {
        "version": 1,
        "charset": C.CHARSET,
        "char_size": C.CHAR_SIZE,
        "pad": C.PAD,
        "feature_dim": int(dim),
        "ink": C.INK,
        "paper": C.PAPER,
        "features": _features_block(),
        "temperature": float(temperature),
        "intercept": [round(float(v), decimals) for v in intercept],
        "coef": [round(float(v), decimals) for v in coef.ravel()],
    }


def write_model(clf, path: str | Path, temperature: float) -> Path:
    payload = build_payload(clf, temperature)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    return out


def predict_proba(clf, features: np.ndarray, temperature: float) -> np.ndarray:
    """Probabilitas kelas lewat softmax bersuhu dari decision_function."""
    scores = np.asarray(clf.decision_function(features), dtype=np.float64)
    z = scores / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def write_golden_vectors(
    clf,
    features_path: str | Path,
    out_path: str | Path,
    n_samples: int = 200,
    seed: int = 0,
) -> Path:
    """Tulis glyph contoh + fitur + prediksi Python untuk uji silang Dart.

    Isi tiap entri:
        ``image``  — 32x32 nilai 0..255, baris demi baris
        ``label``  — karakter yang benar
        ``feature``— 1844 nilai float (dibulatkan 5 desimal)
        ``pred``   — karakter tebakan Python
        ``probs``  — 42 probabilitas (dibulatkan 6 desimal)
    """
    from ocr.features import extract_features_batch

    data = np.load(features_path)
    F, y = data["F"], data["y"]

    # Ambil glyph aslinya dari dataset mentah bila tersedia.
    raw_path = C.DATA_DIR / "chars.npz"
    raw = np.load(raw_path)
    X_raw = raw["X"]

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(F), size=min(n_samples, len(F)), replace=False)

    probs = predict_proba(clf, F[picks], 1.0)
    pred = probs.argmax(axis=1)

    # Pastikan fitur yang ditulis memang hasil ekstraksi dari gambar yang
    # sama, bukan sekadar salinan cache.
    recomputed = extract_features_batch(X_raw[picks])
    if not np.allclose(recomputed, F[picks], atol=1e-4):
        raise RuntimeError(
            "fitur cache tidak cocok dengan hasil ekstraksi ulang; "
            "jalankan ulang `python -m ocr.features`"
        )

    entries = []
    for k, idx in enumerate(picks):
        entries.append(
            {
                "image": X_raw[idx].ravel().tolist(),
                "label": C.IDX_TO_CHAR[int(y[idx])],
                "feature": [round(float(v), 5) for v in F[idx]],
                "pred": C.IDX_TO_CHAR[int(pred[k])],
                "probs": [round(float(v), 6) for v in probs[k]],
            }
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"vectors": entries}), encoding="utf-8")
    return out


def main() -> None:
    import joblib
    from sklearn.svm import LinearSVC

    ap = argparse.ArgumentParser(description="Ekspor model karakter ke JSON.")
    ap.add_argument(
        "--features", type=str, default=str(C.OUTPUT_DIR / "features.npz")
    )
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument(
        "--out", type=str, default=str(C.MODEL_DIR / "char_clf.json")
    )
    ap.add_argument(
        "--golden",
        type=str,
        default=str(C.MODEL_DIR / "golden_vectors.json"),
        help="lokasi golden vectors untuk uji silang Dart",
    )
    ap.add_argument("--n-golden", type=int, default=200)
    args = ap.parse_args()

    from ocr.train_char import fit_temperature, split_by_line

    data = np.load(args.features)
    F, y, ti = data["F"], data["y"], data["text_index"]
    parts = split_by_line(ti)

    clf = LinearSVC(C=args.C, max_iter=5000, random_state=C.RANDOM_SEED)
    clf.fit(F[parts["train"]], y[parts["train"]])
    temperature = fit_temperature(
        clf.decision_function(F[parts["val"]]), y[parts["val"]]
    )

    out = write_model(clf, args.out, temperature)
    print(f"model: {out} ({out.stat().st_size / 1e6:.2f} MB)")

    golden = write_golden_vectors(clf, args.features, args.golden, args.n_golden)
    print(f"golden vectors: {golden} ({golden.stat().st_size / 1e6:.2f} MB)")

    joblib.dump(clf, C.MODEL_DIR / "char_clf.joblib")
    print(f"joblib: {C.MODEL_DIR / 'char_clf.joblib'}")


if __name__ == "__main__":
    main()
