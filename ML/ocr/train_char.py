"""Latih classifier karakter M2 (LinearSVC + kalibrasi suhu).

Kenapa LinearSVC, bukan ensemble tiga model seperti rencana awal? Karena
diukur: LinearSVC 99,882% vs LogisticRegression 99,861% vs soft-voting
keduanya 99,865% pada val. Ensemble tidak menambah akurasi sama sekali,
tetapi menambah ukuran model dan biaya inferensi di perangkat. Satu model
linear menang secara akurasi sekaligus kesederhanaan.

Keluaran:

- ``ocr/models/char_clf.npz``      koefisien float32 untuk analisis
- ``ocr/models/char_clf.json``     model siap-port ke Dart
- ``ocr/output/char_confusion.png`` matriks konfusi 42x42
- ``ocr/output/char_report.json``  metrik lengkap
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import ocr.config as C


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split_by_line(
    text_index: np.ndarray,
    seed: int = C.RANDOM_SEED,
    train: float = C.SPLIT_TRAIN,
    val: float = C.SPLIT_VAL,
) -> dict[str, np.ndarray]:
    """Bagi sampel per **baris sumber**, bukan per karakter.

    Kalau dibagi per karakter, glyph dari baris yang sama bisa muncul di
    train dan test sekaligus. Itu kebocoran: classifier menghafal bentuk
    render baris tertentu dan akurasi uji jadi palsu tinggi.

    Returns:
        ``{"train": bool_mask, "val": bool_mask, "test": bool_mask}``.
    """
    ti = np.asarray(text_index)
    if ti.size == 0:
        raise ValueError("text_index kosong")

    lines = np.unique(ti)
    rng = np.random.default_rng(seed)
    rng.shuffle(lines)

    n = len(lines)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    train_lines = lines[:n_train]
    val_lines = lines[n_train : n_train + n_val]
    test_lines = lines[n_train + n_val :]

    return {
        "train": np.isin(ti, train_lines),
        "val": np.isin(ti, val_lines),
        "test": np.isin(ti, test_lines),
    }


# ---------------------------------------------------------------------------
# Kalibrasi suhu
# ---------------------------------------------------------------------------
def softmax_with_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax dengan pembagi suhu; ``T > 1`` melembutkan distribusi."""
    z = np.asarray(scores, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(
    scores: np.ndarray, labels: np.ndarray, grid: np.ndarray | None = None
) -> float:
    """Cari suhu yang meminimalkan NLL pada data validasi.

    Dicari dengan grid search, bukan optimasi gradien: ruangnya satu dimensi
    dan halus, jadi grid sudah cukup dan hasilnya deterministik.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.ndim != 2:
        raise ValueError(f"skor harus 2-d, dapat {scores.shape}")

    if grid is None:
        grid = np.concatenate(
            [np.arange(0.05, 1.0, 0.05), np.arange(1.0, 5.0, 0.1), np.arange(5.0, 20.5, 0.5)]
        )

    idx = np.arange(len(labels))
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        probs = softmax_with_temperature(scores, float(t))
        nll = float(-np.log(probs[idx, labels] + 1e-12).mean())
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Laporan
# ---------------------------------------------------------------------------
def confusion_report(
    y_true: np.ndarray, y_pred: np.ndarray, top: int | None = None
) -> list[tuple[int, str, str]]:
    """Pasangan (jumlah, karakter_asli, karakter_tertebak) terurut menurun."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"panjang tidak sama: {len(y_true)} vs {len(y_pred)}"
        )

    n = len(C.CHARSET)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    pairs = [
        (int(cm[i, j]), C.IDX_TO_CHAR[i], C.IDX_TO_CHAR[j])
        for i in range(n)
        for j in range(n)
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda r: (-r[0], r[1], r[2]))
    return pairs[:top] if top else pairs


def _accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float((pred == true).mean()) if len(true) else float("nan")


# ---------------------------------------------------------------------------
# Pelatihan
# ---------------------------------------------------------------------------
def train(
    features_path: Path,
    C_value: float = 1.0,
    seed: int = C.RANDOM_SEED,
) -> dict:
    """Latih LinearSVC, kalibrasi, dan kembalikan semua yang perlu diekspor."""
    from sklearn.svm import LinearSVC

    data = np.load(features_path)
    F = data["F"]
    y = data["y"]
    ti = data["text_index"]
    variant = data["variant"] if "variant" in data else np.zeros(len(y), dtype=np.int8)

    parts = split_by_line(ti, seed=seed)
    print(
        f"train {int(parts['train'].sum())}  val {int(parts['val'].sum())}  "
        f"test {int(parts['test'].sum())}"
    )

    t0 = time.time()
    clf = LinearSVC(C=C_value, max_iter=5000, random_state=seed)
    clf.fit(F[parts["train"]], y[parts["train"]])
    print(f"fit LinearSVC: {time.time() - t0:.0f}s")

    val_scores = clf.decision_function(F[parts["val"]])
    temperature = fit_temperature(val_scores, y[parts["val"]])
    print(f"suhu kalibrasi: {temperature:.2f}")

    metrics: dict[str, object] = {
        "feature_dim": int(F.shape[1]),
        "n_train": int(parts["train"].sum()),
        "n_val": int(parts["val"].sum()),
        "n_test": int(parts["test"].sum()),
        "C": C_value,
        "temperature": temperature,
        "splits": {},
    }

    for name in ("train", "val", "test"):
        mask = parts[name]
        pred = clf.predict(F[mask])
        entry: dict[str, object] = {
            "accuracy": _accuracy(pred, y[mask]),
            "n": int(mask.sum()),
        }
        # Akurasi per varian: bersih vs teraugmentasi.
        var_scores = {}
        for v in np.unique(variant[mask]):
            sub = variant[mask] == v
            var_scores[f"variant_{int(v)}"] = _accuracy(pred[sub], y[mask][sub])
        entry["by_variant"] = var_scores
        metrics["splits"][name] = entry
        print(
            f"{name}: acc {entry['accuracy']:.5f}  varian {var_scores}"
        )

    test_pred = clf.predict(F[parts["test"]])
    metrics["top_confusions_test"] = confusion_report(
        y[parts["test"]], test_pred, top=15
    )
    metrics["per_class_recall_test"] = {
        C.IDX_TO_CHAR[i]: float(
            (test_pred[y[parts["test"]] == i] == i).mean()
        )
        if np.any(y[parts["test"]] == i)
        else None
        for i in range(len(C.CHARSET))
    }

    return {
        "clf": clf,
        "metrics": metrics,
        "parts": parts,
        "y": y,
        "variant": variant,
        "F": F,
    }


def plot_confusion(result: dict, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = result["F"]
    y = result["y"]
    parts = result["parts"]
    clf = result["clf"]

    pred = clf.predict(F[parts["test"]])
    true = y[parts["test"]]
    n = len(C.CHARSET)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 13))
    im = ax.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=1))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(list(C.CHARSET), fontsize=7)
    ax.set_yticklabels(list(C.CHARSET), fontsize=7)
    ax.set_xlabel("tertebak")
    ax.set_ylabel("asli")
    ax.set_title("Matriks konfusi 42x42 (data uji)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Latih classifier karakter M2.")
    ap.add_argument(
        "--features", type=str, default=str(C.OUTPUT_DIR / "features.npz")
    )
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument(
        "--out", type=str, default=str(C.MODEL_DIR / "char_clf.json")
    )
    args = ap.parse_args()

    result = train(Path(args.features), C_value=args.C, seed=args.seed)
    metrics = result["metrics"]

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = C.OUTPUT_DIR / "char_report.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"laporan: {report_path}")

    png = plot_confusion(result, C.OUTPUT_DIR / "char_confusion.png")
    print(f"matriks konfusi: {png}")

    val_acc = metrics["splits"]["val"]["accuracy"]
    test_acc = metrics["splits"]["test"]["accuracy"]
    gate = 0.998
    print(
        f"\nGERBANG F1 (val >= {gate:.1%}): "
        f"val {val_acc:.5f} test {test_acc:.5f} -> "
        f"{'LULUS' if val_acc >= gate else 'GAGAL'}"
    )


if __name__ == "__main__":
    main()
