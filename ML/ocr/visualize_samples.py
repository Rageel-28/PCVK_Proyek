"""Visualisasi sampel dataset karakter.

Gerbang kualitas F0: 100 sampel acak harus terlihat seperti glyph yang
benar, terpusat, dan labelnya cocok. Kalau ada kelas yang sampelnya
kacau, lebih baik ketahuan sekarang daripada setelah training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import ocr.config as C


def plot_samples(
    X: np.ndarray,
    y: np.ndarray,
    out_path: str | Path,
    n: int = 100,
    seed: int = 0,
    cols: int = 10,
) -> Path:
    """Simpan grid ``n`` sampel acak beserta labelnya sebagai PNG.

    Args:
        X: ``(N, 32, 32)`` uint8, 255 = tinta.
        y: ``(N,)`` indeks kelas.
        out_path: tujuan berkas PNG.
        n: jumlah sampel yang ditampilkan (dipotong bila melebihi ``len(X)``).
        seed: seed pemilihan sampel agar hasil dapat direproduksi.
        cols: jumlah kolom grid.

    Returns:
        Path berkas yang ditulis.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(X) != len(y):
        raise ValueError(
            f"jumlah gambar ({len(X)}) dan label ({len(y)}) tidak sama"
        )
    if len(X) == 0:
        raise ValueError("dataset kosong")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n = min(n, len(X))
    picks = rng.choice(len(X), size=n, replace=False)

    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 1.1))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax in axes_flat:
        ax.axis("off")

    for ax, idx in zip(axes_flat, picks):
        label_idx = int(y[idx])
        ax.imshow(X[idx], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(C.IDX_TO_CHAR[label_idx], fontsize=9, pad=2)

    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualisasikan sampel dataset karakter.")
    ap.add_argument("--npz", type=str, default=str(C.DATA_DIR / "chars.npz"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", type=str, default=str(C.OUTPUT_DIR / "samples_grid.png")
    )
    args = ap.parse_args()

    data = np.load(args.npz)
    X, y = data["X"], data["y"]
    print(f"{len(X)} sampel, {len(np.unique(y))} kelas terpakai")

    counts = np.bincount(y, minlength=len(C.CHARSET))
    print(f"min/kelas: {counts.min()}  maks/kelas: {counts.max()}")

    out = plot_samples(X, y, args.out, n=args.n, seed=args.seed)
    print(f"grid disimpan ke {out}")


if __name__ == "__main__":
    main()
