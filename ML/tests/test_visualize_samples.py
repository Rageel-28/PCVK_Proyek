"""Tes visualisasi sampel dataset.

Visualisasi adalah gerbang kualitas F0: kalau 100 sampel acak terlihat
berisi, terpusat, dan labelnya cocok, dataset hasil render bisa dipercaya.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C
from ocr.visualize_samples import plot_samples


def _fake_dataset(n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = np.zeros((n, C.CHAR_SIZE, C.CHAR_SIZE), dtype=np.uint8)
    for i in range(n):
        y0 = int(rng.integers(4, 16))
        x0 = int(rng.integers(4, 24))
        X[i, y0 : y0 + 12, x0 : x0 + 6] = C.INK
    y = rng.integers(0, len(C.CHARSET), size=n).astype(np.int64)
    return X, y


def test_plot_samples_writes_a_png(tmp_path: Path) -> None:
    X, y = _fake_dataset()
    out = plot_samples(X, y, tmp_path / "grid.png", n=100, seed=1)
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_samples_creates_parent_directory(tmp_path: Path) -> None:
    X, y = _fake_dataset()
    out = plot_samples(X, y, tmp_path / "nested" / "deep" / "grid.png", n=25)
    assert out.exists()


def test_plot_samples_clamps_when_fewer_samples_than_requested(
    tmp_path: Path,
) -> None:
    X, y = _fake_dataset(n=10)
    out = plot_samples(X, y, tmp_path / "grid.png", n=100)
    assert out.exists()


def test_plot_samples_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    X, y = _fake_dataset()
    a = plot_samples(X, y, tmp_path / "a.png", n=50, seed=7)
    b = plot_samples(X, y, tmp_path / "b.png", n=50, seed=7)
    assert a.read_bytes() == b.read_bytes()


def test_plot_samples_rejects_mismatched_labels(tmp_path: Path) -> None:
    X, y = _fake_dataset()
    with pytest.raises(ValueError):
        plot_samples(X, y[:-5], tmp_path / "grid.png", n=10)
