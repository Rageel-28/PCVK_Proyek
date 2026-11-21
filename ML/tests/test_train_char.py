"""Tes utilitas pelatihan classifier karakter.

Yang diuji adalah bagian murni (split, kalibrasi suhu, laporan konfusi),
bukan proses fitting-nya — fitting diuji lewat gerbang akurasi di laporan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C
from ocr.train_char import (
    confusion_report,
    fit_temperature,
    softmax_with_temperature,
    split_by_line,
)


# ---------------------------------------------------------------------------
# Split bebas kebocoran
# ---------------------------------------------------------------------------
def _fake_text_index(n_lines: int = 100, per_line: int = 5) -> np.ndarray:
    return np.repeat(np.arange(n_lines), per_line).astype(np.int64)


def test_split_covers_every_sample_exactly_once() -> None:
    ti = _fake_text_index()
    parts = split_by_line(ti, seed=0)
    total = sum(int(m.sum()) for m in parts.values())
    assert total == len(ti)
    combined = np.zeros(len(ti), dtype=int)
    for mask in parts.values():
        combined += mask.astype(int)
    assert np.all(combined == 1), "ada sampel yang masuk dua split"


def test_split_never_shares_a_source_line_between_splits() -> None:
    ti = _fake_text_index(200)
    parts = split_by_line(ti, seed=3)
    line_sets = {name: set(np.unique(ti[m]).tolist()) for name, m in parts.items()}
    assert not (line_sets["train"] & line_sets["val"])
    assert not (line_sets["train"] & line_sets["test"])
    assert not (line_sets["val"] & line_sets["test"])


def test_split_is_deterministic_for_a_seed() -> None:
    ti = _fake_text_index()
    a = split_by_line(ti, seed=7)
    b = split_by_line(ti, seed=7)
    for name in a:
        assert np.array_equal(a[name], b[name])


def test_split_ratios_are_approximately_respected() -> None:
    ti = _fake_text_index(1000)
    parts = split_by_line(ti, seed=1)
    n = len(ti)
    assert abs(parts["train"].sum() / n - 0.70) < 0.02
    assert abs(parts["val"].sum() / n - 0.15) < 0.02
    assert abs(parts["test"].sum() / n - 0.15) < 0.02


def test_split_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        split_by_line(np.asarray([], dtype=np.int64))


# ---------------------------------------------------------------------------
# Kalibrasi suhu
# ---------------------------------------------------------------------------
def test_softmax_with_temperature_is_a_distribution() -> None:
    scores = np.random.default_rng(0).normal(size=(10, 5))
    probs = softmax_with_temperature(scores, 1.7)
    assert probs.shape == scores.shape
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()


def test_softmax_temperature_one_matches_plain_softmax() -> None:
    scores = np.random.default_rng(1).normal(size=(4, 3))
    plain = np.exp(scores - scores.max(axis=1, keepdims=True))
    plain /= plain.sum(axis=1, keepdims=True)
    assert np.allclose(softmax_with_temperature(scores, 1.0), plain)


def test_fit_temperature_returns_positive_scalar() -> None:
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(200, 6)) * 3
    labels = scores.argmax(axis=1)          # model sangat percaya diri
    labels[:20] = (labels[:20] + 1) % 6     # tapi 10% salah
    t = fit_temperature(scores, labels)
    assert isinstance(t, float)
    assert 0.05 <= t <= 20.0


def test_fit_temperature_reduces_overconfidence() -> None:
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(300, 5)) * 6
    labels = scores.argmax(axis=1)
    labels[:60] = (labels[:60] + 1) % 5     # 20% salah -> harus dilembutkan

    def nll(t: float) -> float:
        p = softmax_with_temperature(scores, t)
        return float(-np.log(p[np.arange(len(labels)), labels] + 1e-12).mean())

    t = fit_temperature(scores, labels)
    assert nll(t) < nll(1.0), "kalibrasi harus menurunkan NLL"


# ---------------------------------------------------------------------------
# Laporan konfusi
# ---------------------------------------------------------------------------
def test_confusion_report_is_sorted_by_count() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([1, 1, 0, 0, 1, 2])
    report = confusion_report(y_true, y_pred)
    counts = [r[0] for r in report]
    assert counts == sorted(counts, reverse=True)
    assert report[0][:3] == (2, C.IDX_TO_CHAR[0], C.IDX_TO_CHAR[1])


def test_confusion_report_omits_correct_predictions() -> None:
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    assert confusion_report(y_true, y_pred) == []


def test_confusion_report_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        confusion_report(np.array([0, 1]), np.array([0]))
