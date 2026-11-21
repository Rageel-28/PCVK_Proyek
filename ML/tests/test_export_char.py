"""Tes ekspor model karakter ke JSON untuk Dart.

Kontrak Python↔Dart adalah berkas JSON ini. Kalau urutan fitur, charset,
atau konvensi berubah di satu sisi tanpa sisi lain, OCR akan salah total
tanpa error. Karena itu strukturnya diuji ketat.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C
from ocr.export_char import build_payload, predict_proba, write_model


class _StubClassifier:
    """Classifier palsu: skor = jumlah tinta di beberapa piksel saja."""

    def __init__(self, n_classes: int = len(C.CHARSET), dim: int = C.FEATURE_DIM):
        rng = np.random.default_rng(0)
        self.coef_ = rng.normal(size=(n_classes, dim)).astype(np.float32)
        self.intercept_ = rng.normal(size=n_classes).astype(np.float32)
        self.classes_ = np.arange(n_classes)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_.T + self.intercept_


@pytest.fixture(scope="module")
def payload() -> dict:
    return build_payload(_StubClassifier(), temperature=1.5)


def test_payload_declares_the_contract(payload: dict) -> None:
    assert payload["version"] == 1
    assert payload["charset"] == C.CHARSET
    assert payload["char_size"] == C.CHAR_SIZE
    assert payload["pad"] == C.PAD
    assert payload["feature_dim"] == C.FEATURE_DIM
    assert payload["temperature"] == pytest.approx(1.5)


def test_payload_describes_every_feature_block(payload: dict) -> None:
    f = payload["features"]
    assert f["hog"] == {
        "cell": C.HOG_CELL,
        "bins": C.HOG_BINS,
        "block": C.HOG_BLOCK,
        "dim": C.HOG_DIM,
    }
    assert f["zoning"]["grid"] == C.ZONING_GRID
    assert f["zoning"]["dim"] == C.ZONING_DIM
    assert f["projection"]["cols"] == C.PROJECTION_COLS
    assert f["projection"]["rows"] == C.PROJECTION_ROWS
    assert f["projection"]["dim"] == C.PROJECTION_DIM
    assert f["hole"]["max"] == C.HOLE_MAX
    assert f["hole"]["dim"] == C.HOLE_DIM
    total = sum(f[k]["dim"] for k in ("hog", "zoning", "projection", "hole"))
    assert total == C.FEATURE_DIM


def test_payload_coefficients_are_flat_and_rectangular(payload: dict) -> None:
    n = len(C.CHARSET)
    assert len(payload["intercept"]) == n
    assert len(payload["coef"]) == n * C.FEATURE_DIM
    assert all(isinstance(v, float) for v in payload["coef"][:10])


def test_payload_is_json_serialisable_and_small(payload: dict) -> None:
    text = json.dumps(payload, separators=(",", ":"))
    assert len(text) < 4_000_000, f"model terlalu besar: {len(text)/1e6:.1f} MB"
    assert json.loads(text)["charset"] == C.CHARSET


def test_write_model_roundtrips(tmp_path: Path) -> None:
    out = write_model(_StubClassifier(), tmp_path / "char_clf.json", temperature=1.2)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert len(data["coef"]) == len(C.CHARSET) * C.FEATURE_DIM


def test_predict_proba_rows_sum_to_one() -> None:
    clf = _StubClassifier()
    X = np.random.default_rng(1).normal(size=(5, C.FEATURE_DIM)).astype(np.float32)
    probs = predict_proba(clf, X, temperature=1.0)
    assert probs.shape == (5, len(C.CHARSET))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_predict_proba_matches_softmax_of_decision_function() -> None:
    clf = _StubClassifier()
    X = np.random.default_rng(2).normal(size=(3, C.FEATURE_DIM)).astype(np.float32)
    scores = clf.decision_function(X)
    probs = predict_proba(clf, X, temperature=1.0)
    expected = np.exp(scores - scores.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(probs, expected, atol=1e-6)
