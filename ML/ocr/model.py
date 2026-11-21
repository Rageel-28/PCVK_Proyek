"""Model classifier linear yang dimuat dari JSON (padanan Dart)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import ocr.config as C


class LinearCharModel:
    """Classifier linear + softmax bersuhu, dibaca dari ``char_clf.json``."""

    def __init__(self, payload: dict):
        self.version = payload["version"]
        self.charset: str = payload["charset"]
        self.char_size: int = payload["char_size"]
        self.pad: int = payload["pad"]
        self.feature_dim: int = payload["feature_dim"]
        self.temperature: float = float(payload["temperature"])

        n_classes = len(self.charset)
        coef = np.asarray(payload["coef"], dtype=np.float32)
        if coef.size != n_classes * self.feature_dim:
            raise ValueError(
                f"panjang koefisien {coef.size} != "
                f"{n_classes} x {self.feature_dim}"
            )
        self.coef = coef.reshape(n_classes, self.feature_dim)
        self.intercept = np.asarray(payload["intercept"], dtype=np.float32)

    @classmethod
    def load(cls, path: str | Path) -> "LinearCharModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data)

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return features @ self.coef.T + self.intercept

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scores = self.decision_function(np.asarray(features, dtype=np.float32))
        z = scores.astype(np.float64) / self.temperature
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)


def load_model(path: str | Path | None = None) -> LinearCharModel:
    if path is None:
        path = C.MODEL_DIR / "char_clf.json"
    return LinearCharModel.load(path)
