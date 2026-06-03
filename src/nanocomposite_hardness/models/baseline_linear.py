"""Ridge regression on a chosen feature subset (e.g. physics-only sanity check)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_ridge_physics(
    X: np.ndarray,
    y_log_hv: np.ndarray,
    *,
    alpha: float = 1.0,
) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=0)),
        ]
    ).fit(X, y_log_hv)


def predict_ridge(model: Pipeline, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
