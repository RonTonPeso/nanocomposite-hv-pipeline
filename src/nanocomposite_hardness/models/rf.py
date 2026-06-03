"""Random forest baseline."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def fit_rf(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 400,
    max_depth: int | None = 12,
    random_state: int = 0,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_rf(model: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
