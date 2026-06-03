"""XGBoost / LightGBM wrappers (lazy import so XGBoost is optional on hosts missing libomp)."""

from __future__ import annotations

from typing import Any

import numpy as np


def fit_xgb(
    X: np.ndarray,
    y: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
    n_estimators: int = 400,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    random_state: int = 0,
    backend: str = "sklearn_hgb",
):
    params = params or {}
    if backend == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=-1,
            **params,
        )
    elif backend == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
    elif backend == "sklearn_hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        # Maps loosely to XGB/LGBM knobs; pure sklearn — works without libomp wheels.
        model = HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=n_estimators,
            l2_regularization=reg_lambda,
            random_state=random_state,
            **params,
        )
    else:
        raise ValueError(backend)
    model.fit(X, y)
    return model


def predict_xgb(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
