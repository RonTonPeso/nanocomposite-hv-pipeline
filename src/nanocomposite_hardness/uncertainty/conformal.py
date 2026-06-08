"""Split conformal prediction and coverage diagnostics.

We compare three ways of putting an interval around a bootstrap-ensemble point prediction:

- ``naive``: Gaussian interval mean ± z·sigma from the ensemble's predictive std. No coverage
  guarantee; in practice it under-covers because the ensemble std is not calibrated.
- ``split``: split conformal with absolute-residual nonconformity. Distribution-free marginal
  coverage *when calibration and test points are exchangeable*.
- ``normalized``: locally-adaptive (Mondrian-style) conformal that scales the residual by the
  ensemble std, giving variable-width intervals that adapt to per-point difficulty.

The point of the exercise is to show, on real data, that naive under-covers, split conformal
restores nominal coverage on random splits, and *all* methods lose coverage under distribution
shift (extrapolation), because conformal's guarantee rests on exchangeability.

Implemented from scratch (no MAPIE) so the math is explicit and dependency-light.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from nanocomposite_hardness.models.xgb import fit_xgb


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal threshold: the k-th smallest calibration residual.

    The coverage guarantee comes from using the order statistic k = ceil((n + 1)(1 - alpha))
    rather than the plain (1 - alpha) empirical quantile, which is slightly too small. We use the
    order statistic directly (not ``np.quantile``) to avoid interpolation-method ambiguity, and
    round before ``ceil`` so floating-point noise at integer boundaries (e.g. 10*0.9) does not
    bump k up by one. If k > n (too few calibration points for this alpha) the interval is
    unbounded, signalled by ``inf``.
    """
    r = np.asarray(residuals, dtype=float)
    n = r.size
    if n == 0:
        raise ValueError("Need at least one calibration residual.")
    k = math.ceil(round((n + 1) * (1.0 - alpha), 9))
    if k > n:
        return float("inf")
    return float(np.sort(r)[k - 1])


def coverage_and_width(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    """Empirical coverage and mean interval width. Report both: coverage alone is gameable."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (y_true >= lower) & (y_true <= upper)
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(upper - lower)),
    }


def fit_bootstrap_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_members: int = 8,
    backend: str = "sklearn_hgb",
    seed: int = 0,
) -> list:
    """Bagged ensemble of boosting models (same pattern as scripts/train.py screening bundle)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    members = []
    for b in range(n_members):
        idx = rng.integers(0, n, size=n)
        members.append(fit_xgb(X[idx], y[idx], random_state=b, backend=backend))
    return members


def predict_ensemble(models: list, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) over ensemble members."""
    preds = np.stack([m.predict(X) for m in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def evaluate_uq(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    alpha: float = 0.1,
    n_members: int = 8,
    backend: str = "sklearn_hgb",
    seed: int = 0,
    eps: float = 1e-6,
) -> dict:
    """Fit on train, calibrate on cal, score coverage/width on test for all three methods."""
    models = fit_bootstrap_ensemble(
        X[train_idx], y[train_idx], n_members=n_members, backend=backend, seed=seed
    )

    mean_cal, sigma_cal = predict_ensemble(models, X[cal_idx])
    mean_te, sigma_te = predict_ensemble(models, X[test_idx])
    y_cal, y_te = y[cal_idx], y[test_idx]

    # naive: Gaussian interval from ensemble std (no calibration set used)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    naive = coverage_and_width(y_te, mean_te - z * sigma_te, mean_te + z * sigma_te)

    # split conformal: absolute-residual quantile from calibration
    q_abs = conformal_quantile(np.abs(y_cal - mean_cal), alpha)
    split = coverage_and_width(y_te, mean_te - q_abs, mean_te + q_abs)

    # normalized conformal: residual scaled by ensemble std -> adaptive width
    q_norm = conformal_quantile(np.abs(y_cal - mean_cal) / (sigma_cal + eps), alpha)
    norm_half = q_norm * (sigma_te + eps)
    normalized = coverage_and_width(y_te, mean_te - norm_half, mean_te + norm_half)

    return {
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "n_train": int(len(train_idx)),
        "n_calibration": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "naive": naive,
        "split_conformal": split,
        "normalized_conformal": normalized,
    }
