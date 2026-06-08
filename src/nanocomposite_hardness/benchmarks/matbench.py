"""Matbench elastic-moduli benchmark for the composition + gradient-boosting pipeline.

Runs the project's Magpie composition features and its default models on the
``matbench_log_gvrh`` (shear) and ``matbench_log_kvrh`` (bulk) tasks, under Matbench's
official fixed folds, so the resulting MAE is directly comparable to the public leaderboard.

We deliberately use composition-only features (no crystal structure), which is the honest
ceiling for experimental composite data where structures are unavailable. Expect to land
mid-pack, behind structure GNNs.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from matminer.featurizers.composition import ElementProperty
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from nanocomposite_hardness.models.rf import fit_rf, predict_rf
from nanocomposite_hardness.models.xgb import fit_xgb, predict_xgb

# Matbench's canonical CV: 5-fold, shuffled, with this exact seed. Reproducing it here
# (instead of depending on the `matbench` package) keeps fold membership leaderboard-comparable.
MATBENCH_N_SPLITS = 5
MATBENCH_SEED = 18012019

# Supported tasks: short name -> (matminer dataset id, target column).
TASKS: dict[str, tuple[str, str]] = {
    "log_gvrh": ("matbench_log_gvrh", "log10(G_VRH)"),
    "log_kvrh": ("matbench_log_kvrh", "log10(K_VRH)"),
}

# Published leaderboard MAE (log10 GPa), cited reference values, NOT recomputed here.
# Source: matbench.materialsproject.org leaderboards for these tasks.
LEADERBOARD_REFERENCE: dict[str, dict[str, float]] = {
    "log_gvrh": {
        "coNGN (structure GNN)": 0.0670,
        "coGN (structure GNN)": 0.0689,
        "ALIGNN (structure GNN)": 0.0715,
        "MODNet (composition+structure)": 0.0731,
        "Automatminer (composition+structure)": 0.0855,
    },
    "log_kvrh": {
        "coNGN (structure GNN)": 0.0491,
        "coGN (structure GNN)": 0.0535,
        "ALIGNN (structure GNN)": 0.0568,
        "MODNet (composition+structure)": 0.0548,
        "Automatminer (composition+structure)": 0.0680,
    },
}


def official_folds() -> KFold:
    """Matbench's fixed 5-fold split (seeded), for leaderboard-comparable scoring."""
    return KFold(n_splits=MATBENCH_N_SPLITS, shuffle=True, random_state=MATBENCH_SEED)


def featurize_compositions(structures: list, *, preset: str = "magpie") -> tuple[np.ndarray, list[str]]:
    """Magpie ElementProperty features from each structure's composition.

    Same descriptor family as ``features/composition.py``, applied to a single composition
    (Matbench has one composition per row, not a matrix/reinforcement pair).
    """
    fe = ElementProperty.from_preset(preset, impute_nan=True)
    comps = [s.composition for s in structures]
    feats = fe.featurize_many(comps, pbar=False)
    return np.asarray(feats, dtype=float), list(fe.feature_labels())


def load_matbench_task(
    task: str,
    *,
    cache_dir: str | Path = "data/processed",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a Matbench task and return (X, y, feature_names).

    Features are cached to ``cache_dir/matbench_<task>_features.parquet`` so reruns skip the
    (single-threaded, minutes-long) Magpie featurization. Row order is the dataset's canonical
    load order, which the official folds rely on.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(TASKS)}")
    dataset_id, target_col = TASKS[task]

    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"matbench_{task}_features.parquet"

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        y = cached["__target__"].to_numpy(dtype=float)
        feat_cols = [c for c in cached.columns if c != "__target__"]
        X = cached[feat_cols].to_numpy(dtype=float)
        return X, y, feat_cols

    df = load_dataset(dataset_id)
    y = df[target_col].to_numpy(dtype=float)
    X, names = featurize_compositions(df["structure"].tolist())

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(X, columns=names)
    out["__target__"] = y
    out.to_parquet(cache_path, index=False)
    return X, y, names


def _fit_predict(model_name: str, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    """Train one model on a fold and predict. Reuses the project's model wrappers."""
    if model_name in ("sklearn_hgb", "xgboost", "lightgbm"):
        model = fit_xgb(X_tr, y_tr, backend=model_name, random_state=0)
        return predict_xgb(model, X_te)
    if model_name == "rf":
        model = fit_rf(X_tr, y_tr, random_state=0)
        return predict_rf(model, X_te)
    raise ValueError(f"Unknown model {model_name!r}")


def run_task(
    task: str,
    *,
    model_name: str = "sklearn_hgb",
    cache_dir: str | Path = "data/processed",
) -> dict:
    """Run one task across the official folds; return per-fold and aggregate MAE."""
    X, y, names = load_matbench_task(task, cache_dir=cache_dir)

    fold_mae: list[float] = []
    t0 = time.time()
    for train_idx, test_idx in official_folds().split(X):
        pred = _fit_predict(model_name, X[train_idx], y[train_idx], X[test_idx])
        fold_mae.append(float(mean_absolute_error(y[test_idx], pred)))
    elapsed = time.time() - t0

    return {
        "task": task,
        "matbench_dataset": TASKS[task][0],
        "model": model_name,
        "features": "magpie_composition_only",
        "n_samples": int(len(y)),
        "n_features": int(len(names)),
        "fold_mae": fold_mae,
        "mae_mean": float(np.mean(fold_mae)),
        "mae_std": float(np.std(fold_mae)),
        "elapsed_sec": round(elapsed, 1),
        "leaderboard_reference_mae": LEADERBOARD_REFERENCE.get(task, {}),
    }
