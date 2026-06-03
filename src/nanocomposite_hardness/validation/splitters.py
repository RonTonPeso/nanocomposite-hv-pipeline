"""Group-aware and extrapolation-aware dataset splits."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def volume_fraction_strata(vol_frac: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Bin volume fractions for stratification (handles duplicates via rank)."""
    s = pd.Series(vol_frac.astype(float))
    # qcut can fail on too few unique values — fall back to cut
    try:
        binned, _ = pd.qcut(s, q=n_bins, labels=False, retbins=True, duplicates="drop")
    except ValueError:
        binned = pd.cut(s, bins=min(n_bins, max(2, s.nunique())), labels=False)
    return binned.astype(int).to_numpy()


def stratified_group_kfold_indices(
    groups: np.ndarray,
    y_strata: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) respecting paper groups and VF strata."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train_idx, test_idx in sgkf.split(np.zeros(len(groups)), y_strata, groups):
        yield train_idx, test_idx


def extrapolation_volume_fraction_mask(
    vol_frac: np.ndarray,
    *,
    train_quantile: float = 0.9,
) -> tuple[np.ndarray, float]:
    """Return boolean mask for rows above in-distribution training ceiling (VF extrapolation)."""
    thresh = float(np.quantile(vol_frac.astype(float), train_quantile))
    mask = vol_frac.astype(float) > thresh
    return mask, thresh
