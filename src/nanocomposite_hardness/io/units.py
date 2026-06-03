"""Normalize hardness measurements to a single Vickers HV scale."""

from __future__ import annotations

import numpy as np
import pandas as pd


def hardness_to_hv(value: float | np.ndarray, unit: str, *, gpa_to_hv: float = 101.97) -> float | np.ndarray:
    """Convert hardness to Vickers HV number (dimensionless scale used in tables).

    Parameters
    ----------
    value
        Reported hardness magnitude.
    unit
        One of ``hv`` (already Vickers number), ``gpa``, ``kgf_mm2`` (legacy kgf/mm²
        numerically aligned with HV in many papers).
    gpa_to_hv
        Linear factor HV ≈ gpa_to_hv * H_GPa when papers report microhardness in GPa.
        Tune if your literature subset uses a different convention.
    """
    u = unit.lower().strip().replace(" ", "_").replace("²", "2")
    mapping = {
        "hv": 1.0,
        "vickers": 1.0,
        "hv0.1": 1.0,
        "hv0.2": 1.0,
        "hv0.5": 1.0,
        "hv1": 1.0,
    }
    if u in mapping:
        return value * mapping[u]
    if u in ("gpa", "gigapascal"):
        return value * gpa_to_hv
    if u in ("kgf/mm2", "kgf_mm2", "kg/mm2"):
        return value
    raise ValueError(f"Unknown hardness unit: {unit!r}")


def normalize_hardness_column(
    df: pd.DataFrame,
    value_col: str = "hardness_value",
    unit_col: str = "hardness_unit",
    out_col: str = "hv",
    *,
    gpa_to_hv: float = 101.97,
) -> pd.DataFrame:
    """Vectorized normalization to ``hv`` column."""
    out = df.copy()
    units = out[unit_col].astype(str).str.lower().str.strip()
    hv_list = []
    for v, u in zip(out[value_col], units, strict=True):
        hv_list.append(hardness_to_hv(float(v), u, gpa_to_hv=gpa_to_hv))
    out[out_col] = np.asarray(hv_list, dtype=float)
    return out
