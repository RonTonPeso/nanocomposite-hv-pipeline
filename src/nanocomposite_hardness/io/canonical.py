"""Assemble a single canonical table from raw extraction rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nanocomposite_hardness.io.densities import weight_fraction_to_volume_fraction_vec
from nanocomposite_hardness.io.units import normalize_hardness_column


REQUIRED_COLS = [
    "sample_id",
    "source_paper_id",
    "matrix_formula",
    "reinforcement_formula",
    "fabrication_route",
]


def build_canonical_dataset(
    raw_path: str | Path,
    out_path: str | Path,
    *,
    rho_matrix_col: str = "rho_matrix_g_cm3",
    rho_reinf_col: str = "rho_reinf_g_cm3",
    vol_frac_col: str = "vol_frac_reinf",
    wt_frac_col: str | None = "wt_frac_reinf",
    hardness_value_col: str = "hardness_value",
    hardness_unit_col: str = "hardness_unit",
    gpa_to_hv: float = 101.97,
) -> pd.DataFrame:
    """Load raw CSV/parquet, normalize units, resolve volume fraction, write parquet."""
    raw_path = Path(raw_path)
    if raw_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(raw_path)
    else:
        df = pd.read_csv(raw_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = normalize_hardness_column(
        df,
        value_col=hardness_value_col,
        unit_col=hardness_unit_col,
        out_col="hv",
        gpa_to_hv=gpa_to_hv,
    )

    if vol_frac_col in df.columns and df[vol_frac_col].notna().any():
        phi = df[vol_frac_col].astype(float)
    elif wt_frac_col and wt_frac_col in df.columns:
        if rho_matrix_col not in df.columns or rho_reinf_col not in df.columns:
            raise ValueError("Weight fraction conversion requires density columns.")
        w = df[wt_frac_col].astype(float).to_numpy()
        rho_m = df[rho_matrix_col].astype(float).to_numpy()
        rho_r = df[rho_reinf_col].astype(float).to_numpy()
        phi = pd.Series(weight_fraction_to_volume_fraction_vec(w, rho_m, rho_r), index=df.index)
    else:
        raise ValueError("Provide either vol_frac_reinf or wt_frac_reinf + densities.")

    df["vol_frac_reinf"] = phi

    numeric_optional = [
        "temp_c",
        "time_min",
        "pressure_mpa",
        "grain_size_nm",
        "particle_size_nm",
        "heat_treatment_c",
        "rolling_reduction_pct",
    ]
    for c in numeric_optional:
        if c not in df.columns:
            df[c] = np.nan

    if "missingness_mask" not in df.columns:
        df["missingness_mask"] = ""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df
