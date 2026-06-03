"""Physics-informed strengthening proxies as tabular features."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pymatgen.core import Composition


def _shear_modulus_gpa_estimate(matrix_formula: str) -> float:
    """Order-of-magnitude shear modulus (GPa) from elemental heuristic.

    For discovery screening this is a coarse prior; replace with a database lookup
    for production accuracy.
    """
    try:
        comp = Composition(matrix_formula)
    except Exception:
        return 30.0
    # Rudimentary: weighted by common engineering metals (GPa scale).
    g_map = {
        "Al": 26.0,
        "Cu": 48.0,
        "Mg": 17.0,
        "Fe": 82.0,
        "Ti": 44.0,
        "Ni": 76.0,
        "Zn": 43.0,
        "Sn": 18.0,
    }
    els = list(comp.get_el_amt_dict().keys())
    if not els:
        return 30.0
    return float(np.mean([g_map.get(str(e), 35.0) for e in els]))


def hall_petch_feature(grain_size_nm: float | None, k_hp_mpa_sqrt_m: float = 0.15) -> float:
    """σ_HP ∝ d^{-1/2}; return k / sqrt(d_m) as a linear feature proxy (MPa-scale)."""
    if grain_size_nm is None:
        return 0.0
    if not math.isfinite(grain_size_nm) or grain_size_nm <= 0:
        return 0.0
    d_m = grain_size_nm * 1e-9
    return float(k_hp_mpa_sqrt_m / math.sqrt(d_m))


def orowan_stress_mpa(
    particle_size_nm: float | None,
    vol_frac: float,
    shear_modulus_gpa: float,
    burgers_vector_m: float = 2.86e-10,
    poisson: float = 0.33,
) -> float:
    """Orowan bowing stress scale (MPa) — used as a feature, not a constitutive law."""
    if particle_size_nm is None:
        return 0.0
    if not math.isfinite(particle_size_nm) or particle_size_nm <= 0 or vol_frac <= 0:
        return 0.0
    G_pa = max(shear_modulus_gpa, 1e-6)
    G = G_pa * 1e9
    d = particle_size_nm * 1e-9
    b = burgers_vector_m
    nu = poisson
    pref = (0.4 / (2.0 * math.pi * math.sqrt(max(1e-9, 1.0 - nu * nu)))) * G * b / d
    ratio = max(d / (2.0 * b), 1.001)
    ln_term = math.log(ratio)
    fv = math.sqrt(max(vol_frac, 1e-12))
    sigma_pa = pref * ln_term * fv
    return float(sigma_pa / 1e6)


def cte_mismatch_dislocation_proxy(
    delta_alpha_1_k: float,
    delta_t_k: float,
    vol_frac: float,
    particle_size_nm: float | None,
    burgers_vector_m: float = 2.86e-10,
) -> float:
    """Dimensionally consistent proxy ∝ f Δα ΔT / (b d); arbitrary geometric prefactor."""
    if particle_size_nm is None or not math.isfinite(particle_size_nm) or particle_size_nm <= 0:
        return 0.0
    d = particle_size_nm * 1e-9
    b = burgers_vector_m
    return float(12.0 * vol_frac * abs(delta_alpha_1_k) * abs(delta_t_k) / (b * d))


def physics_feature_row(row: pd.Series, matrix_formula_col: str = "matrix_formula") -> dict[str, float]:
    G = _shear_modulus_gpa_estimate(str(row.get(matrix_formula_col, "")))
    d_g = row.get("grain_size_nm")
    d_p = row.get("particle_size_nm")
    vf = float(row.get("vol_frac_reinf", 0.0) or 0.0)
    t_c = row.get("temp_c")
    delta_t = float(t_c) if t_c is not None and math.isfinite(float(t_c)) else 300.0

    # Δα placeholder: user should supply per-row ``delta_alpha_1e6_k`` when known.
    delta_alpha = row.get("delta_alpha_1e6_k")
    if delta_alpha is None or not math.isfinite(float(delta_alpha)):
        delta_alpha = 5.0  # 5e-6 /K default mismatch scale

    return {
        "feat_hall_petch": hall_petch_feature(float(d_g) if d_g is not None and pd.notna(d_g) else None),
        "feat_orowan_mpa": orowan_stress_mpa(
            float(d_p) if d_p is not None and pd.notna(d_p) else None,
            vf,
            G,
        ),
        "feat_cte_rho_proxy": cte_mismatch_dislocation_proxy(
            float(delta_alpha) * 1e-6,
            delta_t,
            vf,
            float(d_p) if d_p is not None and pd.notna(d_p) else None,
        ),
    }


def physics_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = [physics_feature_row(df.loc[i]) for i in df.index]
    return pd.DataFrame(rows, index=df.index)
