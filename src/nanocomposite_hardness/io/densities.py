"""Convert weight fractions to volume fractions using component densities."""

from __future__ import annotations

import numpy as np


def weight_fraction_to_volume_fraction(
    w_reinf: float,
    rho_matrix: float,
    rho_reinf: float,
) -> float:
    """Two-phase rule-of-mixtures inversion: w → φ for reinforcement.

    φ_reinf = (w/ρ_r) / (w/ρ_r + (1-w)/ρ_m)
    """
    if not (0.0 <= w_reinf <= 1.0):
        raise ValueError("w_reinf must be in [0, 1]")
    if rho_matrix <= 0 or rho_reinf <= 0:
        raise ValueError("Densities must be positive")
    num = w_reinf / rho_reinf
    den = num + (1.0 - w_reinf) / rho_matrix
    return float(num / den)


def weight_fraction_to_volume_fraction_vec(
    w: np.ndarray,
    rho_m: np.ndarray,
    rho_r: np.ndarray,
) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    rho_m = np.asarray(rho_m, dtype=float)
    rho_r = np.asarray(rho_r, dtype=float)
    num = w / rho_r
    den = num + (1.0 - w) / rho_m
    return num / den
