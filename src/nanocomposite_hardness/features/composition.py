"""Matminer-based composition features (Magpie / ElementProperty)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition


class CompositionFeaturizer:
    """ElementProperty (Magpie preset) for matrix and reinforcement separately."""

    def __init__(self, preset: str = "magpie"):
        self.preset = preset
        self._matrix_fe = ElementProperty.from_preset(preset, impute_nan=True)
        self._reinf_fe = ElementProperty.from_preset(preset, impute_nan=True)

    @staticmethod
    def _safe_comp(s: str) -> Composition:
        s = (s or "").strip()
        if not s:
            return Composition("H")  # degenerate placeholder; caller should filter
        try:
            return Composition(s)
        except Exception:
            return Composition("H")

    def featurize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expect columns ``matrix_formula`` and ``reinforcement_formula``."""
        m_comps = [self._safe_comp(x) for x in df["matrix_formula"].astype(str)]
        r_comps = [self._safe_comp(x) for x in df["reinforcement_formula"].astype(str)]

        m_feats = self._matrix_fe.featurize_many(m_comps, pbar=False)
        r_feats = self._reinf_fe.featurize_many(r_comps, pbar=False)

        m_labels = [f"m_{lbl}" for lbl in self._matrix_fe.feature_labels()]
        r_labels = [f"r_{lbl}" for lbl in self._reinf_fe.feature_labels()]

        out = pd.DataFrame(np.hstack([m_feats, r_feats]), columns=m_labels + r_labels, index=df.index)
        if "vol_frac_reinf" in df.columns:
            out["vol_frac_reinf"] = df["vol_frac_reinf"].astype(float).values
        return out
