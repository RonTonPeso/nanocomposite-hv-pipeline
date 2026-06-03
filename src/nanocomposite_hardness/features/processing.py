"""Processing route normalization and tabular encoding."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


_ROUTE_SYNONYMS: dict[str, str] = {
    r"spark plasma|sps|field assisted": "spark_plasma_sintering",
    r"powder metallurgy|pm\b|sinter": "powder_metallurgy",
    r"stir cast|stir-cast": "stir_casting",
    r"friction stir|fsp|fsw": "friction_stir_processing",
    r"ball mill.*sinter|milled.*sinter": "ball_milling_sintering",
    r"hot press|hip\b": "hot_pressing",
    r"electro(deposit|phoretic)|e\s*p\s*d": "electrodeposition",
    r"melt.*infilt|infiltration": "melt_infiltration",
    r"in.?situ": "in_situ",
    r"as.?cast|casting": "casting",
}


def normalize_fabrication_route(name: str) -> str:
    """Map free-text route labels to a controlled vocabulary slug."""
    s = (name or "").strip().lower()
    if not s:
        return "unknown"
    for pattern, slug in _ROUTE_SYNONYMS.items():
        if re.search(pattern, s, flags=re.IGNORECASE):
            return slug
    slug = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return slug or "unknown"


class ProcessingEncoder:
    """One-hot encode normalized routes; passthrough numeric processing columns."""

    def __init__(self, numeric_cols: list[str] | None = None):
        self.numeric_cols = numeric_cols or [
            "temp_c",
            "time_min",
            "pressure_mpa",
            "heat_treatment_c",
            "rolling_reduction_pct",
        ]
        self._ohe: OneHotEncoder | None = None
        self._route_categories: list[str] = []

    def fit(self, df: pd.DataFrame) -> ProcessingEncoder:
        routes = df["fabrication_route"].astype(str).map(normalize_fabrication_route)
        self._route_categories = sorted(routes.unique().tolist())
        self._ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._ohe.fit(np.array(self._route_categories).reshape(-1, 1))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._ohe is None:
            raise RuntimeError("Call fit before transform.")
        routes = df["fabrication_route"].astype(str).map(normalize_fabrication_route)
        oh = self._ohe.transform(routes.astype(str).to_numpy().reshape(-1, 1))
        cols = [f"route__{c}" for c in self._ohe.categories_[0]]
        out = pd.DataFrame(oh, columns=cols, index=df.index)
        for c in self.numeric_cols:
            if c in df.columns:
                out[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                out[c] = np.nan
        return out
