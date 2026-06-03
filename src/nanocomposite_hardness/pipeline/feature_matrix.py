"""Combine feature buckets into a single design matrix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nanocomposite_hardness.features.composition import CompositionFeaturizer
from nanocomposite_hardness.features.physics import physics_feature_frame
from nanocomposite_hardness.features.processing import ProcessingEncoder


class FeatureMatrixBuilder:
    def __init__(
        self,
        *,
        use_composition: bool = True,
        use_physics: bool = True,
        use_processing: bool = True,
    ):
        self.use_composition = use_composition
        self.use_physics = use_physics
        self.use_processing = use_processing
        self._comp_fe: CompositionFeaturizer | None = None
        self._proc_enc: ProcessingEncoder | None = None

    def fit(self, df: pd.DataFrame) -> FeatureMatrixBuilder:
        if self.use_composition:
            self._comp_fe = CompositionFeaturizer()
        if self.use_processing:
            self._proc_enc = ProcessingEncoder().fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        parts: list[pd.DataFrame] = []
        if self.use_composition:
            assert self._comp_fe is not None
            parts.append(self._comp_fe.featurize_dataframe(df))
        if self.use_physics:
            parts.append(physics_feature_frame(df))
        if self.use_processing:
            assert self._proc_enc is not None
            parts.append(self._proc_enc.transform(df))
        X = pd.concat(parts, axis=1)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X, list(X.columns)

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        return self.fit(df).transform(df)

    @staticmethod
    def save(X: pd.DataFrame, feature_names: list[str], y: pd.Series, meta: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        out = pd.concat([meta.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        out["y_log_hv"] = y.reset_index(drop=True).values
        out.to_parquet(path, index=False)
        (path.parent / "feature_names.json").write_text(json.dumps(feature_names), encoding="utf-8")


def load_feature_matrix(path: Path) -> tuple[pd.DataFrame, list[str], str]:
    df = pd.read_parquet(path)
    names_path = path.parent / "feature_names.json"
    feature_names = json.loads(names_path.read_text(encoding="utf-8"))
    return df, feature_names, "y_log_hv"
