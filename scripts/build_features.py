"""Canonical parquet → processed feature matrix (full-dataset processing encoder)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nanocomposite_hardness.pipeline.feature_matrix import FeatureMatrixBuilder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, default=Path("data/interim/canonical.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/features.parquet"))
    ap.add_argument("--bundle", type=Path, default=Path("artifacts/feature_builder.joblib"))
    args = ap.parse_args()

    df = pd.read_parquet(args.canonical)
    fb = FeatureMatrixBuilder(
        use_composition=True,
        use_physics=True,
        use_processing=True,
    ).fit(df)
    X, names = fb.transform(df)
    y = np.log(df["hv"].clip(lower=1.0))
    meta = df[["sample_id", "source_paper_id", "matrix_formula", "reinforcement_formula", "hv"]].copy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat([meta.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    out["y_log_hv"] = y.values
    out.to_parquet(args.out, index=False)
    (args.out.parent / "feature_names.json").write_text(json.dumps(names), encoding="utf-8")

    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"builder": fb, "feature_names": names}, args.bundle)
    print(f"Wrote {args.out} and {args.bundle}")


if __name__ == "__main__":
    main()
