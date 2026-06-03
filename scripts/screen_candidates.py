"""Virtual screening: featurize candidates, ensemble mean/std, ranked shortlist."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nanocomposite_hardness.pipeline.feature_matrix import FeatureMatrixBuilder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True, help="Parquet/CSV with same schema as canonical (minus hv).")
    ap.add_argument("--bundle", type=Path, default=Path("artifacts/screening_bundle.joblib"))
    ap.add_argument("--builder", type=Path, default=Path("artifacts/feature_builder_full.joblib"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/screen_ranked.parquet"))
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--max-std", type=float, default=None, help="Optional drop rows with ensemble std above this (log-HV scale).")
    args = ap.parse_args()

    if args.candidates.suffix.lower() in (".parquet", ".pq"):
        cand = pd.read_parquet(args.candidates)
    else:
        cand = pd.read_csv(args.candidates)

    fb: FeatureMatrixBuilder = joblib.load(args.builder)
    X_df, names = fb.transform(cand)
    X = X_df.values.astype(float)

    bundle = joblib.load(args.bundle)
    models = bundle["models"]
    preds = np.stack([m.predict(X) for m in models], axis=0)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)

    out = cand.copy()
    out["pred_log_hv_mean"] = mean
    out["pred_log_hv_std"] = std
    out["pred_hv_mean"] = np.exp(mean)

    if args.max_std is not None:
        out = out[out["pred_log_hv_std"] <= float(args.max_std)]

    out = out.sort_values("pred_hv_mean", ascending=False).head(int(args.top_k))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Wrote top-{args.top_k} to {args.out}")


if __name__ == "__main__":
    main()
