"""Optional Bayesian hyperparameter search (Optuna) for the gradient boosting backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from nanocomposite_hardness.models.xgb import fit_xgb, predict_xgb
from nanocomposite_hardness.pipeline.feature_matrix import FeatureMatrixBuilder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, default=Path("data/interim/canonical.parquet"))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--backend", default="sklearn_hgb")
    ap.add_argument("--out", type=Path, default=Path("artifacts/optuna_best.json"))
    args = ap.parse_args()

    df = pd.read_parquet(args.canonical)
    fb = FeatureMatrixBuilder().fit(df)
    X, _ = fb.transform(df)
    Xv = X.values.astype(float)
    y = np.log(df["hv"].clip(lower=1.0).astype(float).values)

    def objective(trial: optuna.Trial) -> float:
        md = trial.suggest_int("max_depth", 2, 8)
        lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        ne = trial.suggest_int("n_estimators", 100, 600)
        rl = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        kf = KFold(3, shuffle=True, random_state=0)
        scores = []
        for tr, te in kf.split(Xv):
            m = fit_xgb(
                Xv[tr],
                y[tr],
                max_depth=md,
                learning_rate=lr,
                n_estimators=ne,
                reg_lambda=rl,
                random_state=0,
                backend=args.backend,
            )
            pred = predict_xgb(m, Xv[te])
            scores.append(mean_squared_error(y[te], pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
    print("Best:", study.best_params)


if __name__ == "__main__":
    main()
