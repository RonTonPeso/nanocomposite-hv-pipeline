"""Train / CV with honest group splits, baselines, ensembling, optional SHAP and wandb."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from nanocomposite_hardness.explain.shap_utils import shap_summary_bar
from nanocomposite_hardness.models.baseline_linear import fit_ridge_physics, predict_ridge
from nanocomposite_hardness.models.mlp import fit_mlp_torch, predict_mlp_torch
from nanocomposite_hardness.models.rf import fit_rf, predict_rf
from nanocomposite_hardness.models.xgb import fit_xgb, predict_xgb
from nanocomposite_hardness.pipeline.feature_matrix import FeatureMatrixBuilder
from nanocomposite_hardness.validation.splitters import (
    extrapolation_volume_fraction_mask,
    stratified_group_kfold_indices,
    volume_fraction_strata,
)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2_log": float(r2_score(y_true, y_pred)),
    }


def _xgb_fit_kwargs(cfg: DictConfig) -> dict:
    """Defaults so ensemble / extrapolation runs even when primary ``model`` is RF/MLP."""
    m = cfg.model
    return {
        "n_estimators": int(getattr(m, "n_estimators", 300)),
        "max_depth": int(getattr(m, "max_depth", 4)),
        "learning_rate": float(getattr(m, "learning_rate", 0.06)),
        "subsample": float(getattr(m, "subsample", 0.9)),
        "colsample_bytree": float(getattr(m, "colsample_bytree", 0.85)),
        "reg_lambda": float(getattr(m, "reg_lambda", 1.5)),
    }


def _cv_loop(df: pd.DataFrame, cfg: DictConfig, protocol: str) -> dict[str, list[float]]:
    groups = df["source_paper_id"].astype(str).values
    vf = df["vol_frac_reinf"].astype(float).values
    strata = volume_fraction_strata(vf, n_bins=int(cfg.validation.vf_bins))

    if protocol == "group_kfold":
        splits = list(
            stratified_group_kfold_indices(
                groups,
                strata,
                n_splits=int(cfg.validation.n_splits),
                shuffle=bool(cfg.validation.shuffle),
                random_state=int(cfg.validation.random_state),
            )
        )
    else:
        kf = KFold(
            n_splits=int(cfg.validation.n_splits),
            shuffle=bool(cfg.validation.shuffle),
            random_state=int(cfg.validation.random_state),
        )
        splits = list(kf.split(np.arange(len(df))))

    fold_metrics: dict[str, list[float]] = {"xgb_rmse": [], "xgb_r2": [], "ridge_rmse": [], "rf_rmse": []}

    for train_idx, test_idx in splits:
        fb = FeatureMatrixBuilder(
            use_composition=cfg.features.use_composition,
            use_physics=cfg.features.use_physics,
            use_processing=cfg.features.use_processing,
        ).fit(df.iloc[train_idx])
        X_df, names = fb.transform(df)
        X = X_df.values.astype(float)
        y = np.log(df["hv"].clip(lower=1.0).astype(float).values)

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if cfg.model.name == "xgb":
            m = fit_xgb(
                X_tr,
                y_tr,
                random_state=int(cfg.validation.random_state),
                backend=str(getattr(cfg.model, "backend", "sklearn_hgb")),
                **_xgb_fit_kwargs(cfg),
            )
            pred = predict_xgb(m, X_te)
        elif cfg.model.name == "rf":
            m = fit_rf(
                X_tr,
                y_tr,
                n_estimators=int(cfg.model.n_estimators),
                max_depth=int(cfg.model.max_depth) if cfg.model.get("max_depth") is not None else None,
                random_state=int(cfg.validation.random_state),
            )
            pred = predict_rf(m, X_te)
        elif cfg.model.name == "mlp":
            m, meta = fit_mlp_torch(
                X_tr,
                y_tr,
                epochs=int(cfg.model.epochs),
                batch_size=int(cfg.model.batch_size),
                lr=float(cfg.model.lr),
                weight_decay=float(cfg.model.weight_decay),
                seed=int(cfg.validation.random_state),
            )
            pred = predict_mlp_torch(m, meta, X_te)
        else:
            raise ValueError(cfg.model.name)

        met = _metrics(y_te, pred)
        fold_metrics["xgb_rmse"].append(met["rmse_log"])
        fold_metrics["xgb_r2"].append(met["r2_log"])

        phys_cols = [c for c in names if c.startswith("feat_")]
        if phys_cols:
            Xp = X_df[phys_cols].values.astype(float)
            ridge = fit_ridge_physics(Xp[train_idx], y_tr, alpha=1.0)
            pr = predict_ridge(ridge, Xp[test_idx])
            mr = _metrics(y_te, pr)
            fold_metrics["ridge_rmse"].append(mr["rmse_log"])

        rf = fit_rf(X_tr, y_tr, n_estimators=200, max_depth=8, random_state=0)
        prf = predict_rf(rf, X_te)
        mrf = _metrics(y_te, prf)
        fold_metrics["rf_rmse"].append(mrf["rmse_log"])

    return fold_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def _resolve_paths(cfg: DictConfig) -> None:
    cfg.paths.canonical = str((REPO_ROOT / cfg.paths.canonical).resolve())
    cfg.paths.processed_features = str((REPO_ROOT / cfg.paths.processed_features).resolve())
    cfg.paths.artifacts = str((REPO_ROOT / cfg.paths.artifacts).resolve())


def run(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    _resolve_paths(cfg)
    df = pd.read_parquet(cfg.paths.canonical)

    protocol = str(cfg.validation.protocol)
    metrics_primary = _cv_loop(df, cfg, protocol)

    report = {
        "protocol_primary": protocol,
        "primary": {k: float(np.mean(v)) for k, v in metrics_primary.items()},
        "primary_std": {k: float(np.std(v)) for k, v in metrics_primary.items()},
    }

    if cfg.compare_splits and protocol != "random_kfold":
        metrics_rand = _cv_loop(df, cfg, "random_kfold")
        report["random_kfold"] = {k: float(np.mean(v)) for k, v in metrics_rand.items()}
        # Positive ⇒ random split reports lower error than grouped (typical literature leakage).
        report["split_gap_rmse_log_random_minus_group"] = float(
            np.mean(metrics_rand["xgb_rmse"]) - np.mean(metrics_primary["xgb_rmse"])
        )

    vf = df["vol_frac_reinf"].astype(float).values
    mask_ex, thresh = extrapolation_volume_fraction_mask(
        vf, train_quantile=float(cfg.validation.extrapolation_train_q)
    )
    if mask_ex.sum() >= 2 and (~mask_ex).sum() >= 5:
        df_in = df.loc[~mask_ex].reset_index(drop=True)
        df_ex = df.loc[mask_ex].reset_index(drop=True)
        # quick train on in-distribution, test extrapolation VF
        fb = FeatureMatrixBuilder(
            use_composition=cfg.features.use_composition,
            use_physics=cfg.features.use_physics,
            use_processing=cfg.features.use_processing,
        ).fit(df_in)
        X_in, names = fb.transform(df_in)
        X_ex, _ = FeatureMatrixBuilder(
            use_composition=cfg.features.use_composition,
            use_physics=cfg.features.use_physics,
            use_processing=cfg.features.use_processing,
        ).fit(df_in).transform(df_ex)
        y_in = np.log(df_in["hv"].clip(lower=1.0).astype(float).values)
        y_ex = np.log(df_ex["hv"].clip(lower=1.0).astype(float).values)
        backend = str(getattr(cfg.model, "backend", "sklearn_hgb"))
        m = fit_xgb(
            X_in.values.astype(float),
            y_in,
            random_state=0,
            backend=backend,
            **_xgb_fit_kwargs(cfg),
        )
        pred_ex = predict_xgb(m, X_ex.values.astype(float))
        report["extrapolation_vf"] = {"threshold_vol_frac": thresh, **_metrics(y_ex, pred_ex)}
    else:
        report["extrapolation_vf"] = {"note": "insufficient rows for VF extrapolation holdout"}

    out_dir = Path(cfg.paths.artifacts)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Final screening bundle (full data, optimistic encoder — for virtual screening)
    fb_full = FeatureMatrixBuilder(
        use_composition=cfg.features.use_composition,
        use_physics=cfg.features.use_physics,
        use_processing=cfg.features.use_processing,
    ).fit(df)
    X_full_df, names = fb_full.transform(df)
    X_full = X_full_df.values.astype(float)
    y_full = np.log(df["hv"].clip(lower=1.0).astype(float).values)

    rng = np.random.default_rng(int(cfg.bootstrap_seed))
    members = []
    for b in range(int(cfg.ensemble_members)):
        idx = rng.integers(0, len(df), size=len(df))
        members.append(
            fit_xgb(
                X_full[idx],
                y_full[idx],
                random_state=b,
                backend=str(getattr(cfg.model, "backend", "sklearn_hgb")),
                **_xgb_fit_kwargs(cfg),
            )
        )

    bundle = {
        "builder_cfg": OmegaConf.to_container(cfg.features, resolve=True),
        "feature_names": names,
        "models": members,
    }
    joblib.dump(bundle, out_dir / "screening_bundle.joblib")
    joblib.dump(fb_full, out_dir / "feature_builder_full.joblib")

    if len(members) > 0:
        try:
            shap_summary_bar(
                members[0],
                X_full,
                names,
                out_dir / "shap_summary_bar.png",
                max_display=int(cfg.shap_max_display),
            )
        except Exception as e:  # pragma: no cover
            print("SHAP skipped:", e)

    if cfg.use_wandb:
        try:
            import wandb

            wandb.init(project=str(cfg.wandb_project), config=OmegaConf.to_container(cfg, resolve=True))
            wandb.log({f"cv/{k}": v for k, v in report.get("primary", {}).items()})
            wandb.finish()
        except Exception as e:  # pragma: no cover
            print("wandb disabled or failed:", e)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
    run(cfg)
