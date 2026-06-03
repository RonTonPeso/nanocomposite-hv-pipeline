# Nanocomposite Vickers Hardness — ML Pipeline

End-to-end pipeline for predicting Vickers microhardness (HV) of metal–ceramic nanocomposites for computational materials discovery.

## Quick start

```bash
cd nanocomposite-hardness
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Place literature-extracted rows in `data/raw/` (see `data/raw/schema.md`), then:

```bash
python scripts/ingest.py --raw data/raw/your_extract.csv --out data/interim/canonical.parquet
python scripts/train.py
python scripts/screen_candidates.py --candidates data/raw/candidates_demo.csv
```

`scripts/train.py` reads the canonical parquet directly and rebuilds the feature
matrix per CV fold (the processing encoder is fit on training rows only, so CV stays
leakage-free). `scripts/build_features.py` is a separate convenience step that writes a
single full-dataset feature matrix to `data/processed/` for EDA and the notebooks; it is
not consumed by training.

Config overrides (Hydra `compose` CLI, no `@hydra.main` — compatible with Python 3.14):

```bash
python scripts/train.py validation=random model=rf
```

On macOS without Homebrew `libomp`, tree libraries may fail to load; the default boosting backend is `sklearn_hgb` (`configs/model/xgb.yaml`). Install `libomp` and switch to `lightgbm` or `xgboost` for full performance on Linux/HPC.

Optional tuning:

```bash
python scripts/tune_optuna.py --trials 40 --backend sklearn_hgb
```

## Project layout

```
configs/                      Hydra configs (model, validation, features, paths)
data/{raw,interim,processed}/ raw extracts → canonical parquet → feature matrix
src/nanocomposite_hardness/
  io/         unit normalization, weight→volume fraction, canonical assembly
  features/   composition (matminer), physics proxies, processing encoder
  pipeline/   feature matrix builder
  models/     xgb/lgbm/hgb, random forest, ridge, torch MLP
  validation/ stratified group k-fold, VF extrapolation splits
  explain/    SHAP summary plots
scripts/                      ingest, build_features, train, tune_optuna, screen_candidates
tests/                        unit tests (units, physics features)
slurm/                        HPC array job for multi-seed training
notebooks/                    EDA, feature iteration, SHAP analysis
```

## Notes to Self

Always report **group-aware** CV ,paper ID, alongside random splits. The gap between them
diagnoses leakage from related literature rows `train.py` writes
`split_gap_rmse_log_group_minus_random` to `artifacts/cv_report.json`, where a pos
val means the grouped error is higher, random splits are optimistic - leakage.
A volume-fraction extrapolation holdout (`extrapolation_vf`) tests
generalization beyond the training reinforcement-loading range.
