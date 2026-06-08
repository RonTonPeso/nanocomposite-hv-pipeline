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

## Benchmarking against Matbench

The hardness data is small and synthetic for now, so to place the pipeline against published
work we run the same Magpie composition features and default models on the Matbench elastic
moduli tasks (`matbench_log_gvrh`, `matbench_log_kvrh`, ~11k real entries each):

```bash
python scripts/benchmark_matbench.py                          # both tasks, hgb + rf
python scripts/benchmark_matbench.py --tasks log_gvrh --model sklearn_hgb
```

Scoring uses Matbench's official fixed 5-fold split (seed 18012019), so the reported MAE
(log10 GPa) is directly comparable to the public leaderboard. Results land in
`artifacts/matbench_report.json` next to cited SOTA values. These are **composition-only**
features (no crystal structure), the honest ceiling for experimental composites where
structures are unavailable; expect to sit behind structure-based GNNs. First run downloads the
datasets and caches the feature matrices to `data/processed/` so reruns are fast.

## Uncertainty quantification (conformal prediction)

A point prediction is not enough for deciding which composite to fabricate; you need calibrated
intervals. We compare three ways of putting an interval around the bootstrap-ensemble prediction
(naive Gaussian from the ensemble std, split conformal, normalized/locally-adaptive conformal)
and measure empirical coverage **and** mean width under different split regimes:

```bash
python scripts/evaluate_uncertainty.py --target matbench      # quantitative (large N)
python scripts/evaluate_uncertainty.py --target hardness      # illustrative (N=24)
```

Split conformal is implemented from scratch (no MAPIE) using the finite-sample order statistic
k = ceil((n+1)(1-alpha)), see `src/nanocomposite_hardness/uncertainty/conformal.py`. The
demonstrable findings on Matbench (`artifacts/uncertainty_matbench.json`):

- **Naive intervals under-cover** badly (~0.33-0.43 empirical vs 0.90 nominal): the ensemble std
  is not calibrated.
- **Split conformal restores nominal coverage** on a random (exchangeable) split (~0.90).
- **Coverage collapses under extrapolation** (test = highest-modulus tail), even for conformal,
  because the exchangeability assumption breaks. This is exactly the distribution-shift failure
  the project's grouped CV and volume-fraction extrapolation holdout are built to expose.

Always read coverage next to width: an interval can only be trusted if it hits its nominal
coverage without being uselessly wide.

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
  benchmarks/ Matbench elastic-moduli benchmark (leaderboard-comparable)
  uncertainty/ conformal prediction intervals + coverage diagnostics
scripts/                      ingest, build_features, train, tune_optuna, screen_candidates, benchmark_matbench, evaluate_uncertainty
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
