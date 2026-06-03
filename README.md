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
python scripts/build_features.py
python scripts/train.py
python scripts/screen_candidates.py --candidates data/raw/candidates_demo.csv
```

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

See the plan in your brief: `configs/`, `data/{raw,interim,processed}/`, `src/nanocomposite_hardness/`, `scripts/`, `tests/`, `slurm/`, `notebooks/`.

## Honest evaluation

Always report **group-aware** CV (paper ID) alongside random splits. The gap between them diagnoses leakage from correlated literature rows.
