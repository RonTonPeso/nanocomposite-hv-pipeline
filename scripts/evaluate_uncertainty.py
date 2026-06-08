"""Evaluate conformal prediction-interval coverage under different split regimes.

Headline (quantitative) target is Matbench, where N is large enough for coverage to mean
something. The hardness pipeline is included as the intended application, but with N=24 its
numbers are illustrative only.

    python scripts/evaluate_uncertainty.py --target matbench                 # both tasks
    python scripts/evaluate_uncertainty.py --target matbench --tasks log_gvrh
    python scripts/evaluate_uncertainty.py --target hardness

For each regime we compare three intervals at the nominal level: naive ensemble-Gaussian,
split conformal, and normalized (locally-adaptive) conformal. We report coverage AND mean width,
since coverage alone is gameable by widening intervals.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from nanocomposite_hardness.benchmarks.matbench import TASKS, load_matbench_task
from nanocomposite_hardness.pipeline.feature_matrix import FeatureMatrixBuilder
from nanocomposite_hardness.uncertainty.conformal import evaluate_uq
from nanocomposite_hardness.validation.splitters import extrapolation_volume_fraction_mask

REPO_ROOT = Path(__file__).resolve().parents[1]
_METHODS = ["naive", "split_conformal", "normalized_conformal"]


# --- split constructors: each returns (train_idx, cal_idx, test_idx) ---

def random_split(n: int, seed: int, *, test_frac: float = 0.2, cal_frac: float = 0.2):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    n_cal = max(1, int(round(n * cal_frac)))
    return perm[n_test + n_cal:], perm[n_test:n_test + n_cal], perm[:n_test]


def extrapolation_split(values: np.ndarray, seed: int, *, test_frac: float = 0.2, cal_frac_of_rest: float = 0.25):
    """Test = highest-value tail (property-space extrapolation); train/cal from the rest."""
    order = np.argsort(values)
    n = len(values)
    n_test = max(1, int(round(n * test_frac)))
    test = order[-n_test:]
    rest = order[:-n_test]
    rng = np.random.default_rng(seed)
    rng.shuffle(rest)
    n_cal = max(1, int(round(len(rest) * cal_frac_of_rest)))
    return rest[n_cal:], rest[:n_cal], test


def grouped_split(groups: np.ndarray, seed: int, *, test_frac: float = 0.2, cal_frac: float = 0.2):
    """Hold out whole groups (no group spans train/cal/test) so there is no leakage."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist())))
    rng.shuffle(uniq)
    n_g = len(uniq)
    n_test = max(1, int(round(n_g * test_frac)))
    n_cal = max(1, int(round(n_g * cal_frac)))
    test_g = set(uniq[:n_test].tolist())
    cal_g = set(uniq[n_test:n_test + n_cal].tolist())
    idx = np.arange(len(groups))
    test = idx[np.isin(groups, list(test_g))]
    cal = idx[np.isin(groups, list(cal_g))]
    train = idx[~np.isin(groups, list(test_g | cal_g))]
    return train, cal, test


def _aggregate(runs: list[dict]) -> dict:
    out = {}
    for m in _METHODS:
        cov = [r[m]["coverage"] for r in runs]
        wid = [r[m]["mean_width"] for r in runs]
        out[m] = {
            "coverage_mean": float(np.mean(cov)),
            "coverage_std": float(np.std(cov)),
            "width_mean": float(np.mean(wid)),
            "width_std": float(np.std(wid)),
        }
    return out


def _run_regime(X, y, splits, *, alpha, n_members) -> dict:
    runs = [
        evaluate_uq(X, y, tr, cal, te, alpha=alpha, n_members=n_members, seed=s)
        for s, (tr, cal, te) in enumerate(splits)
    ]
    return {
        "nominal_coverage": 1.0 - alpha,
        "n_seeds": len(runs),
        "n_test": int(np.mean([r["n_test"] for r in runs])),
        "methods": _aggregate(runs),
    }


def run_matbench(tasks, *, alpha, seeds, n_members) -> dict:
    report = {}
    for task in tasks:
        X, y, _ = load_matbench_task(task, cache_dir=REPO_ROOT / "data/processed")
        report[task] = {
            "exchangeable": _run_regime(
                X, y, [random_split(len(y), s) for s in range(seeds)],
                alpha=alpha, n_members=n_members,
            ),
            "extrapolation": _run_regime(
                X, y, [extrapolation_split(y, s) for s in range(seeds)],
                alpha=alpha, n_members=n_members,
            ),
        }
    return report


def run_hardness(*, alpha, seeds, n_members) -> dict:
    df = pd.read_parquet(REPO_ROOT / "data/interim/canonical.parquet")
    # Builder fit on full data (illustrative; processing encoder leakage is negligible at N=24).
    fb = FeatureMatrixBuilder().fit(df)
    X_df, _ = fb.transform(df)
    X = X_df.values.astype(float)
    y = np.log(df["hv"].clip(lower=1.0).astype(float).values)
    groups = df["source_paper_id"].astype(str).values
    vf = df["vol_frac_reinf"].astype(float).values
    mask_ex, thresh = extrapolation_volume_fraction_mask(vf)

    def vf_split(seed):
        test = np.where(mask_ex)[0]
        rest = np.where(~mask_ex)[0]
        rng = np.random.default_rng(seed)
        rng.shuffle(rest)
        n_cal = max(1, int(round(len(rest) * 0.3)))
        return rest[n_cal:], rest[:n_cal], test

    return {
        "_caveat": "N=24 synthetic rows: coverage here is directional, not statistically meaningful.",
        "random": _run_regime(X, y, [random_split(len(y), s) for s in range(seeds)],
                              alpha=alpha, n_members=n_members),
        "grouped_by_paper": _run_regime(X, y, [grouped_split(groups, s) for s in range(seeds)],
                                       alpha=alpha, n_members=n_members),
        "vf_extrapolation": {
            "vf_threshold": float(thresh),
            **_run_regime(X, y, [vf_split(s) for s in range(seeds)],
                          alpha=alpha, n_members=n_members),
        },
    }


def _sanitize(obj):
    """Replace non-finite floats with None so the JSON stays valid (inf widths can occur)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["matbench", "hardness"], default="matbench")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ensemble", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.target == "matbench":
        report = run_matbench(args.tasks, alpha=args.alpha, seeds=args.seeds, n_members=args.ensemble)
    else:
        report = run_hardness(alpha=args.alpha, seeds=args.seeds, n_members=args.ensemble)

    out = args.out or (REPO_ROOT / f"artifacts/uncertainty_{args.target}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_sanitize(report), indent=2), encoding="utf-8")

    # Console summary
    print(f"\n=== {args.target} (nominal coverage {1 - args.alpha:.2f}) ===")
    def _print_regime(name, reg):
        print(f"  [{name}] n_test~{reg['n_test']}")
        for m in _METHODS:
            s = reg["methods"][m]
            print(f"    {m:22s} coverage={s['coverage_mean']:.3f}  width={s['width_mean']:.3f}")
    if args.target == "matbench":
        for task, regs in report.items():
            print(f"-- {task} --")
            _print_regime("exchangeable", regs["exchangeable"])
            _print_regime("extrapolation", regs["extrapolation"])
    else:
        for name in ("random", "grouped_by_paper", "vf_extrapolation"):
            _print_regime(name, report[name])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
