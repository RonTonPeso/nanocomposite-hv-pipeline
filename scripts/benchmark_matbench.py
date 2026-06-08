"""Benchmark the composition pipeline on Matbench elastic-moduli tasks.

Reports leaderboard-comparable MAE (official folds) for log_gvrh / log_kvrh, alongside
cited SOTA values so the numbers can be placed against published work.

    python scripts/benchmark_matbench.py                         # both tasks, hgb + rf
    python scripts/benchmark_matbench.py --tasks log_gvrh --model sklearn_hgb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanocomposite_hardness.benchmarks.matbench import TASKS, run_task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS),
        choices=list(TASKS),
        help="Matbench tasks to run (default: both).",
    )
    ap.add_argument(
        "--model",
        nargs="+",
        default=["sklearn_hgb", "rf"],
        choices=["sklearn_hgb", "xgboost", "lightgbm", "rf"],
        help="Models to evaluate (default: gradient boosting + RF baseline).",
    )
    ap.add_argument("--out", type=Path, default=Path("artifacts/matbench_report.json"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    results = []
    for task in args.tasks:
        for model_name in args.model:
            print(f"Running {task} / {model_name} ...", flush=True)
            res = run_task(task, model_name=model_name, cache_dir=args.cache_dir)
            print(
                f"  MAE = {res['mae_mean']:.4f} +/- {res['mae_std']:.4f} "
                f"(log10 GPa, {res['elapsed_sec']}s)",
                flush=True,
            )
            results.append(res)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
