"""Raw CSV → canonical parquet (units, volume fraction)."""

from __future__ import annotations

import argparse
from pathlib import Path

from nanocomposite_hardness.io.canonical import build_canonical_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/interim/canonical.parquet"))
    p.add_argument("--gpa-to-hv", type=float, default=101.97)
    args = p.parse_args()
    build_canonical_dataset(args.raw, args.out, gpa_to_hv=args.gpa_to_hv)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
