#!/usr/bin/env python3
"""Regenerate the synthetic dataset.

    python run_generate.py                 # uses config.N_CUSTOMERS
    python run_generate.py --n 500         # bigger cohort
    python run_generate.py --seed 7        # different draw
"""

from __future__ import annotations

import argparse

import config as cfg
from data.generator import generate


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the synthetic churn dataset.")
    ap.add_argument("--n", type=int, default=cfg.N_CUSTOMERS, help="number of customers")
    ap.add_argument("--seed", type=int, default=cfg.SEED, help="RNG seed")
    args = ap.parse_args()

    frames = generate(n_customers=args.n, seed=args.seed)

    print(f"\nWrote {cfg.DATA_OUT}\n")
    for name, df in frames.items():
        print(f"  {name:18} {len(df):>7,} rows  x {df.shape[1]:>2} cols")
    print(f"  {'latent_truth.json':18} {'(held out)':>7}")

    labels = frames["labels"]
    rate = labels["churned"].mean()
    w = labels["window_days"]
    print(
        f"\n  cutoff {cfg.AS_OF_DATE}  |  personalised label window "
        f"{int(w.min())}-{int(w.max())}d (median {int(w.median())}d)"
        f"\n  churn base rate {rate:.1%} ({int(labels['churned'].sum())}/{len(labels)})"
    )


if __name__ == "__main__":
    main()
