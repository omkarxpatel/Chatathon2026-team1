#!/usr/bin/env python3
"""Run the whole pipeline headlessly and print a summary.

    python run_pipeline.py              # cohort summary
    python run_pipeline.py --show CUST-0001   # full trace for one customer
"""

from __future__ import annotations

import argparse

import config as cfg
from agent.schemas import Action
from pipeline import run_cohort


def show_customer(run, customer_id: str) -> None:
    result = run.by_id().get(customer_id)
    if result is None:
        raise SystemExit(f"No such customer: {customer_id}")

    print("=" * 74)
    print(f"{result.customer_id}   {result.archetype or '(population)'}")
    print("=" * 74)
    print(f"\n[1] RISK   {result.risk.probability:.1%} ({result.risk.band.value})  "
          f"model={result.risk.model_name}")
    for a in result.risk.top(4):
        print(f"      {a.contribution:+.2f}  {a.label} = {a.value:g}")

    t = result.trajectory
    print(f"\n[2] TREND  {t.state.value}  {t.summary()}")
    print("      curve: " + " ".join(f"{v:.2f}" for v in t.curve)
          + f"   ({t.offsets[0]}d ago -> today)")
    for d in t.worsening_signals[:3]:
        print(f"      worsening: {d.label}  {d.earliest:g} -> {d.latest:g}")

    print(f"\n[3] GATE   {result.policy.outcome.value}")
    for c in result.policy.checks:
        print(f"      [{'PASS' if c.passed else 'STOP'}] {c.name}: {c.detail}")

    if result.diagnosis is None:
        print("\n[4] AGENT  not run -- the gate stopped this customer.")
        print(f"\nOUTCOME    {result.outcome_label}")
        return

    d = result.diagnosis
    print(f"\n[4] AGENT  cause={d.cause.value} ({d.cause_confidence:.0%})  "
          f"via {d.diagnoser}")
    for i, step in enumerate(d.reasoning_trace, 1):
        print(f"      {i}. {step}")
    print(f"\n    action: {d.action.value}")
    print(f"    why:    {d.action_rationale}")

    if d.has_message:
        print(f"\n    {cfg.SIMULATION_BANNER}")
        print(f"    Subject: {d.message_subject}")
        for line in (d.message_body or "").split("\n"):
            print(f"    | {line}")

    print(f"\n[5] GUARDRAILS  {result.guardrails.verdict}")
    for c in result.guardrails.checks:
        if not c.passed:
            print(f"      [FAIL] {c.name}: {c.detail}")
    print(f"\nOUTCOME    {result.outcome_label}  ->  {result.final_action.value}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="print the full trace for one customer id")
    args = ap.parse_args()

    run = run_cohort()

    if args.show:
        show_customer(run, args.show)
        return

    print(f"\n{run.fit.summary()}")
    print(f"diagnoser: {run.diagnoser_name}  |  as of {run.as_of}\n")

    table = run.table()
    print(table.head(15).to_string(index=False))

    print("\noutcomes:")
    for name, count in table["outcome"].value_counts().items():
        print(f"   {count:>4}  {name}")
    print("\nactions:")
    for name, count in table["action"].value_counts().items():
        print(f"   {count:>4}  {name}")

    print("\ntrend (direction of travel, independent of the gate):")
    for name, count in table["trend"].value_counts().items():
        print(f"   {count:>4}  {name}")

    early = [r for r in run.results if r.early_warning]
    peers = [r for r in run.results
             if r.risk.band.value != "HIGH" and not r.drifting]
    if early and peers:
        rate = sum(r.churn_label for r in early) / len(early)
        peer_rate = sum(r.churn_label for r in peers) / len(peers)
        print(f"\nearly warning: {len(early)} customers below the HIGH band whose risk "
              f"is climbing.\n   they churned {rate:.0%} of the time against {peer_rate:.0%} "
              f"for the {len(peers)} non-drifting\n   customers sitting beside them. "
              f"none of them are surfaced by risk alone.")
        print("   " + ", ".join(r.customer_id for r in
              sorted(early, key=lambda r: -r.trajectory.relative_momentum)[:8]) + ", ...")

    print("\nseeded demo archetypes:")
    seeded = table[table["archetype"] != ""]
    print(seeded[["customer_id", "archetype", "risk", "band", "trend",
                  "outcome", "cause", "action"]].to_string(index=False))


if __name__ == "__main__":
    main()
