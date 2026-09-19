#!/usr/bin/env python3
"""Run the whole pipeline headlessly and print a summary.

    python run_pipeline.py                    # cohort summary
    python run_pipeline.py --show CUST-0001   # the agent's full run for one customer
"""

from __future__ import annotations

import argparse

import config as cfg
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

    print(f"\n[2] GATE   {result.policy.outcome.value}")
    for c in result.policy.checks:
        print(f"      [{'PASS' if c.passed else 'STOP'}] {c.name}: {c.detail}")

    if result.agent_run is None:
        print("\n[3] AGENT  not run -- the gate stopped this customer.")
        print(f"\nOUTCOME    {result.outcome_label}")
        return

    agent = result.agent_run
    print(f"\n[3] AGENT  {agent.brain}  --  {agent.tool_calls} steps, "
          f"stopped because it {agent.stop_reason}")
    for step in agent.steps:
        print(f"\n      {step.index}. {step.thought}")
        print(f"         -> {step.tool}({', '.join(f'{k}={v!r}' for k, v in step.arguments.items())[:60]})")
        print(f"         =  {step.headline}")

    d = agent.diagnosis
    print(f"\n    cause:  {d.cause.value} ({d.cause_confidence:.0%} confidence)")
    print(f"    action: {d.action.value}")
    print(f"    why:    {d.action_rationale}")

    if d.has_message:
        print(f"\n    {cfg.SIMULATION_BANNER}")
        print(f"    Subject: {d.message_subject}")
        for line in (d.message_body or "").split("\n"):
            print(f"    | {line}")

    print(f"\n[4] GUARDRAILS  {result.guardrails.verdict}")
    for c in result.guardrails.checks:
        if not c.passed:
            print(f"      [FAIL] {c.name}: {c.detail}")
    print(f"\nOUTCOME    {result.outcome_label}  ->  {result.final_action.value}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="print the agent's full run for one customer id")
    args = ap.parse_args()

    run = run_cohort()

    if args.show:
        show_customer(run, args.show)
        return

    print(f"\n{run.fit.summary()}")
    stats = run.agent_stats()
    print(f"agent: {run.agent_name}  |  as of {run.as_of}")
    print(f"       {stats['customers_investigated']} customers investigated, "
          f"{stats['tool_calls']} tool calls ({stats['avg_steps']} per customer), "
          f"{stats['drafted']} drafts, {stats['left_alone']} left alone, "
          f"{stats['blocked']} blocked by guardrails\n")

    table = run.table()
    print(table.head(15).to_string(index=False))

    print("\noutcomes:")
    for name, count in table["outcome"].value_counts().items():
        print(f"   {count:>4}  {name}")
    print("\nactions:")
    for name, count in table["action"].value_counts().items():
        print(f"   {count:>4}  {name}")
    print("\ntools the agent reached for:")
    for row in run.tool_usage():
        print(f"   {row['calls']:>4}  {row['tool']}")

    print("\nseeded demo archetypes:")
    seeded = table[table["archetype"] != ""]
    print(seeded[["customer_id", "archetype", "risk", "band",
                  "outcome", "cause", "action", "steps"]].to_string(index=False))


if __name__ == "__main__":
    main()
