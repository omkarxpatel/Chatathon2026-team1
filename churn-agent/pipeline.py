"""Wires the layers together. Signals -> risk -> gate -> agent -> guardrails.

The whole flow is visible in run_cohort() below, in order, with the layer
boundary marked. If you read one file in this project, read this one.

    1. RAW EVENTS      orders, sessions, emails, support tickets
    2. FEATURES        29 signals per customer, computed strictly before
                       the cutoff date
    3. RISK MODEL      logistic regression -> a calibrated probability
                       and exact per-feature attributions. No LLM.
    4. POLICY GATE     deterministic rules. Customers who fail never
                       reach the agent, so no tokens are spent on people
                       we are not allowed to contact.
    5. AGENT           reads the unstructured signals through tools,
                       works out WHY, and commits to a decision. Never
                       computes the risk number.
    6. GUARDRAILS      check the agent's output. Unsure means send nothing.
    7. HUMAN           reviews every draft. There is no send path.

Deliberately not a class with state. The dashboard calls run_cohort()
once and caches the result; the CLI calls it and prints. Nothing here
holds a session.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

import config as cfg
from agent import guardrails
from agent.loop import build_brief, get_brain, run_agent
from agent.policy import GateOutcome, PolicyDecision, evaluate as evaluate_policy
from agent.schemas import Action, AgentDiagnosis, AgentRun
from data.store import EventStore
from features.extract import extract_frame
from features.trajectory import Trajectory, TrajectoryState
from features.trajectory import build as build_trajectories
from scoring.base import FitReport, RiskResult
from scoring.logistic import LogisticRiskScorer


@dataclass
class CustomerResult:
    """Everything we concluded about one customer, in pipeline order."""

    customer_id: str
    archetype: str | None
    features: dict[str, float]
    risk: RiskResult
    trajectory: Trajectory                     # direction of travel, not level
    policy: PolicyDecision
    agent_run: AgentRun | None                 # None when the gate stopped it
    guardrails: guardrails.GuardrailReport | None
    churn_label: bool                          # ground truth, for evaluation only

    @property
    def diagnosis(self) -> AgentDiagnosis | None:
        return self.agent_run.diagnosis if self.agent_run else None

    @property
    def final_action(self) -> Action:
        if self.guardrails is not None:
            return self.guardrails.final_action
        return Action.NO_ACTION

    @property
    def drifting(self) -> bool:
        """On the path past churners took, whatever today's score says.

        Deliberately NOT part of the policy gate. The gate decides who we
        are allowed to contact; this decides who is worth a second look.
        Letting a trend open the gate would contact people on a rising
        curve who still have product in the cupboard.
        """
        return self.trajectory.state is TrajectoryState.CLIMBING

    @property
    def early_warning(self) -> bool:
        """Climbing, but not yet scoring high enough to be worked.

        The customers this feature exists for. Nothing in the queue
        surfaces them today, and out of fold they churn at 62% against
        30% for the non-drifting peers they sit beside -- 2.05x, stable
        across five seeds. See SCORING.md.
        """
        return self.drifting and self.risk.band.value != "HIGH"

    @property
    def outcome_label(self) -> str:
        """One word for the cohort table."""
        if not self.policy.eligible:
            return {
                GateOutcome.SUPPRESSED_UNSUBSCRIBED: "suppressed (consent)",
                GateOutcome.SUPPRESSED_COOLDOWN: "suppressed (cooldown)",
                GateOutcome.SUPPRESSED_STILL_STOCKED: "suppressed (still stocked)",
                GateOutcome.NO_ACTION_LOW_RISK: "no action (low risk)",
            }[self.policy.outcome]
        if self.guardrails and self.guardrails.downgraded:
            return "blocked (guardrail)"
        if self.final_action == Action.NO_ACTION:
            return "no action (agent)"
        return "acted"


@dataclass
class CohortRun:
    results: list[CustomerResult]
    fit: FitReport
    as_of: date
    agent_name: str

    def by_id(self) -> dict[str, CustomerResult]:
        return {r.customer_id: r for r in self.results}

    # ------------------------------------------------------------------
    def investigated(self) -> list[CustomerResult]:
        return [r for r in self.results if r.agent_run is not None]

    def agent_stats(self) -> dict[str, int | float]:
        """What the agent actually did, for the run summary on screen."""
        runs = [r.agent_run for r in self.investigated()]
        calls = sum(run.tool_calls for run in runs)
        return {
            "customers_investigated": len(runs),
            "tool_calls": calls,
            "avg_steps": round(calls / len(runs), 1) if runs else 0.0,
            "drafted": sum(1 for r in self.results
                           if r.diagnosis and r.diagnosis.has_message),
            "left_alone": sum(1 for r in self.investigated()
                              if r.final_action == Action.NO_ACTION),
            "blocked": sum(1 for r in self.results
                           if r.guardrails and r.guardrails.downgraded),
        }

    def tool_usage(self) -> list[dict]:
        """How often the agent reached for each tool, most-used first."""
        counts: dict[str, int] = {}
        for r in self.investigated():
            for step in r.agent_run.steps:
                counts[step.tool] = counts.get(step.tool, 0) + 1
        return [{"tool": name, "calls": n}
                for name, n in sorted(counts.items(), key=lambda kv: -kv[1])]

    def funnel(self) -> list[dict]:
        """How the cohort narrows, stage by stage.

        This is the argument the project is making, as a list: most
        at-risk customers should NOT be contacted, and each stage says
        which principle removed them. The dashboard draws it; the logic
        lives here because it is derived from the run, not from layout.
        """
        order = ["consent", "cooldown", "risk_band", "supply"]
        labels = {
            "consent": "Have not opted out",
            "cooldown": "Not contacted recently",
            "risk_band": "At meaningful risk",
            "supply": "Actually out of product",
        }

        # Index of each customer's first failing check; len(order) if none.
        first_fail = []
        for r in self.results:
            idx = len(order)
            for i, c in enumerate(r.policy.checks):
                if not c.passed:
                    idx = i
                    break
            first_fail.append(idx)

        total = len(self.results)
        stages = [{"stage": "Scored by the risk model", "n": total, "dropped": 0,
                   "note": "whole cohort"}]
        for i, name in enumerate(order):
            remaining = sum(1 for f in first_fail if f > i)
            previous = stages[-1]["n"]
            stages.append({
                "stage": labels[name], "n": remaining,
                "dropped": previous - remaining,
                "note": f"{previous - remaining} removed by the {name} rule",
            })

        acted = sum(1 for r in self.results if r.final_action != Action.NO_ACTION)
        stages.append({
            "stage": "Agent recommended a response", "n": acted,
            "dropped": stages[-1]["n"] - acted,
            "note": f"{stages[-1]['n'] - acted} left alone by the agent",
        })
        return stages

    def table(self) -> pd.DataFrame:
        """The cohort view: one row per customer, ranked by risk."""
        rows = [
            {
                "customer_id": r.customer_id,
                "risk": round(r.risk.probability, 3),
                "band": r.risk.band.value,
                "gate": r.policy.outcome.value,
                "outcome": r.outcome_label,
                "cause": r.diagnosis.cause.value if r.diagnosis else "",
                "action": r.final_action.value,
                "steps": r.agent_run.tool_calls if r.agent_run else 0,
                "days_supply_left": round(r.features["days_of_supply_remaining"]),
                "gap_vs_median": round(r.features["reorder_gap_ratio"], 2),
                "trend": r.trajectory.state.value,
                "momentum": round(r.trajectory.momentum, 1),
                "archetype": r.archetype or "",
                "churned": r.churn_label,
            }
            for r in self.results
        ]
        return pd.DataFrame(rows).sort_values("risk", ascending=False).reset_index(drop=True)


def run_cohort(store: EventStore | None = None, as_of: date | None = None) -> CohortRun:
    store = store or EventStore.load()
    as_of = as_of or cfg.AS_OF_DATE

    # ======================================================================
    # LAYER 1 -- DETERMINISTIC. Pure Python and sklearn. No LLM anywhere
    # below this comment and above the next one.
    # ======================================================================
    X = extract_frame(store, as_of=as_of)
    y = store.label_series().loc[X.index]

    scorer = LogisticRiskScorer()
    fit = scorer.fit(X, y)
    risks = {r.customer_id: r for r in scorer.score_batch(X)}

    # Same model, rescored at a grid of earlier dates -> a risk CURVE per
    # customer. Answers "which direction", which the snapshot cannot.
    trajectories = build_trajectories(store, scorer, as_of)

    events = {cid: store.events_for(cid, as_of) for cid in X.index}
    features = {cid: X.loc[cid].to_dict() for cid in X.index}

    policies = {
        cid: evaluate_policy(risks[cid], features[cid], events[cid]) for cid in X.index
    }

    # ======================================================================
    # LAYER 2 -- AGENT. Only customers that cleared the gate get here, so
    # we never spend a token on someone we are not allowed to contact.
    # Each one gets its own investigation: the agent chooses which tools
    # to call and stops as soon as the evidence answers the question.
    # ======================================================================
    brain = get_brain()

    results: list[CustomerResult] = []
    for cid in X.index:
        run = report = None
        if policies[cid].eligible:
            run = run_agent(brain, build_brief(risks[cid], features[cid]), events[cid])
            report = guardrails.review(run.diagnosis, features[cid], events[cid])

        results.append(CustomerResult(
            customer_id=str(cid),
            archetype=store.archetype(str(cid)),
            features=features[cid],
            risk=risks[cid],
            trajectory=trajectories[str(cid)],
            policy=policies[cid],
            agent_run=run,
            guardrails=report,
            churn_label=store.label(str(cid)),
        ))

    results.sort(key=lambda r: -r.risk.probability)
    return CohortRun(results=results, fit=fit, as_of=as_of,
                     agent_name=getattr(brain, "name", "unknown"))
