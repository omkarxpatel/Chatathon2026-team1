"""Wires the layers together. Signals -> risk -> gate -> agent -> guardrails.

The whole architecture is visible in run_cohort() below, in order, with
the layer boundary marked. If you read one file in this project, read
this one.

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
from agent.diagnose import build_request, get_diagnoser
from agent.policy import GateOutcome, PolicyDecision, evaluate as evaluate_policy
from agent.schemas import Action, AgentDiagnosis
from data.store import EventStore
from features.extract import FEATURE_NAMES, extract_frame
from scoring.base import FitReport, RiskResult
from scoring.logistic import LogisticRiskScorer


@dataclass
class CustomerResult:
    """Everything we concluded about one customer, in pipeline order."""

    customer_id: str
    archetype: str | None
    features: dict[str, float]
    risk: RiskResult
    policy: PolicyDecision
    diagnosis: AgentDiagnosis | None          # None when the gate stopped it
    guardrails: guardrails.GuardrailReport | None
    churn_label: bool                          # ground truth, for evaluation only

    @property
    def final_action(self) -> Action:
        if self.guardrails is not None:
            return self.guardrails.final_action
        return Action.NO_ACTION

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
    diagnoser_name: str

    def by_id(self) -> dict[str, CustomerResult]:
        return {r.customer_id: r for r in self.results}

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
        stages = [{"stage": "Scored", "n": total, "dropped": 0, "note": "whole cohort"}]
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
            "stage": "Agent chose to act", "n": acted,
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
                "days_supply_left": round(r.features["days_of_supply_remaining"]),
                "gap_vs_median": round(r.features["reorder_gap_ratio"], 2),
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

    events = {cid: store.events_for(cid, as_of) for cid in X.index}
    features = {cid: X.loc[cid].to_dict() for cid in X.index}

    policies = {
        cid: evaluate_policy(risks[cid], features[cid], events[cid]) for cid in X.index
    }

    # ======================================================================
    # LAYER 2 -- AGENT. Only customers that cleared the gate get here, so
    # we never spend a token on someone we are not allowed to contact.
    # ======================================================================
    diagnoser = get_diagnoser()

    results: list[CustomerResult] = []
    for cid in X.index:
        diagnosis = report = None
        if policies[cid].eligible:
            request = build_request(risks[cid], features[cid], events[cid])
            diagnosis = diagnoser.diagnose(request)
            report = guardrails.review(diagnosis, features[cid], events[cid])

        results.append(CustomerResult(
            customer_id=str(cid),
            archetype=store.archetype(str(cid)),
            features=features[cid],
            risk=risks[cid],
            policy=policies[cid],
            diagnosis=diagnosis,
            guardrails=report,
            churn_label=store.label(str(cid)),
        ))

    results.sort(key=lambda r: -r.risk.probability)
    return CohortRun(results=results, fit=fit, as_of=as_of,
                     diagnoser_name=getattr(diagnoser, "name", "unknown"))
