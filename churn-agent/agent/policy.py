"""The deterministic gate. Runs BEFORE the agent, every time.

Nothing here is a judgement call and nothing here calls a model. These are
the rules the brand has already decided on, and the agent does not get a
vote on them. A customer who fails this gate never reaches the LLM -- we
do not spend tokens reasoning about someone we are not allowed to contact.

Check order is deliberate, cheapest and most absolute first:

  1. consent      they unsubscribed or filed a complaint. Never contact.
  2. cooldown     we sent them a targeted message N days ago. Not again.
  3. risk band    LOW band. There is no problem to solve.
  4. supply       they still have product on hand. Nothing is wrong yet.

Only a customer who clears all four is handed to the agent.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from pydantic import BaseModel
from enum import Enum

import config as cfg
from data.store import CustomerEvents
from data.text_bank import TARGETED_CAMPAIGN_TYPES
from scoring.base import RiskResult, band_at_least


class GateOutcome(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    SUPPRESSED_UNSUBSCRIBED = "SUPPRESSED_UNSUBSCRIBED"
    SUPPRESSED_COOLDOWN = "SUPPRESSED_COOLDOWN"
    SUPPRESSED_STILL_STOCKED = "SUPPRESSED_STILL_STOCKED"
    NO_ACTION_LOW_RISK = "NO_ACTION_LOW_RISK"


class PolicyCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class PolicyDecision(BaseModel):
    customer_id: str
    outcome: GateOutcome
    reason: str
    checks: list[PolicyCheck]

    @property
    def eligible(self) -> bool:
        return self.outcome == GateOutcome.ELIGIBLE


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------


def check_consent(ev: CustomerEvents) -> PolicyCheck:
    """Unsubscribes and spam complaints are permanent. No score overrides them."""
    if len(ev.messages) == 0:
        return PolicyCheck(name="consent", passed=True, detail="No unsubscribe or complaint on record.")
    flagged = ev.messages[ev.messages["event"].isin(["unsubscribe", "complaint"])]
    if len(flagged):
        kind = str(flagged.iloc[-1]["event"])
        when = flagged["ts"].max().date()
        return PolicyCheck(
            name="consent", passed=False,
            detail=f"Customer recorded a '{kind}' on {when}. Permanently suppressed.",
        )
    return PolicyCheck(name="consent", passed=True, detail="No unsubscribe or complaint on record.")


def last_targeted_send(ev: CustomerEvents):
    """Most recent one-to-one flow message. Broadcast newsletters do not count.

    Counting broadcasts would suppress almost the whole list -- the brand
    emails everyone every few days -- and the cooldown would be theatre.
    What we are protecting against is stacking targeted outreach on top of
    targeted outreach.
    """
    if len(ev.messages) == 0:
        return None
    sends = ev.messages[
        (ev.messages["event"] == "sent")
        & (ev.messages["campaign_type"].isin(TARGETED_CAMPAIGN_TYPES))
    ]
    return None if len(sends) == 0 else sends["ts"].max().date()


def check_cooldown(ev: CustomerEvents) -> PolicyCheck:
    last = last_targeted_send(ev)
    if last is None:
        return PolicyCheck(name="cooldown", passed=True,
                           detail="No targeted outreach on record.")
    days = (ev.as_of - last).days
    if days < cfg.COOLDOWN_DAYS:
        return PolicyCheck(
            name="cooldown", passed=False,
            detail=(f"Targeted message sent {days} days ago ({last}); "
                    f"cooldown is {cfg.COOLDOWN_DAYS} days."),
        )
    return PolicyCheck(name="cooldown", passed=True,
                       detail=f"Last targeted message was {days} days ago ({last}).")


def check_risk_band(risk: RiskResult) -> PolicyCheck:
    if not band_at_least(risk.band, cfg.MIN_BAND_TO_ACT):
        return PolicyCheck(
            name="risk_band", passed=False,
            detail=(f"Risk {risk.probability:.0%} is {risk.band.value}; "
                    f"minimum to act is {cfg.MIN_BAND_TO_ACT}."),
        )
    return PolicyCheck(name="risk_band", passed=True,
                       detail=f"Risk {risk.probability:.0%} is {risk.band.value}.")


def check_supply(features: dict[str, float]) -> PolicyCheck:
    """Product physics as a hard rule.

    If the tub they bought has not run out yet, there is nothing to fix.
    A long gap on a 76-day pack is not a lapse, it is arithmetic. This is
    the check that stops the demo's CUST-0002 from being contacted, and it
    is deliberately NOT left to the model to infer -- servings divided by
    servings-per-day is a fact, not an estimate.
    """
    remaining = features.get("days_of_supply_remaining", 0.0)
    if remaining is None or pd.isna(remaining):
        remaining = 0.0
    if remaining > cfg.SUPPLY_BUFFER_DAYS:
        return PolicyCheck(
            name="supply", passed=False,
            detail=(f"Still has ~{remaining:.0f} days of product on hand "
                    f"(buffer is {cfg.SUPPLY_BUFFER_DAYS} days). Not due yet."),
        )
    if remaining >= 0:
        return PolicyCheck(name="supply", passed=True,
                           detail=f"About {remaining:.0f} days of product left; due shortly.")
    return PolicyCheck(name="supply", passed=True,
                       detail=f"Ran out roughly {abs(remaining):.0f} days ago.")


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

_FAILURE_OUTCOME = {
    "consent": GateOutcome.SUPPRESSED_UNSUBSCRIBED,
    "cooldown": GateOutcome.SUPPRESSED_COOLDOWN,
    "risk_band": GateOutcome.NO_ACTION_LOW_RISK,
    "supply": GateOutcome.SUPPRESSED_STILL_STOCKED,
}


def evaluate(risk: RiskResult, features: dict[str, float], ev: CustomerEvents) -> PolicyDecision:
    checks = [
        check_consent(ev),
        check_cooldown(ev),
        check_risk_band(risk),
        check_supply(features),
    ]
    # Every check runs so the dashboard can show the whole trace, but the
    # first failure is what decides the outcome.
    for check in checks:
        if not check.passed:
            return PolicyDecision(
                customer_id=risk.customer_id,
                outcome=_FAILURE_OUTCOME[check.name],
                reason=check.detail,
                checks=checks,
            )
    return PolicyDecision(
        customer_id=risk.customer_id,
        outcome=GateOutcome.ELIGIBLE,
        reason="Cleared all policy checks; handed to the agent.",
        checks=checks,
    )
