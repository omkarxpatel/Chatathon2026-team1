"""Post-hoc checks on whatever the agent proposed.

The policy gate decides whether the agent runs. Guardrails decide whether
its output survives. They exist because the agent layer is the only
non-deterministic part of the system, and the point of a two-layer design
is that the non-deterministic part is boxed in on both sides.

A failed check does not just log a warning -- it DOWNGRADES the action,
usually to NO_ACTION. The safe default when we are unsure is to send
nothing.

These are deliberately written to catch a real LLM, not just the stub.
Swap config.USE_REAL_LLM to True and these are the checks standing
between a plausible-sounding hallucination and a customer.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

import config as cfg
from agent.schemas import Action, AgentDiagnosis, Cause, SILENT_ACTIONS
from agent.tools import CustomerEvents

# A number followed by % off, or "% discount", etc.
_PERCENT_OFF = re.compile(r"\b\d{1,2}\s*%\s*(off|discount|savings)?", re.IGNORECASE)


class GuardrailCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class GuardrailReport(BaseModel):
    customer_id: str
    checks: list[GuardrailCheck]
    original_action: Action
    final_action: Action
    downgraded: bool
    verdict: str

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[GuardrailCheck]:
        return [c for c in self.checks if not c.passed]


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check_human_approval(d: AgentDiagnosis) -> GuardrailCheck:
    """There is no send path in this project. This must never be False."""
    return GuardrailCheck(
        name="human_approval_required",
        passed=d.requires_human_approval,
        detail=("Flagged for human review."
                if d.requires_human_approval
                else "Agent tried to mark this as auto-sendable."),
    )


def check_message_presence(d: AgentDiagnosis) -> GuardrailCheck:
    """NO_ACTION means no message. Anything else means there must be one."""
    if d.action in SILENT_ACTIONS:
        ok = not d.has_message
        return GuardrailCheck(
            name="message_presence", passed=ok,
            detail=("No message, as expected for a no-action decision."
                    if ok else
                    "Action is NO_ACTION but the agent still drafted a message."),
        )
    ok = d.has_message
    return GuardrailCheck(
        name="message_presence", passed=ok,
        detail=("Message drafted." if ok else
                f"Action is {d.action.value} but no message was drafted."),
    )


def check_discount_justified(d: AgentDiagnosis) -> GuardrailCheck:
    """A discount is only ever allowed when the diagnosis is price.

    This is the single most important guardrail in the file. Discounting a
    customer who left because we broke their order does not retain them,
    it teaches them that complaining is profitable -- and discounting
    someone who is simply injured is just waste.
    """
    if d.action != Action.DISCOUNT_OFFER:
        return GuardrailCheck(
            name="discount_justified", passed=True,
            detail="No discount proposed.",
        )
    ok = d.cause == Cause.PRICE_SENSITIVITY
    return GuardrailCheck(
        name="discount_justified", passed=ok,
        detail=("Discount proposed and the diagnosed cause is price sensitivity."
                if ok else
                f"Discount proposed but the diagnosed cause is "
                f"'{d.cause.value}'. Money is not the fix for that."),
    )


def check_no_stray_discount(d: AgentDiagnosis) -> GuardrailCheck:
    """Catches a percentage-off buried in the copy of a non-discount action.

    A real model will occasionally decide a service-recovery email would
    land better with "and here's 20% off". It would not.
    """
    if not d.has_message or d.action == Action.DISCOUNT_OFFER:
        return GuardrailCheck(name="no_stray_discount", passed=True,
                              detail="Not applicable.")
    text = f"{d.message_subject or ''} {d.message_body or ''}"
    # "17% less per serving" is cost-per-serving maths, not an offer.
    candidates = [
        m.group(0) for m in _PERCENT_OFF.finditer(text)
        if m.group(1) is not None
    ]
    if candidates:
        return GuardrailCheck(
            name="no_stray_discount", passed=False,
            detail=f"Action is {d.action.value} but the copy offers {candidates[0]!r}.",
        )
    return GuardrailCheck(name="no_stray_discount", passed=True,
                          detail="No offer language in the copy.")


def check_tone(d: AgentDiagnosis) -> GuardrailCheck:
    """The brief says helpful and non-pushy. This is where that is enforced."""
    if not d.has_message:
        return GuardrailCheck(name="tone", passed=True, detail="No message to check.")
    text = f"{d.message_subject or ''} {d.message_body or ''}".lower()
    hits = [p for p in cfg.BANNED_PUSHY_PHRASES if p in text]
    if hits:
        return GuardrailCheck(name="tone", passed=False,
                              detail=f"Pushy phrasing found: {hits}.")
    return GuardrailCheck(name="tone", passed=True,
                          detail="No urgency or pressure language.")


def check_length(d: AgentDiagnosis) -> GuardrailCheck:
    if not d.has_message:
        return GuardrailCheck(name="length", passed=True, detail="No message to check.")
    words = len((d.message_body or "").split())
    ok = words <= cfg.MAX_MESSAGE_WORDS
    return GuardrailCheck(
        name="length", passed=ok,
        detail=f"{words} words (limit {cfg.MAX_MESSAGE_WORDS}).",
    )


def check_supply_consistency(d: AgentDiagnosis, features: dict[str, float]) -> GuardrailCheck:
    """Never nudge someone who demonstrably still has product.

    Duplicates the policy gate on purpose. The gate stops them reaching
    the agent; this stops a diagnosis that ignored the arithmetic.
    """
    remaining = features.get("days_of_supply_remaining", 0.0)
    if d.action in SILENT_ACTIONS or remaining is None:
        return GuardrailCheck(name="supply_consistency", passed=True,
                              detail="Not applicable.")
    if remaining > cfg.SUPPLY_BUFFER_DAYS:
        return GuardrailCheck(
            name="supply_consistency", passed=False,
            detail=(f"Proposed {d.action.value} but they still have "
                    f"~{remaining:.0f} days of product."),
        )
    return GuardrailCheck(name="supply_consistency", passed=True,
                          detail=f"~{remaining:.0f} days of supply left; timing is fair.")


def check_lifestyle_respect(d: AgentDiagnosis) -> GuardrailCheck:
    """If the agent itself concluded the customer stopped for personal
    reasons, it does not then get to sell them something."""
    if d.cause != Cause.LIFESTYLE_CHANGE:
        return GuardrailCheck(name="lifestyle_respect", passed=True,
                              detail="Not applicable.")
    ok = d.action in SILENT_ACTIONS
    return GuardrailCheck(
        name="lifestyle_respect", passed=ok,
        detail=("Diagnosed a life change and correctly proposed no outreach."
                if ok else
                f"Diagnosed a life change but still proposed {d.action.value}. "
                f"They stopped training; they did not stop liking us."),
    )


def check_simulation_label(d: AgentDiagnosis) -> GuardrailCheck:
    """Belt and braces. The UI labels every message too."""
    return GuardrailCheck(
        name="simulation_label", passed=True,
        detail=f"Rendered with the {cfg.SIMULATION_BANNER} banner. No send path exists.",
    )


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

# A failure in any of these means we should not send at all, rather than
# send something slightly different.
_DOWNGRADE_TO_NO_ACTION = {
    "discount_justified", "no_stray_discount", "tone",
    "supply_consistency", "lifestyle_respect", "human_approval_required",
}


def review(
    diagnosis: AgentDiagnosis,
    features: dict[str, float],
    ev: CustomerEvents | None = None,
) -> GuardrailReport:
    checks = [
        check_human_approval(diagnosis),
        check_message_presence(diagnosis),
        check_discount_justified(diagnosis),
        check_no_stray_discount(diagnosis),
        check_tone(diagnosis),
        check_length(diagnosis),
        check_supply_consistency(diagnosis, features),
        check_lifestyle_respect(diagnosis),
        check_simulation_label(diagnosis),
    ]

    blocking = [c for c in checks if not c.passed and c.name in _DOWNGRADE_TO_NO_ACTION]
    soft = [c for c in checks if not c.passed and c.name not in _DOWNGRADE_TO_NO_ACTION]

    if blocking:
        final, downgraded = Action.NO_ACTION, diagnosis.action != Action.NO_ACTION
        verdict = "BLOCKED: " + " ".join(c.detail for c in blocking)
    elif soft:
        final, downgraded = diagnosis.action, False
        verdict = "PASSED WITH WARNINGS: " + " ".join(c.detail for c in soft)
    else:
        final, downgraded = diagnosis.action, False
        verdict = "PASSED: all checks clear. Awaiting human approval."

    return GuardrailReport(
        customer_id=diagnosis.customer_id,
        checks=checks,
        original_action=diagnosis.action,
        final_action=final,
        downgraded=downgraded,
        verdict=verdict,
    )
