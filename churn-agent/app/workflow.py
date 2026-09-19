"""Plain-language labels and draft-review helpers for the workspace.

Everything here is presentation. No scoring, no policy, no agent logic --
those live in the pipeline, and this file only decides what to call them
in front of a person who does retention for a living and does not care
what a logistic regression is.
"""

from agent.guardrails import review
from agent.policy import GateOutcome
from agent.schemas import Action, AgentStep, Cause, StepKind
from agent.tools import TOOL_LABELS
from pipeline import CustomerResult

ACTION_LABELS = {
    Action.NO_ACTION: "Give them space",
    Action.REPLENISHMENT_REMINDER: "Send a helpful reminder",
    Action.PRODUCT_GUIDANCE: "Help with the product",
    Action.VALUE_EDUCATION: "Explain ways to save",
    Action.SERVICE_RECOVERY: "Resolve the service issue",
    Action.LOYALTY_RECOGNITION: "Recognize their loyalty",
    Action.HUMAN_ESCALATION: "Ask a teammate to help",
    Action.DISCOUNT_OFFER: "Consider a price offer",
}

CAUSE_LABELS = {
    Cause.SERVICE_FAILURE: "An unresolved service experience",
    Cause.PRICE_SENSITIVITY: "Price may be holding them back",
    Cause.LIFESTYLE_CHANGE: "Their circumstances have changed",
    Cause.STILL_SUPPLIED: "They still have enough product",
    Cause.PRODUCT_DISSATISFACTION: "The product did not meet expectations",
    Cause.ROUTINE_LAPSE: "They have fallen out of their routine",
    Cause.UNCLEAR: "The reason needs a closer look",
}

HOLD_LABELS = {
    GateOutcome.SUPPRESSED_UNSUBSCRIBED: ("Do not contact", "They opted out of outreach"),
    GateOutcome.SUPPRESSED_COOLDOWN: ("Wait before following up", "They received a recent message"),
    GateOutcome.SUPPRESSED_STILL_STOCKED: ("Wait until they need more", "They still have enough product"),
    GateOutcome.NO_ACTION_LOW_RISK: ("No outreach needed", "Their risk is low"),
}

# Why the agent never got to look at this customer. Shown where the
# investigation timeline would otherwise be.
HOLD_REASONS = {
    GateOutcome.SUPPRESSED_UNSUBSCRIBED:
        "They asked us to stop, so the agent was not asked to look. Consent is never overridden by a risk score.",
    GateOutcome.SUPPRESSED_COOLDOWN:
        "We contacted them recently, so the agent was not asked to look. Stacking outreach on outreach is how people unsubscribe.",
    GateOutcome.SUPPRESSED_STILL_STOCKED:
        "Their last order has not run out yet, so there is nothing to fix and the agent was not asked to look.",
    GateOutcome.NO_ACTION_LOW_RISK:
        "Their risk is low, so the agent was not asked to look. Investigating everyone would cost more than it saves.",
}

# What would have to change for this customer to come back into the queue.
# Answering that is more use to a reviewer than repeating the rule at them.
WHAT_CHANGES = {
    GateOutcome.SUPPRESSED_UNSUBSCRIBED:
        "Nothing changes this. An opt-out is permanent, whatever their risk score says.",
    GateOutcome.SUPPRESSED_COOLDOWN:
        "They come back into the queue once the cooldown has passed, if they are still at risk.",
    GateOutcome.SUPPRESSED_STILL_STOCKED:
        "They come back into the queue as their supply runs low, which is when a reminder is actually useful.",
    GateOutcome.NO_ACTION_LOW_RISK:
        "They come back into the queue if their buying or browsing changes.",
}

CHECK_LABELS = {
    "consent": "Permission to contact", "cooldown": "Time since the last message",
    "risk_band": "Risk level", "supply": "Estimated product remaining",
    "human_approval_required": "Human review required", "message_presence": "Draft is present",
    "discount_justified": "Offer matches the reason", "no_stray_discount": "No unexpected discounts",
    "tone": "Helpful, respectful language", "length": "Message length",
    "supply_consistency": "Appropriate timing", "lifestyle_respect": "Personal circumstances respected",
    "simulation_label": "Demo clearly labeled",
}


def can_review(result: CustomerResult) -> bool:
    return bool(result.policy.eligible and result.diagnosis and result.diagnosis.has_message
                and result.final_action != Action.NO_ACTION)


def next_step(result: CustomerResult) -> str:
    if not result.policy.eligible:
        return HOLD_LABELS[result.policy.outcome][0]
    if result.guardrails and result.guardrails.downgraded:
        return "Hold for a closer review"
    return ACTION_LABELS[result.final_action]


def reason(result: CustomerResult) -> str:
    if not result.policy.eligible:
        return HOLD_LABELS[result.policy.outcome][1]
    if result.guardrails and result.guardrails.downgraded:
        return "The suggested response did not pass review"
    return CAUSE_LABELS[result.diagnosis.cause] if result.diagnosis else "No outreach recommended"


def explanation(result: CustomerResult) -> str:
    if not result.policy.eligible:
        return result.policy.reason
    if result.guardrails and result.guardrails.downgraded:
        return " ".join(c.detail for c in result.guardrails.failures)
    return result.diagnosis.action_rationale if result.diagnosis else "No outreach recommended."


# --------------------------------------------------------------------------
# the agent's run, described for a person
# --------------------------------------------------------------------------


def step_label(step: AgentStep) -> str:
    return TOOL_LABELS.get(step.tool, step.tool.replace("_", " ").capitalize())


def run_summary(result: CustomerResult) -> str:
    """One line on what the agent did, for the top of the timeline."""
    run = result.agent_run
    if run is None:
        return "The agent was not asked to look at this customer."
    looked = len(run.investigation)
    if run.stop_reason != "decided":
        return f"The agent checked {looked} things and stopped without a clear answer."
    verb = {Action.NO_ACTION: "recommended leaving them alone",
            Action.HUMAN_ESCALATION: "handed them to a teammate"}.get(
                run.diagnosis.action, "drafted a response")
    return (f"The agent checked {looked} thing{'s' if looked != 1 else ''} about this "
            f"customer, then {verb}.")


def confidence_label(result: CustomerResult) -> str:
    d = result.diagnosis
    if d is None:
        return ""
    value = d.cause_confidence
    word = "fairly confident" if value >= 0.8 else "reasonably sure" if value >= 0.6 else "not certain"
    return f"{word} ({value:.0%})"


def hold_reason(result: CustomerResult) -> str:
    return HOLD_REASONS.get(result.policy.outcome, result.policy.reason)


def what_changes(result: CustomerResult) -> str:
    return WHAT_CHANGES.get(result.policy.outcome, "")


# --------------------------------------------------------------------------
# draft review
# --------------------------------------------------------------------------


def validate_draft(result: CustomerResult, subject: str, body: str) -> list[str]:
    """Recheck edited copy before it can be marked reviewed or exported."""
    if not can_review(result):
        return ["This customer is not eligible for a response."]
    errors = []
    if not subject.strip():
        errors.append("Add a subject before marking this draft reviewed.")
    if not body.strip():
        errors.append("Add a message before marking this draft reviewed.")
    edited = result.diagnosis.model_copy(update={
        "message_subject": subject.strip(), "message_body": body.strip(),
    })
    errors.extend(c.detail for c in review(edited, result.features).failures)
    return list(dict.fromkeys(errors))
