"""Presentation and draft-review helpers. Scoring and policy stay in the pipeline."""

from agent.guardrails import review
from agent.policy import GateOutcome
from agent.schemas import Action, Cause
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
