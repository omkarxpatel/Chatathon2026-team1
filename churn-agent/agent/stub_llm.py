"""A deterministic stand-in for the LLM.

Build this first, wire everything through it, and the whole pipeline --
scoring, policy, diagnosis, guardrails, dashboard -- runs with no API key,
no network, and no cost. Swap in the real model with one flag in
config.py; nothing else changes, because both return AgentDiagnosis.

It is rules, not intelligence, and it is honest about that. What it is
faithful about is SHAPE: same schema, same reasoning-trace structure, same
refusal to act when acting would be tasteless. If the stub can drive the
demo end to end, the real model swapping in is a one-line change rather
than an integration project.

Rules fire in priority order. The order encodes the argument: check
whether there is even a problem before deciding how to fix it, and check
whether we caused the problem before asking the customer to spend money.
"""

from __future__ import annotations

import config as cfg
from agent.schemas import Action, AgentDiagnosis, Cause, DiagnosisRequest

# Phrases that mean "this person has stopped training for a reason that
# has nothing to do with us". Crude, and deliberately so -- this is
# exactly the judgement a real model does better, and the reason the
# agent layer exists at all.
def _quote(text: str, limit: int = 160) -> str:
    """Trim to a word boundary so traces do not end mid-word."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


LIFESTYLE_MARKERS = [
    "injur", "torn", "rotator", "surgery", "acl", "broke my", "fracture",
    "out of the gym", "not training", "stopped training", "can't train",
    "cannot train", "moving overseas", "relocating", "pregnan", "physio",
]

DISSATISFACTION_CATEGORIES = {"taste_mixability", "return_refund"}
SERVICE_CATEGORIES = {"damaged_shipment", "shipping_delay", "billing"}


class StubDiagnoser:
    """Same interface as the real diagnoser. See agent/diagnose.py."""

    name = "stub_llm"

    def diagnose(self, request: DiagnosisRequest) -> AgentDiagnosis:
        f = request.features
        tickets = request.tool_output.get("get_ticket_text", {})
        usage = request.tool_output.get("get_product_usage", {})
        orders = request.tool_output.get("get_order_history", {})
        engagement = request.tool_output.get("get_engagement_summary", {})

        ticket_list = tickets.get("tickets", [])
        unresolved = [t for t in ticket_list if not t.get("resolved")]
        negative = [t for t in ticket_list if t.get("sentiment", 0) < -0.2]
        recent_negative_unresolved = [
            t for t in unresolved
            if t.get("sentiment", 0) < -0.2 and t.get("days_ago", 999) <= 120
        ]

        trace: list[str] = [
            f"Risk model returned {request.probability:.0%} ({request.band}). "
            f"Taking that as given -- my job is to work out why, not to re-score it.",
        ]
        if request.top_attributions:
            top = request.top_attributions[0]
            trace.append(
                f"Largest single driver: {top.label} = {top.value:g} "
                f"({top.direction})."
            )

        # ---------------------------------------------------------------
        # 1. Is there even a problem? Product physics first.
        # ---------------------------------------------------------------
        remaining = usage.get("estimated_days_remaining")
        if remaining is not None and remaining > cfg.SUPPLY_BUFFER_DAYS:
            trace.append(
                f"{usage.get('explanation', '')} They are not overdue -- the gap "
                f"is pack size, not disengagement."
            )
            return self._build(
                request, trace, Cause.STILL_SUPPLIED, 0.9, Action.NO_ACTION,
                "They still have product on hand. A reminder now would be noise, "
                "and a discount now would train them to wait for one.",
            )

        # ---------------------------------------------------------------
        # 2. Did WE cause this? Service failure outranks everything else.
        # ---------------------------------------------------------------
        if recent_negative_unresolved:
            worst = min(recent_negative_unresolved, key=lambda t: t["sentiment"])
            trace.append(
                f"{len(recent_negative_unresolved)} unresolved ticket(s) with negative "
                f"sentiment, oldest {max(t['days_ago'] for t in recent_negative_unresolved)} "
                f"days ago. Category: {worst['category']}."
            )
            trace.append(
                'Quoting the customer: "' + _quote(worst["text"]) + '"' 
            )
            trace.append(
                "They did not drift away -- we broke something and then did not "
                "answer. Offering a discount here reads as buying silence."
            )
            action = (
                Action.HUMAN_ESCALATION if len(recent_negative_unresolved) >= 3
                else Action.SERVICE_RECOVERY
            )
            return self._build(
                request, trace, Cause.SERVICE_FAILURE, 0.88, action,
                "Fix the thing we got wrong, acknowledge the delay in replying, and "
                "make it right at no cost. No offer attached -- the problem is not price.",
            )

        # ---------------------------------------------------------------
        # 3. Have they told us they've stopped, for reasons of their own?
        # ---------------------------------------------------------------
        lifestyle_hits = [
            t for t in ticket_list
            if any(marker in t.get("text", "").lower() for marker in LIFESTYLE_MARKERS)
        ]
        if lifestyle_hits:
            hit = lifestyle_hits[-1]
            trace.append(
                f"Ticket from {hit['days_ago']} days ago reads as a life change, not "
                'a complaint: "' + _quote(hit["text"]) + '"' 
            )
            csat = [t["csat"] for t in ticket_list if t.get("csat") is not None]
            if csat:
                trace.append(
                    f"CSAT history {csat} and no unresolved tickets. Nothing is wrong "
                    f"with the relationship -- they have stopped training."
                )
            return self._build(
                request, trace, Cause.LIFESTYLE_CHANGE, 0.85, Action.NO_ACTION,
                "They are not lapsing, they are injured. Marketing protein to someone "
                "who just told support they are out of the gym is the failure mode "
                "this system exists to avoid. Leave them alone; they said they will "
                "be back.",
            )

        # ---------------------------------------------------------------
        # 4. Are they blocked on price rather than drifting?
        # ---------------------------------------------------------------
        promo_share = orders.get("promo_order_share", f.get("promo_order_share", 0.0)) or 0.0
        visits_since = f.get("sessions_since_last_order", 0.0)
        billing_tickets = [t for t in ticket_list if t.get("category") == "billing"]
        if promo_share >= 0.6 and visits_since >= 3:
            trace.append(
                f"{promo_share:.0%} of their orders used a promo code, and they have "
                f"visited {visits_since:.0f} times since last buying without "
                f"converting. That is intent with something blocking it."
            )
            if billing_tickets:
                trace.append(
                    'Billing ticket confirms it: "'
                    + _quote(billing_tickets[-1]["text"]) + '"' 
                )
            trace.append(
                "A percentage discount would work once and then teach them to wait "
                "for the next one. Better to change the unit economics."
            )
            return self._build(
                request, trace, Cause.PRICE_SENSITIVITY, 0.82, Action.VALUE_EDUCATION,
                "Show cost per serving on the larger pack and point at subscribe-and-save. "
                "That lowers their real price without running another promo.",
            )

        # ---------------------------------------------------------------
        # 5. Did the product itself disappoint them?
        # ---------------------------------------------------------------
        product_complaints = [
            t for t in negative if t.get("category") in DISSATISFACTION_CATEGORIES
        ]
        if product_complaints:
            hit = product_complaints[-1]
            trace.append(
                f"Negative ticket about {hit['category']}: "
                + '"' + _quote(hit["text"]) + '"' 
            )
            return self._build(
                request, trace, Cause.PRODUCT_DISSATISFACTION, 0.7, Action.PRODUCT_GUIDANCE,
                "The complaint is about the product, not the price. Offer a fix or a "
                "different format; do not discount the thing they did not enjoy.",
            )

        # ---------------------------------------------------------------
        # 6. Nothing specific. They have just run out and not come back.
        # ---------------------------------------------------------------
        if usage.get("ran_out"):
            trace.append(
                f"{usage.get('explanation', '')} No complaint on file and no sign of a "
                f"life change -- this looks like an ordinary lapse."
            )
        else:
            trace.append("No complaint, no life change, no price signal on file.")
        clicks = engagement.get("click_rate")
        if clicks is not None:
            trace.append(
                f"Email click rate {clicks:.0%} over {engagement.get('emails_delivered', 0)} "
                f"delivered. Opens are ignored -- Apple MPP inflates them."
            )
        return self._build(
            request, trace, Cause.ROUTINE_LAPSE, 0.55, Action.REPLENISHMENT_REMINDER,
            "One low-key reminder that they are probably out, with an easy way to say "
            "'stop sending these'. No offer -- there is no evidence price is the issue.",
        )

    # ------------------------------------------------------------------
    def _build(
        self, request: DiagnosisRequest, trace: list[str], cause: Cause,
        confidence: float, action: Action, rationale: str,
    ) -> AgentDiagnosis:
        from agent.copywriter import draft

        subject, body = draft(action, request)
        trace.append(f"Selected action: {action.value}.")
        return AgentDiagnosis(
            customer_id=request.customer_id,
            cause=cause, cause_confidence=confidence, reasoning_trace=trace,
            action=action, action_rationale=rationale,
            message_subject=subject, message_body=body,
            requires_human_approval=True, diagnoser=self.name,
        )
