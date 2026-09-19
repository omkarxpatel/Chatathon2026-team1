"""A deterministic brain, so the whole thing runs with no API key.

It drives the real loop. It is asked "what next?" one turn at a time, it
answers with one tool call, and it only sees what it has already fetched
-- same protocol as the Claude brain in agent/claude_agent.py, same
tools, same transcript on screen. Swap config.USE_REAL_LLM and the
dashboard cannot tell the difference, because there is nothing structural
to tell apart.

It is rules, not intelligence, and it is honest about that. What it is
faithful about is SHAPE: it investigates before it concludes, it stops as
soon as the evidence answers the question, and it refuses to act when
acting would be tasteless.

The order of the cascade is the argument the product is making:

  1. is there even a problem?      supply arithmetic first
  2. did WE cause it?              unresolved complaints outrank everything
  3. have they told us they quit?  a life change is not a lapse
  4. is price the blocker?         then teach value, do not discount
  5. did the product disappoint?   fix the product, do not discount it
  6. none of the above             one quiet reminder, no offer
"""

from __future__ import annotations

from typing import Any

import config as cfg
from agent.loop import Move
from agent.schemas import AgentStep, Cause, CustomerBrief

# Phrases that mean "this person has stopped training for a reason that
# has nothing to do with us". Crude, and deliberately so -- this is
# exactly the judgement a real model does better, and the reason the
# agent layer exists at all.
LIFESTYLE_MARKERS = [
    "injur", "torn", "rotator", "surgery", "acl", "broke my", "fracture",
    "out of the gym", "not training", "stopped training", "can't train",
    "cannot train", "moving overseas", "relocating", "pregnan", "physio",
]

DISSATISFACTION_CATEGORIES = {"taste_mixability", "return_refund"}


def _quote(text: str, limit: int = 140) -> str:
    """Trim to a word boundary so a quoted ticket does not end mid-word."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


class RuleBrain:
    """Same interface as the Claude brain. See agent/loop.py for the protocol."""

    name = "rules"

    # The cascade is a pure function of what has been fetched so far, so
    # there is no state to reset and nothing to remember between turns.
    def reset(self) -> None:
        pass

    def observe(self, step: AgentStep) -> None:
        pass

    def observe_error(self, message: str) -> None:
        pass

    # ------------------------------------------------------------------
    def next_move(self, brief: CustomerBrief, steps: list[AgentStep]) -> Move:
        seen: dict[str, Any] = {s.tool: s.observation for s in steps}
        f = brief.features

        # 1. Is there even a problem? Product physics first.
        if "check_product_supply" not in seen:
            return Move(
                "Before reading anything into the gap, work out whether they have "
                "actually run out.",
                "check_product_supply")

        supply = seen["check_product_supply"]
        remaining = supply.get("estimated_days_remaining")
        if remaining is not None and remaining > cfg.SUPPLY_BUFFER_DAYS:
            return Move(
                f"{supply.get('explanation', '')} The gap is pack size, not "
                f"disengagement, so there is nothing to fix.",
                "recommend_no_contact", {
                    "cause": Cause.STILL_SUPPLIED.value, "confidence": 0.9,
                    "rationale": ("They still have product on hand. A reminder now "
                                  "would be noise, and a discount now would train "
                                  "them to wait for one."),
                })

        # 2. Did WE cause this? Service failure outranks everything else.
        if "read_support_tickets" not in seen:
            return Move(
                "They are out of product. Read their support history to see whether "
                "we caused this, or whether they told us something about themselves.",
                "read_support_tickets")

        tickets = seen["read_support_tickets"].get("tickets", [])
        unresolved = [t for t in tickets if not t.get("resolved")]
        negative = [t for t in tickets if t.get("sentiment", 0) < -0.2]
        open_complaints = [t for t in unresolved
                           if t.get("sentiment", 0) < -0.2 and t.get("days_ago", 999) <= 120]

        if open_complaints:
            worst = min(open_complaints, key=lambda t: t["sentiment"])
            oldest = max(t["days_ago"] for t in open_complaints)
            if "read_order_history" not in seen:
                return Move(
                    f'An unanswered complaint about a '
                    f'{str(worst["category"]).replace("_", " ")}, {oldest} days old: '
                    f'"{_quote(worst["text"])}" — pull the orders so the reply names '
                    f'the one that went wrong.',
                    "read_order_history")

            # Duplicating the policy gate on purpose. The gate decided we
            # may contact them; this confirms it before words get written.
            if "check_contact_policy" not in seen:
                return Move(
                    "Before writing anything, confirm we are still allowed to contact "
                    "them and have not just emailed them.",
                    "check_contact_policy")

            evidence = self._evidence(seen)
            if len(open_complaints) >= 3:
                return Move(
                    f"{len(open_complaints)} unresolved complaints, all negative. This "
                    f"is past what an automated message should touch.",
                    "escalate_to_human", {
                        "cause": Cause.SERVICE_FAILURE.value, "confidence": 0.9,
                        "rationale": ("Several open complaints and no reply from us. A "
                                      "template here would make it worse."),
                        "internal_note": self._draft(
                            "human_escalation", brief, evidence)[1],
                    })
            subject, body = self._draft("service_recovery", brief, evidence)
            return Move(
                "They did not drift away — we broke something and then did not answer. "
                "Fix that, and attach no offer: a discount on top of an unanswered "
                "complaint reads as buying silence.",
                "propose_outreach", {
                    "cause": Cause.SERVICE_FAILURE.value, "confidence": 0.88,
                    "action": "service_recovery",
                    "rationale": ("Fix the thing we got wrong, acknowledge the delay in "
                                  "replying, and make it right at no cost. The problem "
                                  "is not price."),
                    "subject": subject, "body": body,
                })

        # 3. Have they told us they have stopped, for reasons of their own?
        lifestyle = [t for t in tickets
                     if any(m in str(t.get("text", "")).lower() for m in LIFESTYLE_MARKERS)]
        if lifestyle:
            hit = lifestyle[0]
            csat = [t["csat"] for t in tickets if t.get("csat") is not None]
            satisfied = f" CSAT history {csat}, nothing unresolved." if csat else ""
            return Move(
                f'Their ticket from {hit["days_ago"]} days ago reads as a life change, '
                f'not a complaint: "{_quote(hit["text"])}".{satisfied}',
                "recommend_no_contact", {
                    "cause": Cause.LIFESTYLE_CHANGE.value, "confidence": 0.85,
                    "rationale": ("They are not lapsing, they are injured. Marketing "
                                  "protein to someone who just told support they are "
                                  "out of the gym is the failure mode this system "
                                  "exists to avoid. They said they will be back."),
                })

        # 4. Are they blocked on price rather than drifting?
        if "read_order_history" not in seen:
            return Move(
                "No open complaint and no life change on file. Check what they have "
                "been buying and whether they only buy on promotion.",
                "read_order_history")

        orders = seen["read_order_history"]
        promo_share = orders.get("promo_order_share") or 0.0
        if "read_engagement" not in seen:
            return Move(
                f"{promo_share:.0%} of their orders used a promo code. Check whether "
                f"they are still browsing without buying.",
                "read_engagement")

        if "check_contact_policy" not in seen:
            return Move(
                "Whatever I recommend here involves writing to them, so confirm first "
                "that we are allowed to and have not just emailed them.",
                "check_contact_policy")

        engagement = seen["read_engagement"]
        visits_since = f.get("sessions_since_last_order", 0.0)
        evidence = self._evidence(seen)

        if promo_share >= 0.6 and visits_since >= 3:
            billing = [t for t in tickets if t.get("category") == "billing"]
            confirm = (f' A billing ticket says as much: "{_quote(billing[-1]["text"])}".'
                       if billing else "")
            subject, body = self._draft("value_education", brief, evidence)
            return Move(
                f"They have visited {visits_since:.0f} times since last buying without "
                f"converting.{confirm} That is intent with price in the way — and a "
                f"percentage off would work once, then teach them to wait for the next one.",
                "propose_outreach", {
                    "cause": Cause.PRICE_SENSITIVITY.value, "confidence": 0.82,
                    "action": "value_education",
                    "rationale": ("Show cost per serving on the larger pack and point at "
                                  "subscribe-and-save. That lowers their real price "
                                  "without running another promo."),
                    "subject": subject, "body": body,
                })

        # 5. Did the product itself disappoint them?
        complaints = [t for t in negative if t.get("category") in DISSATISFACTION_CATEGORIES]
        if complaints:
            hit = complaints[-1]
            subject, body = self._draft("product_guidance", brief, evidence)
            return Move(
                f'Their {str(hit["category"]).replace("_", " ")} complaint is about the '
                f'product, not the service: "{_quote(hit["text"])}".',
                "propose_outreach", {
                    "cause": Cause.PRODUCT_DISSATISFACTION.value, "confidence": 0.7,
                    "action": "product_guidance",
                    "rationale": ("The complaint is about the product, not the price. "
                                  "Offer a fix or a different format; do not discount "
                                  "the thing they did not enjoy."),
                    "subject": subject, "body": body,
                })

        # 6. Nothing specific. They have run out and not come back.
        clicks = engagement.get("click_rate")
        engagement_line = (
            f" Click rate {clicks:.0%} across {engagement.get('emails_delivered', 0)} "
            f"emails — opens ignored, Apple MPP inflates them."
            if clicks is not None else "")
        subject, body = self._draft("replenishment_reminder", brief, evidence)
        return Move(
            f"No complaint, no life change, no price signal.{engagement_line} This looks "
            f"like an ordinary lapse.",
            "propose_outreach", {
                "cause": Cause.ROUTINE_LAPSE.value, "confidence": 0.55,
                "action": "replenishment_reminder",
                "rationale": ("One low-key reminder that they are probably out, with an "
                              "easy way to stop getting them. No offer — there is no "
                              "evidence price is the issue."),
                "subject": subject, "body": body,
            })

    # ------------------------------------------------------------------
    @staticmethod
    def _evidence(seen: dict[str, Any]) -> dict[str, Any]:
        return seen

    @staticmethod
    def _draft(action: str, brief: CustomerBrief, evidence: dict[str, Any]):
        from agent.copywriter import draft

        return draft(action, brief, evidence)
