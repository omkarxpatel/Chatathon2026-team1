"""Message templates for the stub diagnoser.

Separate file because this is the part the team will iterate on most and
it needs no knowledge of the rules engine. The real LLM writes its own
copy; these are what the stub produces so the demo has something real to
show on screen.

House style, enforced here and checked again in guardrails.py:
  - say why you are writing in the first line
  - no urgency, no countdowns, no "don't miss out"
  - no percentage-off unless the diagnosis is specifically price
  - always give them a way to hear from us less
  - short
"""

from __future__ import annotations

from agent.schemas import Action, DiagnosisRequest

OPT_DOWN = "If you'd rather not get these, you can turn them off in your preferences."


def _anchor_product(request: DiagnosisRequest) -> tuple[str, float, int]:
    """Name, days of supply, and days since the last order's anchor item."""
    usage = request.tool_output.get("get_product_usage", {})
    items = usage.get("last_order_items") or []
    if not items:
        return "your usual order", 30.0, int(usage.get("last_order_days_ago", 0) or 0)
    anchor = max(items, key=lambda i: i.get("days_of_supply", 0))
    return (
        anchor.get("name", "your usual order"),
        float(anchor.get("days_of_supply", 30.0)),
        int(usage.get("last_order_days_ago", 0) or 0),
    )


def _damaged_item(request: DiagnosisRequest) -> str | None:
    """The product from the refunded/cancelled order, if there is one.

    Service recovery has to name the thing that actually went wrong. The
    anchor product from their last good delivery is the wrong answer --
    that is the order that worked.
    """
    orders = request.tool_output.get("get_order_history", {}).get("orders", [])
    broken = [o for o in orders if o.get("status") in ("refunded", "cancelled")]
    if not broken:
        return None
    items = sorted(broken, key=lambda o: o["days_ago"])[0].get("items") or []
    if not items:
        return None
    # items look like "1x Whey Isolate, 2 lb"
    return items[0].split("x ", 1)[-1]


def draft(action: Action, request: DiagnosisRequest) -> tuple[str | None, str | None]:
    if action == Action.NO_ACTION:
        return None, None

    name, supply, days_ago = _anchor_product(request)
    usage = request.tool_output.get("get_product_usage", {})
    tickets = request.tool_output.get("get_ticket_text", {}).get("tickets", [])
    out_for = int(usage.get("days_since_ran_out", 0) or 0)

    if action == Action.SERVICE_RECOVERY:
        unresolved = [t for t in tickets if not t.get("resolved")]
        oldest = max((t["days_ago"] for t in unresolved), default=0)
        broken_item = _damaged_item(request) or name
        return (
            "About your last order, and the reply you didn't get",
            (
                f"Hi,\n\n"
                f"You wrote to us {oldest} days ago about your order arriving damaged, "
                f"and you haven't had a proper answer. That's our mistake twice over "
                f"— the shipment and the silence — and I'm sorry.\n\n"
                f"Here's what I've done: the damaged order is refunded in full, and "
                f"I've flagged the packaging problem to our fulfilment team.\n\n"
                f"If you'd like a replacement {broken_item} sent out, reply and I'll arrange "
                f"it today at no charge — nothing to return. If you'd rather leave it, "
                f"that's completely fine and I won't chase you about it.\n\n"
                f"— Customer Care"
            ),
        )

    if action == Action.HUMAN_ESCALATION:
        return (
            "[INTERNAL] Escalate to a human — do not auto-send",
            (
                "Internal note, not customer-facing.\n\n"
                f"{len([t for t in tickets if not t.get('resolved')])} unresolved tickets, "
                f"all negative, oldest "
                f"{max((t['days_ago'] for t in tickets if not t.get('resolved')), default=0)} "
                f"days old. This has gone past what an automated message should touch.\n\n"
                "Recommend a named person calls or writes personally, resolves the open "
                "tickets first, and only then considers anything else. No marketing "
                "message should go to this customer until the tickets are closed."
            ),
        )

    if action == Action.VALUE_EDUCATION:
        orders = request.tool_output.get("get_order_history", {})
        promo_share = orders.get("promo_order_share", 0.0)
        return (
            "Some maths on cost per serving (no offer attached)",
            (
                f"Hi,\n\n"
                f"No promo in this email — just some arithmetic in case it's useful.\n\n"
                f"You've mostly been buying the 2 lb isolate: $42.99 for 30 servings, "
                f"which is $1.43 a scoop. The 5 lb tub is $89.99 for 76 servings — "
                f"$1.18 a scoop, so about 17% less per serving, and it lasts roughly "
                f"76 days instead of 30.\n\n"
                f"We also run subscribe-and-save, which holds a lower price "
                f"permanently rather than you having to catch a sale. You've used a "
                f"promo code on {promo_share:.0%} of your orders, so that may suit "
                f"you better than waiting for the next one.\n\n"
                f"If neither is right, no problem at all. {OPT_DOWN}\n\n"
                f"— Customer Care"
            ),
        )

    if action == Action.PRODUCT_GUIDANCE:
        return (
            "About the issue you mentioned",
            (
                f"Hi,\n\n"
                f"You mentioned a problem with the product rather than the service, "
                f"and I don't want to just sell you the same thing again.\n\n"
                f"Two things that might help: the unflavoured isolate mixes more "
                f"cleanly if the flavour was the issue, and we can swap an unopened "
                f"tub for a different one at no cost.\n\n"
                f"Tell me which and I'll sort it. If you've moved on to something "
                f"else, that's genuinely fine. {OPT_DOWN}\n\n"
                f"— Customer Care"
            ),
        )

    if action == Action.LOYALTY_RECOGNITION:
        return (
            "Thanks for sticking with us",
            (
                f"Hi,\n\n"
                f"No ask in this one. You've been ordering with us for a while and "
                f"we noticed. Thank you.\n\n"
                f"If there's ever something you want us to stock, reply and tell us. "
                f"{OPT_DOWN}\n\n"
                f"— Customer Care"
            ),
        )

    if action == Action.DISCOUNT_OFFER:
        return (
            "A one-off price on your usual",
            (
                f"Hi,\n\n"
                f"You've been buying {name} regularly and we'd rather not lose you "
                f"over price. Here's 15% on your next order if it helps.\n\n"
                f"If price wasn't the issue, tell us what was — that's more useful "
                f"to us than the sale. {OPT_DOWN}\n\n"
                f"— Customer Care"
            ),
        )

    # REPLENISHMENT_REMINDER
    ran_out_line = (
        f"by our maths you'd have run out about {out_for} days ago"
        if out_for > 0
        else "you're probably close to the end of it"
    )
    return (
        f"You're probably about out of {name}",
        (
            f"Hi,\n\n"
            f"Your last {name} went out {days_ago} days ago. At about a serving a "
            f"day that's roughly {supply:.0f} days' worth, so {ran_out_line}.\n\n"
            f"No offer attached and nothing to do if you're sorted — just flagging "
            f"it in case it slipped your mind.\n\n"
            f"If you've changed what you're taking, or stepped back from training "
            f"for a bit, you can update that in your preferences and we'll stop "
            f"sending these.\n\n"
            f"— Customer Care"
        ),
    )
