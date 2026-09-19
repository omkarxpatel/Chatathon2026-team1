"""Tools the agent can call.

Each one is a plain function over the event store that returns JSON-safe
data. They are the agent's only way to see the customer: the LLM is never
handed the raw dataframes, and it is never handed the risk model.

TOOL_SCHEMAS below is the Anthropic tool-use format, so the real LLM path
can do genuine tool calling. The stub calls the same functions directly.
Both paths see identical data, which is what makes the stub a fair
stand-in rather than a mock.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable

import pandas as pd

import config as cfg
from data import catalog
from data.store import CustomerEvents, EventStore
from agent.policy import last_targeted_send


def _days_ago(ts, as_of: date) -> int:
    return int((as_of - pd.Timestamp(ts).date()).days)


# --------------------------------------------------------------------------
# the tools
# --------------------------------------------------------------------------


def get_ticket_text(ev: CustomerEvents) -> dict[str, Any]:
    """Support tickets in full. This is the unstructured signal the whole
    agent layer exists to read -- do not summarise it away."""
    if len(ev.tickets) == 0:
        return {"count": 0, "tickets": [],
                "note": "This customer has never contacted support."}
    tickets = [
        {
            "ticket_id": r["ticket_id"],
            "days_ago": _days_ago(r["created_ts"], ev.as_of),
            "category": r["category"],
            "status": r["status"],
            "resolved": r["status"] == "resolved",
            "sentiment": round(float(r["sentiment"]), 2),
            "csat": None if pd.isna(r["csat"]) else int(r["csat"]),
            "text": r["text"],
        }
        for _, r in ev.tickets.iterrows()
    ]
    return {
        "count": len(tickets),
        "unresolved_count": sum(1 for t in tickets if not t["resolved"]),
        "tickets": sorted(tickets, key=lambda t: t["days_ago"]),
    }


def get_order_history(ev: CustomerEvents) -> dict[str, Any]:
    """What they bought, when, for how much, and whether a promo was used."""
    orders = ev.orders
    if len(orders) == 0:
        return {"count": 0, "orders": []}

    lines_by_order: dict[str, list[str]] = {}
    for _, ln in ev.order_lines.iterrows():
        product = catalog.CATALOG.get(str(ln["sku"]))
        name = product.name if product else str(ln["sku"])
        lines_by_order.setdefault(str(ln["order_id"]), []).append(
            f"{int(ln['quantity'])}x {name}"
        )

    rows = [
        {
            "order_id": r["order_id"],
            "days_ago": _days_ago(r["ts"], ev.as_of),
            "items": lines_by_order.get(str(r["order_id"]), []),
            "total": round(float(r["total"]), 2),
            "promo_code": None if pd.isna(r["promo_code"]) else str(r["promo_code"]),
            "status": r["status"],
        }
        for _, r in orders.iterrows()
    ]
    rows.sort(key=lambda r: r["days_ago"])
    promo_used = sum(1 for r in rows if r["promo_code"])
    return {
        "count": len(rows),
        "promo_order_share": round(promo_used / len(rows), 2),
        "orders": rows,
    }


def get_product_usage(ev: CustomerEvents) -> dict[str, Any]:
    """The depletion arithmetic, shown as arithmetic.

    Serving counts make this computable: a 30-serving tub at one scoop a
    day runs out in 30 days. The agent gets the working, not a verdict.
    """
    orders = ev.fulfilled_orders
    if len(orders) == 0:
        return {"has_orders": False,
                "note": "No delivered orders, so no supply can be calculated."}

    last = orders.iloc[-1]
    last_date = pd.Timestamp(last["ts"]).date()
    lines = ev.order_lines[ev.order_lines["order_id"] == last["order_id"]]

    breakdown = []
    for _, ln in lines.iterrows():
        sku = str(ln["sku"])
        product = catalog.CATALOG.get(sku)
        if not product:
            continue
        qty = int(ln["quantity"])
        breakdown.append({
            "sku": sku,
            "name": product.name,
            "quantity": qty,
            "servings": product.servings * qty,
            "serving_size": product.serving_size,
            "typical_servings_per_day": product.servings_per_day,
            "days_of_supply": round(product.expected_days_supply * qty, 1),
        })

    basket = [(b["sku"], b["quantity"]) for b in breakdown]
    total_supply = catalog.basket_days_supply(basket)
    elapsed = (ev.as_of - last_date).days
    remaining = total_supply - elapsed

    return {
        "has_orders": True,
        "last_order_days_ago": elapsed,
        "last_order_items": breakdown,
        # The anchor product is the one that decides when they are due back.
        "anchor_days_of_supply": round(total_supply, 1),
        "estimated_days_remaining": round(remaining, 1),
        "ran_out": remaining < 0,
        "days_since_ran_out": round(-remaining, 1) if remaining < 0 else 0,
        "explanation": (
            f"Last delivered order was {elapsed} days ago and contained about "
            f"{total_supply:.0f} days of product at typical usage, so they have "
            + (f"roughly {remaining:.0f} days left."
               if remaining >= 0 else
               f"been out for roughly {-remaining:.0f} days.")
        ),
    }


def check_contact_cooldown(ev: CustomerEvents) -> dict[str, Any]:
    """Whether we are allowed to send anything at all right now."""
    last = last_targeted_send(ev)
    if last is None:
        return {"in_cooldown": False, "last_targeted_send": None,
                "cooldown_days": cfg.COOLDOWN_DAYS,
                "note": "No targeted outreach has ever been sent to this customer."}
    days = (ev.as_of - last).days
    return {
        "in_cooldown": days < cfg.COOLDOWN_DAYS,
        "last_targeted_send": last.isoformat(),
        "days_since_last_targeted_send": days,
        "cooldown_days": cfg.COOLDOWN_DAYS,
    }


def get_engagement_summary(ev: CustomerEvents) -> dict[str, Any]:
    """Email and browsing behaviour in plain numbers.

    Opens are reported but flagged: Apple Mail Privacy Protection fires
    them automatically for a slice of the list, so a high open rate next
    to a zero click rate means nothing.
    """
    m = ev.messages
    delivered = int((m["event"] == "delivered").sum()) if len(m) else 0
    clicks = int((m["event"] == "click").sum()) if len(m) else 0
    opens = int((m["event"] == "open").sum()) if len(m) else 0

    sessions = ev.sessions
    recent = 0
    if len(sessions):
        recent = int((sessions["ts"] >= pd.Timestamp(ev.as_of) - pd.Timedelta(days=30)).sum())

    return {
        "emails_delivered": delivered,
        "clicks": clicks,
        "click_rate": round(clicks / delivered, 3) if delivered else None,
        "opens": opens,
        "open_rate": round(opens / delivered, 3) if delivered else None,
        "open_rate_caveat": "Opens are inflated by Apple MPP. Trust clicks.",
        "sessions_last_30d": recent,
        "days_since_last_session": (
            int((ev.as_of - sessions["ts"].max().date()).days) if len(sessions) else None
        ),
    }


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

TOOLS: dict[str, Callable[[CustomerEvents], dict[str, Any]]] = {
    "get_ticket_text": get_ticket_text,
    "get_order_history": get_order_history,
    "get_product_usage": get_product_usage,
    "check_contact_cooldown": check_contact_cooldown,
    "get_engagement_summary": get_engagement_summary,
}


def run_all(ev: CustomerEvents) -> dict[str, Any]:
    """Call every tool. Cheap here, and it keeps the stub and the real
    model looking at exactly the same evidence."""
    return {name: fn(ev) for name, fn in TOOLS.items()}


def run(name: str, ev: CustomerEvents) -> dict[str, Any]:
    if name not in TOOLS:
        raise KeyError(f"unknown tool: {name}. Available: {sorted(TOOLS)}")
    return TOOLS[name](ev)


# Anthropic tool-use schemas, for the real-LLM path in diagnose.py. Every
# tool takes the customer_id it is already scoped to, so the model has no
# way to ask about somebody else.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": name,
        "description": (fn.__doc__ or "").strip().split("\n")[0],
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer under review.",
                }
            },
            "required": ["customer_id"],
        },
    }
    for name, fn in TOOLS.items()
]
