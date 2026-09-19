"""The agent's hands: what it can look at, and what it can do about it.

Two kinds of tool, and the split is the whole design.

INVESTIGATE tools read one customer's history. They are the agent's only
window onto the customer -- it is never handed the raw dataframes, and it
is never handed the risk model. It decides which of these to call and in
what order, and every call it makes is recorded and shown to the reviewer.

DECIDE tools end the run. There are exactly three ways for the agent to
finish -- propose outreach, recommend no contact, or escalate to a human
-- and each one is a tool call with a schema. That is deliberate: the
agent cannot dribble out a conclusion in prose, it has to commit to a
structured answer that the guardrails can then check.

Everything is described in Anthropic tool-use format, so the real model
in agent/claude_agent.py calls these directly. The deterministic brain in
agent/rules_agent.py calls the same registry through the same loop, which
is what makes it a fair stand-in rather than a mock.

Every tool is pre-scoped to the customer under review. There is no
argument anywhere that lets the agent ask about somebody else.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

import pandas as pd

import config as cfg
from agent.policy import last_targeted_send
from agent.schemas import Action, AgentDiagnosis, Cause, StepKind
from data import catalog
from data.store import CustomerEvents


class ToolError(ValueError):
    """Bad arguments from the agent. Fed back into the loop so it can retry."""


@dataclass(frozen=True)
class Tool:
    """One callable capability, described once for both brains and the UI."""

    name: str
    kind: StepKind
    label: str                                   # what the reviewer sees
    blurb: str                                   # one line of it, for the UI
    description: str                             # what the model sees
    input_schema: dict[str, Any]
    run: Callable[[CustomerEvents, dict], dict] | None = None
    headline: Callable[[dict], str] | None = None

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema}


def _obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


def _days_ago(ts, as_of: date) -> int:
    return int((as_of - pd.Timestamp(ts).date()).days)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# ==========================================================================
# INVESTIGATE -- reading the customer
# ==========================================================================


def read_support_tickets(ev: CustomerEvents, args: dict) -> dict[str, Any]:
    only_unresolved = bool(args.get("only_unresolved", False))
    rows = [
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
    rows.sort(key=lambda t: t["days_ago"])
    unresolved = [t for t in rows if not t["resolved"]]
    shown = unresolved if only_unresolved else rows
    return {
        "count": len(rows),
        "unresolved_count": len(unresolved),
        "filter": "unresolved only" if only_unresolved else "all tickets",
        "tickets": shown,
        "note": ("This customer has never contacted support."
                 if not rows else
                 "Full text is included on purpose -- do not skim it."),
    }


def _tickets_headline(o: dict) -> str:
    if not o["count"]:
        return "Never contacted support"
    unresolved = o["unresolved_count"]
    if not unresolved:
        return f"{_plural(o['count'], 'support ticket')}, all resolved"
    oldest = max((t["days_ago"] for t in o["tickets"] if not t["resolved"]), default=0)
    return (f"{_plural(o['count'], 'support ticket')}, {unresolved} still open "
            f"— oldest is {oldest} days old")


def read_order_history(ev: CustomerEvents, args: dict) -> dict[str, Any]:
    limit = int(args.get("limit", 12) or 12)
    if limit < 1:
        raise ToolError("limit must be at least 1")
    orders = ev.orders
    if len(orders) == 0:
        return {"count": 0, "orders": [], "note": "No orders on record."}

    lines_by_order: dict[str, list[str]] = {}
    for _, ln in ev.order_lines.iterrows():
        product = catalog.CATALOG.get(str(ln["sku"]))
        name = product.name if product else str(ln["sku"])
        lines_by_order.setdefault(str(ln["order_id"]), []).append(
            f"{int(ln['quantity'])}x {name}")

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
        "failed_or_refunded": [r["order_id"] for r in rows
                               if r["status"] in ("refunded", "cancelled")],
        "showing": min(limit, len(rows)),
        "orders": rows[:limit],
    }


def _orders_headline(o: dict) -> str:
    if not o["count"]:
        return "No orders on record"
    parts = [f"{_plural(o['count'], 'order')}, last one {o['orders'][0]['days_ago']} days ago"]
    if o["promo_order_share"] >= 0.5:
        parts.append(f"{o['promo_order_share']:.0%} used a promo code")
    if o["failed_or_refunded"]:
        parts.append(f"{len(o['failed_or_refunded'])} refunded or cancelled")
    return " — ".join(parts)


def check_product_supply(ev: CustomerEvents, args: dict) -> dict[str, Any]:
    """The depletion arithmetic, shown as arithmetic."""
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
            "sku": sku, "name": product.name, "quantity": qty,
            "servings": product.servings * qty,
            "serving_size": product.serving_size,
            "typical_servings_per_day": product.servings_per_day,
            "days_of_supply": round(product.expected_days_supply * qty, 1),
        })

    total_supply = catalog.basket_days_supply([(b["sku"], b["quantity"]) for b in breakdown])
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
               f"been out for roughly {-remaining:.0f} days.")),
    }


def _supply_headline(o: dict) -> str:
    if not o.get("has_orders"):
        return "No delivered orders to work from"
    anchor = max(o["last_order_items"], key=lambda i: i["days_of_supply"], default=None)
    pack = f" ({anchor['name']})" if anchor else ""
    if o["ran_out"]:
        return f"Ran out about {o['days_since_ran_out']:.0f} days ago{pack}"
    return f"Still has about {o['estimated_days_remaining']:.0f} days of product left{pack}"


def check_contact_policy(ev: CustomerEvents, args: dict) -> dict[str, Any]:
    """Whether we are allowed to contact them at all right now."""
    opted_out = False
    if len(ev.messages):
        opted_out = bool(ev.messages["event"].isin(["unsubscribe", "complaint"]).any())
    last = last_targeted_send(ev)
    if last is None:
        return {"opted_out": opted_out, "in_cooldown": False,
                "last_targeted_send": None, "cooldown_days": cfg.COOLDOWN_DAYS,
                "note": "No targeted outreach has ever been sent to this customer."}
    days = (ev.as_of - last).days
    return {
        "opted_out": opted_out,
        "in_cooldown": days < cfg.COOLDOWN_DAYS,
        "last_targeted_send": last.isoformat(),
        "days_since_last_targeted_send": days,
        "cooldown_days": cfg.COOLDOWN_DAYS,
    }


def _contact_headline(o: dict) -> str:
    if o.get("opted_out"):
        return "Opted out of marketing — never contact"
    if o.get("last_targeted_send") is None:
        return "No targeted outreach ever sent"
    days = o["days_since_last_targeted_send"]
    if o["in_cooldown"]:
        return f"Last targeted email {days} days ago — inside the {o['cooldown_days']}-day cooldown"
    return f"Last targeted email {days} days ago — clear to contact"


def read_engagement(ev: CustomerEvents, args: dict) -> dict[str, Any]:
    """Email and browsing behaviour in plain numbers.

    Opens are reported but flagged: Apple Mail Privacy Protection fires
    them automatically for a slice of the list, so a high open rate next
    to a zero click rate means nothing.
    """
    window = int(args.get("window_days", 30) or 30)
    if window < 1:
        raise ToolError("window_days must be at least 1")

    m = ev.messages
    delivered = int((m["event"] == "delivered").sum()) if len(m) else 0
    clicks = int((m["event"] == "click").sum()) if len(m) else 0
    opens = int((m["event"] == "open").sum()) if len(m) else 0

    sessions = ev.sessions
    recent = 0
    if len(sessions):
        cutoff = pd.Timestamp(ev.as_of) - pd.Timedelta(days=window)
        recent = int((sessions["ts"] >= cutoff).sum())
    return {
        "emails_delivered": delivered,
        "clicks": clicks,
        "click_rate": round(clicks / delivered, 3) if delivered else None,
        "opens": opens,
        "open_rate": round(opens / delivered, 3) if delivered else None,
        "open_rate_caveat": "Opens are inflated by Apple MPP. Trust clicks.",
        "window_days": window,
        "sessions_in_window": recent,
        "days_since_last_session": (
            int((ev.as_of - sessions["ts"].max().date()).days) if len(sessions) else None),
    }


def _engagement_headline(o: dict) -> str:
    rate = o["click_rate"]
    clicks = ("no clicks" if not o["clicks"]
              else f"{o['clicks']} clicks ({rate:.0%})") if rate is not None else "no email history"
    since = o["days_since_last_session"]
    visit = "never visited the site" if since is None else f"last visit {since} days ago"
    return f"{clicks} from {o['emails_delivered']} emails — {visit}"


INVESTIGATE_TOOLS: list[Tool] = [
    Tool(
        name="check_product_supply", kind=StepKind.INVESTIGATE,
        label="Check product supply",
        blurb="Works out from serving counts whether they have actually run out, or just bought a bigger tub.",
        description=(
            "Work out whether this customer has actually run out. Returns the "
            "serving count of their last delivered order, the typical servings "
            "per day, and the resulting days of supply remaining -- negative "
            "means they ran out that many days ago. Call this FIRST: a long gap "
            "after a 76-day tub is arithmetic, not disengagement, and there is "
            "nothing to diagnose if they are still stocked."),
        input_schema=_obj({}),
        run=check_product_supply, headline=_supply_headline,
    ),
    Tool(
        name="read_support_tickets", kind=StepKind.INVESTIGATE,
        label="Read support tickets",
        blurb="Reads every support conversation in full, including the ones nobody answered.",
        description=(
            "Full text of every support conversation, with category, status, "
            "sentiment and CSAT. This is the unstructured signal the whole agent "
            "layer exists to read -- an unanswered damaged-shipment complaint and "
            "a torn rotator cuff both live here and both change the answer."),
        input_schema=_obj({"only_unresolved": {
            "type": "boolean",
            "description": "Return only tickets that are still open. Default false."}}),
        run=read_support_tickets, headline=_tickets_headline,
    ),
    Tool(
        name="read_order_history", kind=StepKind.INVESTIGATE,
        label="Read order history",
        blurb="What they bought, when, and whether they only ever buy on promotion.",
        description=(
            "What they bought, when, for how much, whether a promo code was used, "
            "and whether anything was refunded or cancelled. Use it to tell a "
            "price-led buyer from a steady one."),
        input_schema=_obj({"limit": {
            "type": "integer",
            "description": "How many recent orders to return. Default 12."}}),
        run=read_order_history, headline=_orders_headline,
    ),
    Tool(
        name="read_engagement", kind=StepKind.INVESTIGATE,
        label="Check email and site activity",
        blurb="Email clicks and site visits. Clicks, not opens — opens are inflated by Apple Mail.",
        description=(
            "Email clicks, opens and site sessions. Judge engagement on clicks: "
            "opens are inflated by Apple Mail Privacy Protection and mean little. "
            "Browsing without buying is intent with something in the way."),
        input_schema=_obj({"window_days": {
            "type": "integer",
            "description": "Session window in days. Default 30."}}),
        run=read_engagement, headline=_engagement_headline,
    ),
    Tool(
        name="check_contact_policy", kind=StepKind.INVESTIGATE,
        label="Check contact permission",
        blurb="Confirms we are allowed to contact them and have not just emailed them.",
        description=(
            "Whether we are allowed to contact this person right now: opt-out "
            "status and days since the last targeted send against the cooldown. "
            "Broadcast newsletters do not count toward the cooldown."),
        input_schema=_obj({}),
        run=check_contact_policy, headline=_contact_headline,
    ),
]


# ==========================================================================
# DECIDE -- the three ways a run can end
# ==========================================================================

_CAUSE_ENUM = [c.value for c in Cause]
_OUTREACH_ACTIONS = [
    Action.REPLENISHMENT_REMINDER, Action.PRODUCT_GUIDANCE, Action.VALUE_EDUCATION,
    Action.SERVICE_RECOVERY, Action.LOYALTY_RECOGNITION, Action.DISCOUNT_OFFER,
]

_CONFIDENCE = {"type": "number", "minimum": 0, "maximum": 1,
               "description": "How sure you are of the cause, 0 to 1."}
_CAUSE = {"type": "string", "enum": _CAUSE_ENUM,
          "description": "Why this customer is lapsing."}

DECIDE_TOOLS: list[Tool] = [
    Tool(
        name="propose_outreach", kind=StepKind.DECIDE,
        label="Draft a message",
        blurb="Recommends a specific kind of outreach and writes the draft for you to edit.",
        description=(
            "Recommend contacting the customer, and write the message. Pick the "
            "action that fits the CAUSE, not the risk score. Reach for "
            "discount_offer almost never -- if price is genuinely the blocker, "
            "value_education (cost per serving, larger packs, subscribe-and-save) "
            "keeps the relationship instead of teaching them to wait for a sale. "
            "A person reviews every draft before it goes anywhere."),
        input_schema=_obj({
            "cause": _CAUSE,
            "confidence": _CONFIDENCE,
            "action": {"type": "string", "enum": [a.value for a in _OUTREACH_ACTIONS],
                       "description": "The kind of outreach to send."},
            "rationale": {"type": "string",
                          "description": "One or two sentences on why this action fits the cause. Shown to the reviewer."},
            "subject": {"type": "string", "description": "Subject line."},
            "body": {"type": "string",
                     "description": f"The message. Under {cfg.MAX_MESSAGE_WORDS} words, no urgency, no countdowns, and always a way to hear from us less often."},
        }, ["cause", "confidence", "action", "rationale", "subject", "body"]),
    ),
    Tool(
        name="recommend_no_contact", kind=StepKind.DECIDE,
        label="Recommend leaving them alone",
        blurb="Recommends sending nothing, and says why. A common and correct answer.",
        description=(
            "Recommend sending nothing. This is a correct and common answer, not "
            "a failure. Someone who has told support they are injured has not "
            "lapsed, they have stopped, and marketing protein at them is the "
            "failure mode this system exists to avoid. Say plainly why."),
        input_schema=_obj({
            "cause": _CAUSE,
            "confidence": _CONFIDENCE,
            "rationale": {"type": "string",
                          "description": "Why contacting them would be wrong or pointless. Shown to the reviewer."},
        }, ["cause", "confidence", "rationale"]),
    ),
    Tool(
        name="escalate_to_human", kind=StepKind.DECIDE,
        label="Hand to a teammate",
        blurb="Hands the customer to a named teammate with a briefing, instead of sending anything.",
        description=(
            "Hand the customer to a named person instead of sending anything. Use "
            "this when the situation has gone past what an automated message "
            "should touch -- several unresolved complaints, or anything where a "
            "template would make it worse."),
        input_schema=_obj({
            "cause": _CAUSE,
            "confidence": _CONFIDENCE,
            "rationale": {"type": "string", "description": "Why this needs a person."},
            "internal_note": {"type": "string",
                              "description": "Briefing for the teammate picking this up. Internal, never sent to the customer."},
        }, ["cause", "confidence", "rationale", "internal_note"]),
    ),
]


def _confidence(args: dict) -> float:
    try:
        value = float(args.get("confidence", 0.5))
    except (TypeError, ValueError):
        raise ToolError("confidence must be a number between 0 and 1")
    return min(1.0, max(0.0, value))


def _cause(args: dict) -> Cause:
    try:
        return Cause(str(args.get("cause", "")).strip().lower())
    except ValueError:
        raise ToolError(f"cause must be one of {_CAUSE_ENUM}")


def build_decision(tool_name: str, args: dict, customer_id: str, brain: str,
                   trace: list[str]) -> AgentDiagnosis:
    """Turn a DECIDE tool call into the diagnosis everything downstream reads.

    requires_human_approval is set here rather than taken from the agent.
    It is not the agent's to choose, and there is no send path for it to
    choose its way out of.
    """
    common = dict(customer_id=customer_id, cause=_cause(args),
                  cause_confidence=_confidence(args), reasoning_trace=list(trace),
                  requires_human_approval=True, diagnoser=brain)
    rationale = str(args.get("rationale", "")).strip()

    if tool_name == "recommend_no_contact":
        return AgentDiagnosis(**common, action=Action.NO_ACTION,
                              action_rationale=rationale or "No outreach recommended.")
    if tool_name == "escalate_to_human":
        note = str(args.get("internal_note", "")).strip()
        return AgentDiagnosis(
            **common, action=Action.HUMAN_ESCALATION,
            action_rationale=rationale or "Needs a person, not a template.",
            message_subject="[INTERNAL] Hand to a teammate — not customer-facing",
            message_body=note or rationale)
    if tool_name == "propose_outreach":
        try:
            action = Action(str(args.get("action", "")).strip().lower())
        except ValueError:
            raise ToolError(f"action must be one of {[a.value for a in _OUTREACH_ACTIONS]}")
        if action not in _OUTREACH_ACTIONS:
            raise ToolError(f"{action.value} is not an outreach action. Use "
                            f"recommend_no_contact or escalate_to_human instead.")
        body = str(args.get("body", "")).strip()
        if not body:
            raise ToolError("propose_outreach needs a message body.")
        return AgentDiagnosis(**common, action=action,
                              action_rationale=rationale or f"Proposed {action.value}.",
                              message_subject=str(args.get("subject", "")).strip(),
                              message_body=body)
    raise ToolError(f"unknown decision tool: {tool_name}")


# ==========================================================================
# registry
# ==========================================================================

ALL_TOOLS: list[Tool] = INVESTIGATE_TOOLS + DECIDE_TOOLS
TOOLS: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}
TOOL_SCHEMAS: list[dict[str, Any]] = [t.schema() for t in ALL_TOOLS]

TOOL_LABELS: dict[str, str] = {t.name: t.label for t in ALL_TOOLS}


def investigate(name: str, ev: CustomerEvents, args: dict | None = None) -> tuple[dict, str]:
    """Run one investigate tool. Returns (observation, one-line headline)."""
    tool = TOOLS.get(name)
    if tool is None or tool.run is None:
        raise ToolError(f"unknown tool: {name}. Available: "
                        f"{sorted(t.name for t in ALL_TOOLS)}")
    observation = tool.run(ev, args or {})
    return observation, (tool.headline(observation) if tool.headline else "")
