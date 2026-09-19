"""Customer events -> feature vector. Pure functions, no I/O, no LLM.

This is layer 1 of the two-layer architecture. Everything here is
arithmetic you could do in a spreadsheet, which is the point: the number
that drives the decision has to be auditable.

TEMPORAL FIREWALL
    The caller hands in a CustomerEvents that has already been truncated
    at `as_of` (see data/store.py). Nothing in this module reads a file,
    so it cannot reach past that truncation even by accident.

LEAKAGE FIREWALL
    This module does not import LatentState and does not read
    latent_truth.json. smoke_test.py greps for both and fails the build if
    that ever changes.

The interesting features are the product-physics ones. `days_of_supply_
remaining` asks a physical question -- have they actually run out? -- and
is what stops a 5 lb tub buyer from looking identical to a churner.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from data.catalog import CATALOG, basket_days_supply
from data.store import CustomerEvents

# Window length used as the "never happened" sentinel for days-since
# features, so a customer who never clicked reads as 240 rather than
# something that blows up the scaler.
NEVER = 240.0

# A rate with no denominator is UNDEFINED, not zero. Writing 0.0 for
# "never added anything to a cart in 90 days" makes a dead customer look
# like a perfect converter and flips the coefficient sign. These come back
# as NaN and are median-imputed in the model pipeline; the fact that the
# customer was inactive is already carried by sessions_30d and the
# days_since_* features, so no information is lost.
UNDEFINED = float("nan")

FEATURE_NAMES: list[str] = [
    # cadence / recency
    "days_since_last_order",
    "personal_median_interpurchase",
    "reorder_gap_ratio",
    "last_gap_vs_median",
    # product physics
    "days_of_supply_remaining",
    "supply_overdue_ratio",
    # order economics
    "order_count_lifetime",
    "order_count_180d",
    "aov_last",
    "aov_trend",
    "promo_order_share",
    "mean_discount_depth",
    "refund_rate",
    "distinct_skus",
    # web behaviour
    "sessions_30d",
    "session_trend_30_over_90",
    "sessions_since_last_order",
    "cart_adds_since_last_order",
    "days_since_last_session",
    # email engagement
    "click_rate_90d",
    "click_rate_trend",
    "days_since_last_click",
    "open_rate_90d",
    # support
    "tickets_90d",
    "unresolved_tickets",
    "mean_ticket_sentiment",
    "last_csat",
    "days_since_negative_ticket",
    # tenure
    "tenure_days",
]

# --------------------------------------------------------------------------
# Which features the MODEL sees, and why it is a short list.
# --------------------------------------------------------------------------
# All 29 features above are computed and shown in the dashboard. Only these
# eight are fed to the scorer.
#
# Why so few? With ~93 churn events in a 200-customer cohort, fitting 24
# predictors gives under 4 events per variable. The accepted floor is 10.
# Below it, logistic coefficients are unstable: correlated features trade
# weight and their SIGNS FLIP, so the attribution panel starts claiming
# things like "more product on hand means more churn risk". An attribution
# you cannot defend out loud is worse than no attribution.
#
# Eight predictors on 93 events is ~11.6 events per variable, and the
# signs come out reading like common sense. One feature per signal family,
# each the most interpretable member of its correlated group.
MODEL_FEATURES: list[str] = [
    "reorder_gap_ratio",          # cadence, personalised to the customer
    "click_rate_90d",             # email engagement (clicks, not MPP-inflated opens)
    "days_since_last_session",    # web recency
    # Mostly reads as engagement: someone still visiting has not left.
    # The "browsing but blocked on price" reading is an interaction with
    # promo_order_share that a linear model cannot express -- diagnosing
    # that is the agent layer's job, not the scorer's.
    "sessions_since_last_order",
    "promo_order_share",          # discount dependency
    "order_count_lifetime",       # depth of the relationship
]

# Note what is NOT in that list: anything from the support inbox.
#
# unresolved_tickets and mean_ticket_sentiment are both computed above and
# both shown in the dashboard, but neither is fed to the scorer. They are
# collinear with each other, sparse (most customers never write in), and
# they contribute nothing stable on a cohort this size -- their
# coefficients sit at noise level and their signs wander between runs.
#
# The better reason is architectural. "Two open tickets about a damaged
# shipment, photos sent, nobody replied for two weeks" is not a number.
# Compressing it to a count and handing it to a linear model throws away
# the only part that tells you what to actually do about it. The agent
# reads those tickets directly via get_ticket_text.
#
# So the split is clean: structured behaviour goes to the model,
# unstructured text goes to the LLM. Each layer gets the kind of signal it
# is actually good at.

# days_of_supply_remaining is deliberately NOT here, and not because it is
# weak -- it is the most important number in the project. It is not a
# statistical correlate, it is arithmetic: servings / servings-per-day. We
# know it, we do not need to estimate it. Asking a logistic regression to
# learn a fact we can compute is how you get a coefficient with the wrong
# sign on your most important feature.
#
# So it is enforced as a hard rule in the policy gate instead (see
# agent/policy.py: "still has product on hand" suppresses outreach
# whatever the model score says) and shown prominently in the drill-down.
# The model handles uncertainty; the gate handles certainty.
EXCLUDED_FROM_MODEL: list[str] = [f for f in FEATURE_NAMES if f not in MODEL_FEATURES]

# Plain-English labels for the dashboard. Keeping them next to the feature
# list means a new feature that forgets a label is obvious.
FEATURE_LABELS: dict[str, str] = {
    "days_since_last_order": "Days since last delivered order",
    "personal_median_interpurchase": "Their own median reorder interval",
    "reorder_gap_ratio": "Current gap as a multiple of their median",
    "last_gap_vs_median": "Previous gap vs. their median (were they already slowing?)",
    "days_of_supply_remaining": "Days of product still on hand (negative = ran out)",
    "supply_overdue_ratio": "Gap as a multiple of the pack's days-of-supply",
    "order_count_lifetime": "Orders in the observation window",
    "order_count_180d": "Orders in the last 180 days",
    "aov_last": "Value of their last order",
    "aov_trend": "Recent order value vs. earlier orders",
    "promo_order_share": "Share of orders placed with a promo code",
    "mean_discount_depth": "Average discount taken",
    "refund_rate": "Share of orders refunded or cancelled",
    "distinct_skus": "Distinct products bought",
    "sessions_30d": "Site sessions in the last 30 days",
    "session_trend_30_over_90": "Recent browsing vs. their 90-day rate",
    "sessions_since_last_order": "Site visits since their last order",
    "cart_adds_since_last_order": "Add-to-carts since their last order",
    "days_since_last_session": "Days since they last visited",
    "click_rate_90d": "Email click rate, last 90 days",
    "click_rate_trend": "Recent click rate vs. 90-day click rate",
    "days_since_last_click": "Days since they last clicked an email",
    "open_rate_90d": "Email open rate, last 90 days (inflated by Apple MPP)",
    "tickets_90d": "Support tickets in the last 90 days",
    "unresolved_tickets": "Support tickets still open",
    "mean_ticket_sentiment": "Average ticket sentiment (-1 to +1)",
    "last_csat": "Most recent CSAT score (1-5)",
    "days_since_negative_ticket": "Days since their last negative ticket",
    "tenure_days": "Days since signup",
}


# ==========================================================================
# helpers
# ==========================================================================


def _days(a: date, b: date) -> float:
    return float((a - b).days)


def _safe_ratio(num: float, den: float, default: float = UNDEFINED, cap: float = 10.0) -> float:
    if den is None or den <= 0 or not np.isfinite(den):
        return default
    return float(np.clip(num / den, -cap, cap))


def _count_between(ts: pd.Series, as_of: date, lo_days: int, hi_days: int = 0) -> int:
    """Events in the window [as_of - lo_days, as_of - hi_days)."""
    if ts is None or len(ts) == 0:
        return 0
    lo = pd.Timestamp(as_of) - pd.Timedelta(days=lo_days)
    hi = pd.Timestamp(as_of) - pd.Timedelta(days=hi_days)
    return int(((ts >= lo) & (ts < hi)).sum())


# ==========================================================================
# feature groups
# ==========================================================================


def cadence_features(ev: CustomerEvents) -> dict[str, float]:
    orders = ev.fulfilled_orders
    if len(orders) == 0:
        return {
            "days_since_last_order": NEVER,
            "personal_median_interpurchase": NEVER,
            "reorder_gap_ratio": 0.0,
            "last_gap_vs_median": 0.0,
        }

    dates = [t.date() for t in orders["ts"]]
    last = max(dates)
    since = _days(ev.as_of, last)

    gaps = [float((b - a).days) for a, b in zip(dates, dates[1:])]
    # A single-order customer has no personal baseline; fall back to the
    # catalog cadence of what they bought rather than inventing one.
    median = float(np.median(gaps)) if gaps else _expected_supply_of_last_order(ev) or 30.0

    return {
        "days_since_last_order": since,
        "personal_median_interpurchase": median,
        "reorder_gap_ratio": _safe_ratio(since, median, cap=8.0),
        "last_gap_vs_median": _safe_ratio(gaps[-1], median, cap=8.0) if gaps else 1.0,
    }


def _expected_supply_of_last_order(ev: CustomerEvents) -> float:
    """Days of product the most recent delivered order contained."""
    orders = ev.fulfilled_orders
    if len(orders) == 0 or len(ev.order_lines) == 0:
        return 0.0
    last_id = orders.iloc[-1]["order_id"]
    lines = ev.order_lines[ev.order_lines["order_id"] == last_id]
    basket = [
        (str(r["sku"]), int(r["quantity"]))
        for _, r in lines.iterrows()
        if str(r["sku"]) in CATALOG
    ]
    return float(basket_days_supply(basket))


def supply_features(ev: CustomerEvents) -> dict[str, float]:
    """Product physics: a 30-serving tub at 1 scoop/day runs out in 30 days.

    This is the feature that separates 'lapsed' from 'still has product'.
    """
    orders = ev.fulfilled_orders
    supply = _expected_supply_of_last_order(ev)
    if len(orders) == 0 or supply <= 0:
        return {"days_of_supply_remaining": -NEVER, "supply_overdue_ratio": 8.0}

    last = max(t.date() for t in orders["ts"])
    since = _days(ev.as_of, last)
    return {
        "days_of_supply_remaining": float(np.clip(supply - since, -180.0, 180.0)),
        "supply_overdue_ratio": _safe_ratio(since, supply, cap=8.0),
    }


def order_features(ev: CustomerEvents) -> dict[str, float]:
    all_orders = ev.orders
    orders = ev.fulfilled_orders
    n = len(orders)
    if n == 0:
        return {
            "order_count_lifetime": 0.0, "order_count_180d": 0.0, "aov_last": 0.0,
            "aov_trend": UNDEFINED, "promo_order_share": UNDEFINED,
            "mean_discount_depth": UNDEFINED, "refund_rate": UNDEFINED,
            "distinct_skus": 0.0,
        }

    totals = orders["total"].astype(float).to_numpy()
    recent = totals[-2:].mean()
    earlier = totals[:-2].mean() if n > 2 else UNDEFINED

    subtotal = orders["subtotal"].astype(float).to_numpy()
    discount = orders["discount_amount"].astype(float).to_numpy()
    depth = float(np.mean(np.divide(discount, np.where(subtotal > 0, subtotal, 1.0))))

    refunded = int((all_orders["status"] != "fulfilled").sum()) if len(all_orders) else 0

    return {
        "order_count_lifetime": float(n),
        "order_count_180d": float(_count_between(orders["ts"], ev.as_of, 180)),
        "aov_last": float(totals[-1]),
        "aov_trend": _safe_ratio(recent, earlier, cap=5.0) - 1.0,
        "promo_order_share": float(orders["promo_code"].notna().mean()),
        "mean_discount_depth": depth,
        "refund_rate": _safe_ratio(refunded, len(all_orders), cap=1.0),
        "distinct_skus": float(ev.order_lines["sku"].nunique()) if len(ev.order_lines) else 0.0,
    }


def session_features(ev: CustomerEvents) -> dict[str, float]:
    """Browsing behaviour, including the price-sensitivity tell.

    `sessions_since_last_order` is the one that matters: someone visiting
    repeatedly without buying is showing intent that something is
    blocking. A high count next to a high promo share is the signature of
    a customer waiting for a sale, not one drifting away.
    """
    s = ev.sessions
    if len(s) == 0:
        return {
            "sessions_30d": 0.0, "session_trend_30_over_90": UNDEFINED,
            "sessions_since_last_order": 0.0, "cart_adds_since_last_order": 0.0,
            "days_since_last_session": NEVER,
        }

    s30 = _count_between(s["ts"], ev.as_of, 30)
    s90 = _count_between(s["ts"], ev.as_of, 90)

    orders = ev.fulfilled_orders
    if len(orders):
        since_order = s[s["ts"] > orders["ts"].max()]
    else:
        since_order = s

    return {
        "sessions_30d": float(s30),
        # >1 means browsing is accelerating; <1 means going quiet.
        "session_trend_30_over_90": _safe_ratio(s30, s90 / 3.0, cap=6.0),
        "sessions_since_last_order": float(len(since_order)),
        "cart_adds_since_last_order": float(since_order["added_to_cart"].sum()),
        "days_since_last_session": _days(ev.as_of, s["ts"].max().date()),
    }


def email_features(ev: CustomerEvents) -> dict[str, float]:
    """Clicks are the primary signal. Opens are kept but distrusted:
    Apple Mail Privacy Protection fires them automatically for a chunk of
    the list, which is why open rate is a weaker feature than click rate."""
    m = ev.messages
    if len(m) == 0:
        return {
            "click_rate_90d": UNDEFINED, "click_rate_trend": UNDEFINED,
            "days_since_last_click": NEVER, "open_rate_90d": UNDEFINED,
        }

    def rate(event: str, window: int) -> float:
        sub = m[m["ts"] >= pd.Timestamp(ev.as_of) - pd.Timedelta(days=window)]
        delivered = int((sub["event"] == "delivered").sum())
        return _safe_ratio(int((sub["event"] == event).sum()), delivered, cap=1.0)

    clicks = m[m["event"] == "click"]
    c30, c90 = rate("click", 30), rate("click", 90)

    return {
        "click_rate_90d": c90,
        "click_rate_trend": _safe_ratio(c30, c90, cap=4.0),
        "days_since_last_click": (
            _days(ev.as_of, clicks["ts"].max().date()) if len(clicks) else NEVER
        ),
        "open_rate_90d": rate("open", 90),
    }


def support_features(ev: CustomerEvents) -> dict[str, float]:
    t = ev.tickets
    if len(t) == 0:
        # No tickets is not the same as unhappy. Neutral sentiment, neutral
        # CSAT, and "no negative ticket in the window".
        return {
            "tickets_90d": 0.0, "unresolved_tickets": 0.0,
            "mean_ticket_sentiment": UNDEFINED, "last_csat": UNDEFINED,
            "days_since_negative_ticket": NEVER,
        }

    recent = t[t["created_ts"] >= pd.Timestamp(ev.as_of) - pd.Timedelta(days=180)]
    csat = t["csat"].dropna()
    negative = t[t["sentiment"].astype(float) < -0.2]

    return {
        "tickets_90d": float(_count_between(t["created_ts"], ev.as_of, 90)),
        "unresolved_tickets": float((t["status"] == "open").sum()),
        "mean_ticket_sentiment": (
            float(recent["sentiment"].astype(float).mean()) if len(recent) else UNDEFINED
        ),
        "last_csat": float(csat.iloc[-1]) if len(csat) else UNDEFINED,
        "days_since_negative_ticket": (
            _days(ev.as_of, negative["created_ts"].max().date()) if len(negative) else NEVER
        ),
    }


# ==========================================================================
# entry point
# ==========================================================================


def extract(ev: CustomerEvents) -> dict[str, float]:
    """The one function the rest of the pipeline calls."""
    out: dict[str, float] = {}
    out.update(cadence_features(ev))
    out.update(supply_features(ev))
    out.update(order_features(ev))
    out.update(session_features(ev))
    out.update(email_features(ev))
    out.update(support_features(ev))
    out["tenure_days"] = max(0.0, _days(ev.as_of, ev.signup_date))

    missing = set(FEATURE_NAMES) - set(out)
    if missing:
        raise KeyError(f"feature group did not produce: {sorted(missing)}")
    return {k: float(out[k]) for k in FEATURE_NAMES}


def extract_frame(store, customer_ids: list[str] | None = None, as_of: date | None = None):
    """Feature matrix for a whole cohort, indexed by customer_id."""
    ids = customer_ids or store.customer_ids
    rows = {cid: extract(store.events_for(cid, as_of)) for cid in ids}
    return pd.DataFrame.from_dict(rows, orient="index")[FEATURE_NAMES]
