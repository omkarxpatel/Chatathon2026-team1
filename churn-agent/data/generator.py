"""Synthetic event-stream generator for a DTC sports nutrition brand.

DESIGN RULE (the one that matters)
----------------------------------
Every customer has a LATENT STATE -- still_training, brand_affinity,
price_sensitivity, goal, a buy-till-you-die purchase rate and dropout
probability. Nothing observable is a deterministic readout of it.

    latent state  --(noise)-->  order timing
                  --(noise)-->  order size / basket
                  --(noise)-->  session frequency
                  --(noise)-->  email opens and clicks
                  --(noise)-->  ticket sentiment and CSAT

The latent state is written to a separate file (latent_truth.json) that
features/ and scoring/ never import. Churn is therefore something the model
has to *infer* from a noisy multi-signal picture, not something it can read
off a column. That is what makes the demo honest.

Churn label: no fulfilled order in the LABEL_WINDOW_DAYS window AFTER
AS_OF_DATE. Features are computed strictly before AS_OF_DATE.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

import config as cfg
from data import catalog
from data.schema import (
    ChurnLabel,
    Customer,
    Goal,
    LatentState,
    MessageEvent,
    MessageEventType,
    Order,
    OrderLine,
    SupportTicket,
    TicketCategory,
    TicketStatus,
    WebSession,
)
from data.text_bank import CAMPAIGN_TYPES, TARGETED_CAMPAIGN_TYPES, pick as pick_text

REGIONS = ["US-West", "US-East", "US-Central", "US-South", "CA"]
CHANNELS = ["paid_social", "organic_search", "referral", "influencer", "email_capture"]

# Roughly a quarter of the list has Apple Mail Privacy Protection firing
# opens automatically. This is why the feature extractor leans on CLICKS.
MPP_SHARE = 0.25


# ==========================================================================
# Small helpers
# ==========================================================================


def _ts(d: date, rng: np.random.Generator) -> datetime:
    """Attach a plausible time of day to a date."""
    hour = int(np.clip(rng.normal(15, 4), 6, 23))
    return datetime.combine(d, time(hour, int(rng.integers(0, 60))))


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _tone(sentiment: float) -> str:
    if sentiment < -0.2:
        return "neg"
    if sentiment > 0.35:
        return "pos"
    return "neu"


def _affinity_at(latent: LatentState, d: date) -> float:
    """Brand affinity drifts linearly across the simulation."""
    span = max((cfg.SIM_END - cfg.SIM_START).days, 1)
    t = float(np.clip((d - cfg.SIM_START).days / span, 0.0, 1.0))
    return _clip01(
        latent.brand_affinity_start
        + t * (latent.brand_affinity_end - latent.brand_affinity_start)
    )


def _training_at(latent: LatentState, d: date) -> bool:
    return not (latent.training_stopped_on and d >= latent.training_stopped_on)


def _goal_at(latent: LatentState, d: date) -> Goal:
    if latent.goal_changed_on and d < latent.goal_changed_on and latent.previous_goal:
        return latent.previous_goal
    return latent.goal


# ==========================================================================
# Latent state
# ==========================================================================


def draw_latent_state(customer_id: str, signup: date, rng: np.random.Generator) -> LatentState:
    affinity_start = float(np.clip(rng.beta(5, 3), 0.05, 0.98))
    drift = rng.normal(-0.04, cfg.BRAND_AFFINITY_DRIFT)  # slight downward pull
    affinity_end = _clip01(affinity_start + drift)

    goal = Goal(rng.choice([g.value for g in Goal], p=[0.38, 0.34, 0.28]))
    previous_goal, goal_changed_on = None, None
    if rng.random() < cfg.P_GOAL_CHANGES:
        others = [g for g in Goal if g != goal]
        previous_goal = Goal(rng.choice([g.value for g in others]))
        goal_changed_on = cfg.SIM_START + timedelta(
            days=int(rng.integers(20, max(21, (cfg.AS_OF_DATE - cfg.SIM_START).days)))
        )

    training_stopped_on = None
    if rng.random() < cfg.P_TRAINING_STOPS:
        training_stopped_on = cfg.SIM_START + timedelta(
            days=int(rng.integers(30, max(31, (cfg.SIM_END - cfg.SIM_START).days - 20)))
        )

    return LatentState(
        customer_id=customer_id,
        still_training=training_stopped_on is None,
        training_stopped_on=training_stopped_on,
        brand_affinity_start=affinity_start,
        brand_affinity_end=affinity_end,
        price_sensitivity=float(np.clip(rng.beta(2.4, 3.2), 0.02, 0.98)),
        goal=goal,
        goal_changed_on=goal_changed_on,
        previous_goal=previous_goal,
        # Gamma-distributed purchase speed: population heterogeneity, mean 1.0
        latent_purchase_rate=float(rng.gamma(shape=5.0, scale=0.2)),
        dropout_p=float(np.clip(rng.beta(1.3, 14.0), 0.005, 0.5)),
        satisfaction=float(np.clip(rng.beta(6, 2.6) * 0.6 + affinity_start * 0.4, 0.02, 0.99)),
        true_dropout_date=None,   # filled in during simulation
        alive_at_cutoff=True,     # filled in during simulation
    )


# ==========================================================================
# Behaviour simulation
# ==========================================================================


def _pick_basket(goal: Goal, rng: np.random.Generator) -> list[tuple[str, int]]:
    weights = catalog.GOAL_AFFINITY[goal]
    skus = list(weights)
    probs = np.array([weights[s] for s in skus], dtype=float)
    probs /= probs.sum()

    roll = rng.random()
    n_lines = 1 if roll < 0.55 else (2 if roll < 0.88 else 3)
    n_lines = min(n_lines, len(skus))

    chosen = rng.choice(skus, size=n_lines, replace=False, p=probs)
    return [(str(s), 2 if rng.random() < 0.12 else 1) for s in chosen]


def _build_order(
    customer_id: str, seq: int, d: date, lines: list[tuple[str, int]],
    latent: LatentState, rng: np.random.Generator, status: str = "fulfilled",
) -> tuple[Order, list[OrderLine]]:
    order_id = f"{customer_id}-O{seq:03d}"
    line_models, subtotal = [], 0.0
    for sku, qty in lines:
        price = catalog.get(sku).price
        total = round(price * qty, 2)
        subtotal += total
        line_models.append(OrderLine(
            order_id=order_id, customer_id=customer_id, sku=sku,
            quantity=qty, unit_price=price, line_total=total,
        ))

    # Promo usage is driven by price sensitivity -- noisily.
    p_promo = _clip01(0.08 + 0.72 * latent.price_sensitivity + rng.normal(0, 0.12))
    promo_code, discount = None, 0.0
    if rng.random() < p_promo:
        rate = float(rng.choice([0.10, 0.15, 0.20], p=[0.5, 0.35, 0.15]))
        promo_code = f"SAVE{int(rate * 100)}"
        discount = round(subtotal * rate, 2)

    order = Order(
        order_id=order_id, customer_id=customer_id, ts=_ts(d, rng),
        subtotal=round(subtotal, 2), discount_amount=discount,
        promo_code=promo_code, total=round(subtotal - discount, 2),
        status=status, n_lines=len(line_models),
    )
    return order, line_models


def _simulate_orders(
    customer: Customer, latent: LatentState, rng: np.random.Generator
) -> tuple[list[Order], list[OrderLine], list[tuple[date, float]]]:
    """Buy-till-you-die: order, then flip a coin on whether you are still alive.

    Also returns (order_date, days_of_supply) pairs so the session
    simulator knows when each customer actually ran out of product.
    """
    orders: list[Order] = []
    lines: list[OrderLine] = []
    supply_log: list[tuple[date, float]] = []

    d = customer.signup_date
    seq = 0
    while d <= cfg.SIM_END and seq < 60:
        seq += 1
        basket = _pick_basket(_goal_at(latent, d), rng)
        roll = rng.random()
        status = "refunded" if roll < 0.02 else ("cancelled" if roll < 0.028 else "fulfilled")
        order, line_models = _build_order(
            customer.customer_id, seq, d, basket, latent, rng, status=status
        )
        orders.append(order)
        lines.extend(line_models)
        supply_log.append((d, catalog.basket_days_supply(basket)))

        # Dropout coin flip, BG/NBD style. Two things raise it: having
        # stopped training, and having lost interest in the brand. The
        # second is what makes churn learnable at all -- affinity drives
        # session rate and click rate too, so a fading customer leaves a
        # noisy trail before they go. It is never deterministic.
        if _training_at(latent, d):
            # Two latent drivers, weighted: how much they like the brand
            # and how well they have been treated. Both leave observable
            # trails (affinity -> sessions and clicks, satisfaction ->
            # ticket sentiment and resolution), so churn is learnable
            # without being readable.
            health = 0.7 * _affinity_at(latent, d) + 0.3 * latent.satisfaction
            p_drop = float(np.clip(
                latent.dropout_p * (1.0 + 3.0 * (1.0 - health)), 0.005, 0.6
            ))
        else:
            p_drop = 0.72
        if rng.random() < p_drop:
            latent.true_dropout_date = d
            break

        # Next order = when the anchor product runs out, stretched by how
        # much they still care, how much the price bothers them, and their
        # personal purchase speed.
        supply = catalog.basket_days_supply(basket)
        affinity = _affinity_at(latent, d)
        stretch = 1.0 / float(np.clip(affinity, 0.35, 1.0))
        # Price-sensitive customers run the tub down and then hold out for
        # a promo instead of reordering the day they run out.
        price_stretch = 1.0 + 0.85 * latent.price_sensitivity
        speed = float(np.clip(latent.latent_purchase_rate, 0.45, 2.2))
        noise = float(rng.lognormal(0.0, cfg.ORDER_TIMING_NOISE))

        gap = max(5.0, supply * stretch * price_stretch * noise / speed)
        d = d + timedelta(days=int(round(gap)))

    latent.alive_at_cutoff = (
        latent.true_dropout_date is None or latent.true_dropout_date > cfg.AS_OF_DATE
    )
    # Only events inside the recorded window exist as far as the model knows.
    keep = [o for o in orders if cfg.SIM_START <= o.ts.date() <= cfg.SIM_END]
    keep_ids = {o.order_id for o in keep}
    return keep, [ln for ln in lines if ln.order_id in keep_ids], supply_log


def _engagement_at(latent: LatentState, d: date) -> float:
    """Composite drive to interact with the brand. Never exposed directly."""
    eng = _affinity_at(latent, d)
    if not _training_at(latent, d):
        eng *= 0.25
    if latent.true_dropout_date and d > latent.true_dropout_date:
        # Interest decays after going dormant rather than stopping dead.
        days = (d - latent.true_dropout_date).days
        eng *= float(0.5 ** (days / 25.0))
    return _clip01(eng)


def _simulate_sessions(
    customer: Customer, latent: LatentState, orders: list[Order],
    supply_log: list[tuple[date, float]], rng: np.random.Generator,
) -> list[WebSession]:
    sessions: list[WebSession] = []
    order_dates = {o.ts.date() for o in orders}
    start = max(customer.signup_date, cfg.SIM_START)

    n = 0
    d = start
    while d <= cfg.SIM_END:
        eng = _engagement_at(latent, d)

        # Out-of-stock browsing. Once the tub is empty and they have not
        # reordered, price-sensitive customers keep coming back to check
        # whether anything has gone on sale. This is what produces the
        # "lots of sessions, no orders" signature the model needs to see
        # in the population before it can recognise it in an individual.
        overdue_boost = 0.0
        prior = [(od, sup) for od, sup in supply_log if od <= d]
        if prior:
            last_od, last_supply = prior[-1]
            if (d - last_od).days > last_supply:
                overdue_boost = 1.8 * latent.price_sensitivity * max(eng, 0.15)

        weekly_rate = (
            (0.08 + 0.80 * eng + overdue_boost)
            * float(rng.lognormal(0, cfg.SESSION_RATE_NOISE))
        )
        count = int(rng.poisson(weekly_rate))

        # Pre-purchase browsing burst.
        upcoming = any(0 <= (od - d).days <= 6 for od in order_dates)
        if upcoming:
            count += int(rng.integers(1, 4))

        goal = _goal_at(latent, d)
        for _ in range(count):
            n += 1
            sd = d + timedelta(days=int(rng.integers(0, 7)))
            if sd > cfg.SIM_END:
                continue
            viewed = [str(s) for s in rng.choice(
                list(catalog.GOAL_AFFINITY[goal]),
                size=int(rng.integers(1, 4)), replace=False,
            )]
            # Price-sensitive browsers add to cart and stall more often.
            p_cart = _clip01(0.18 + 0.35 * eng + 0.22 * latent.price_sensitivity)
            added = rng.random() < p_cart
            checkout = added and rng.random() < 0.45
            converted = any(0 <= (od - sd).days <= 2 for od in order_dates) and checkout
            sessions.append(WebSession(
                session_id=f"{customer.customer_id}-S{n:04d}",
                customer_id=customer.customer_id,
                ts=_ts(sd, rng),
                duration_s=int(np.clip(rng.lognormal(4.9, 0.8), 20, 2400)),
                pages_viewed=int(np.clip(rng.poisson(2 + 5 * eng) + 1, 1, 30)),
                products_viewed=viewed,
                added_to_cart=bool(added),
                checkout_started=bool(checkout),
                converted=bool(converted),
            ))
        d += timedelta(days=7)
    return sessions


def _simulate_messages(
    customer: Customer, latent: LatentState, rng: np.random.Generator
) -> list[MessageEvent]:
    """Brand-wide campaign cadence; per-customer response is the signal."""
    events: list[MessageEvent] = []
    mpp = rng.random() < MPP_SHARE  # auto-opens inflate this customer's open rate
    unsubscribed = False

    start = max(customer.signup_date, cfg.SIM_START)
    d = start + timedelta(days=int(rng.integers(0, 5)))
    n = 0
    while d <= cfg.SIM_END:
        if unsubscribed:
            break
        n += 1
        mid = f"{customer.customer_id}-M{n:03d}"
        # ~12% of sends are targeted flow messages rather than broadcast.
        targeted = rng.random() < 0.12
        pool = TARGETED_CAMPAIGN_TYPES if targeted else CAMPAIGN_TYPES
        campaign = str(rng.choice(pool))
        eng = _engagement_at(latent, d)
        base = _ts(d, rng)

        def emit(kind: MessageEventType, offset_h: int = 0) -> None:
            events.append(MessageEvent(
                event_id=f"{mid}-{kind.value}", message_id=mid,
                customer_id=customer.customer_id,
                ts=base + timedelta(hours=offset_h),
                channel="email", campaign_type=campaign, event=kind,
            ))

        emit(MessageEventType.SENT, 0)
        if rng.random() < 0.015:
            emit(MessageEventType.BOUNCE, 1)
            d += timedelta(days=int(rng.integers(4, 8)))
            continue
        emit(MessageEventType.DELIVERED, 1)

        p_open = _clip01(0.10 + 0.58 * eng + rng.normal(0, cfg.EMAIL_ENGAGEMENT_NOISE))
        if mpp:
            p_open = max(p_open, 0.88)  # machine open, means nothing
        if rng.random() < p_open:
            emit(MessageEventType.OPEN, 3)
            p_click = _clip01(0.06 + 0.48 * eng + rng.normal(0, cfg.EMAIL_ENGAGEMENT_NOISE))
            if rng.random() < p_click:
                emit(MessageEventType.CLICK, 5)

        if rng.random() < _clip01(0.0035 - 0.0031 * eng):
            emit(MessageEventType.UNSUBSCRIBE, 7)
            unsubscribed = True
        elif rng.random() < _clip01(0.0015 - 0.0013 * eng):
            emit(MessageEventType.COMPLAINT, 7)
            unsubscribed = True

        d += timedelta(days=int(rng.integers(4, 8)))
    return events


def _simulate_tickets(
    customer: Customer, latent: LatentState, rng: np.random.Generator
) -> list[SupportTicket]:
    tickets: list[SupportTicket] = []
    span_days = (cfg.SIM_END - max(customer.signup_date, cfg.SIM_START)).days
    if span_days <= 0:
        return tickets

    expected = 0.9 + 2.6 * (1.0 - latent.satisfaction)
    count = int(rng.poisson(expected))
    for i in range(count):
        offset = int(rng.integers(0, max(1, span_days)))
        created = max(customer.signup_date, cfg.SIM_START) + timedelta(days=offset)

        sentiment = float(np.clip(
            (latent.satisfaction - 0.5) * 1.8 + rng.normal(0, cfg.TICKET_SENTIMENT_NOISE),
            -1.0, 1.0,
        ))
        tone = _tone(sentiment)
        if tone == "pos":
            category = TicketCategory.PRAISE if rng.random() < 0.5 else TicketCategory.PRODUCT_QUESTION
        elif tone == "neg":
            category = TicketCategory(rng.choice([
                TicketCategory.DAMAGED_SHIPMENT.value, TicketCategory.SHIPPING_DELAY.value,
                TicketCategory.TASTE_MIXABILITY.value, TicketCategory.BILLING.value,
                TicketCategory.RETURN_REFUND.value,
            ]))
        else:
            category = TicketCategory(rng.choice([
                TicketCategory.PRODUCT_QUESTION.value, TicketCategory.SHIPPING_DELAY.value,
                TicketCategory.BILLING.value,
            ]))

        resolved = rng.random() < _clip01(0.62 + 0.3 * latent.satisfaction)
        resolved_ts, csat, status = None, None, TicketStatus.OPEN
        if resolved:
            status = TicketStatus.RESOLVED
            resolved_ts = _ts(created + timedelta(days=int(rng.integers(0, 6))), rng)
            csat = int(np.clip(round(3 + 2 * sentiment + rng.normal(0, 0.6)), 1, 5))

        tickets.append(SupportTicket(
            ticket_id=f"{customer.customer_id}-T{i + 1:02d}",
            customer_id=customer.customer_id,
            created_ts=_ts(created, rng), resolved_ts=resolved_ts,
            category=category, status=status, sentiment=round(sentiment, 3),
            csat=csat, text=pick_text(category, tone, rng),
        ))
    return tickets


# ==========================================================================
# Seeded demo archetypes
#
# These five are hand-built rather than sampled, so the stage demo is
# identical every run. Each one exists to prove a specific point about why
# a risk score alone is not a decision.
# ==========================================================================

_A = cfg.AS_OF_DATE


def _d(days_before_cutoff: int) -> date:
    return _A - timedelta(days=days_before_cutoff)


def _fixed_customer(cid: str, archetype: str, signup_offset: int) -> Customer:
    return Customer(
        customer_id=cid, signup_date=_d(signup_offset), region="US-West",
        acquisition_channel="organic_search", archetype=archetype,
    )


def _orders_from_plan(
    cid: str, plan: list[tuple[int, list[tuple[str, int]], str | None, str]],
    rng: np.random.Generator,
) -> tuple[list[Order], list[OrderLine]]:
    """plan = [(days_before_cutoff, [(sku, qty)], promo_code, status), ...]"""
    orders, lines = [], []
    for seq, (offset, basket, promo, status) in enumerate(sorted(plan, key=lambda r: -r[0]), 1):
        d = _d(offset)
        order_id = f"{cid}-O{seq:03d}"
        subtotal, line_models = 0.0, []
        for sku, qty in basket:
            price = catalog.get(sku).price
            total = round(price * qty, 2)
            subtotal += total
            line_models.append(OrderLine(
                order_id=order_id, customer_id=cid, sku=sku,
                quantity=qty, unit_price=price, line_total=total,
            ))
        rate = 0.15 if promo else 0.0
        discount = round(subtotal * rate, 2)
        orders.append(Order(
            order_id=order_id, customer_id=cid, ts=_ts(d, rng),
            subtotal=round(subtotal, 2), discount_amount=discount, promo_code=promo,
            total=round(subtotal - discount, 2), status=status, n_lines=len(line_models),
        ))
        lines.extend(line_models)
    return orders, lines


def _campaign_stream(
    cid: str, rng: np.random.Generator, open_rate: float, click_rate: float,
    every: int = 6, last_send_days_before: int = 0, stop_before_cutoff: int = 0,
    targeted_last: bool = False,
) -> list[MessageEvent]:
    """Regular sends with a fixed response rate. Deliberately simple: for the
    archetypes we want the engagement level to be exactly what we say it is."""
    events: list[MessageEvent] = []
    offsets = list(range(int((_A - cfg.SIM_START).days), stop_before_cutoff - 1, -every))
    if last_send_days_before and last_send_days_before not in offsets:
        offsets.append(last_send_days_before)
    for n, off in enumerate(sorted(set(offsets), reverse=True), 1):
        d = _d(off)
        if d < cfg.SIM_START:
            continue
        mid = f"{cid}-M{n:03d}"
        # The final send can be forced to a targeted flow message -- that is
        # what puts CUST-0004 inside the cooldown window.
        is_last_targeted = targeted_last and off == last_send_days_before
        campaign = str(rng.choice(
            TARGETED_CAMPAIGN_TYPES if is_last_targeted else CAMPAIGN_TYPES
        ))
        base = _ts(d, rng)

        def emit(kind: MessageEventType, hours: int) -> None:
            events.append(MessageEvent(
                event_id=f"{mid}-{kind.value}", message_id=mid, customer_id=cid,
                ts=base + timedelta(hours=hours), channel="email",
                campaign_type=campaign, event=kind,
            ))

        emit(MessageEventType.SENT, 0)
        emit(MessageEventType.DELIVERED, 1)
        # Recent sends are less engaged for lapsing customers.
        decay = 1.0 if off > 60 else 0.55
        if rng.random() < open_rate * decay:
            emit(MessageEventType.OPEN, 3)
            if rng.random() < click_rate * decay:
                emit(MessageEventType.CLICK, 4)
    return events


def _sessions_at(
    cid: str, rng: np.random.Generator, spec: list[tuple[int, int, bool]]
) -> list[WebSession]:
    """spec = [(days_before_cutoff, n_sessions, added_to_cart), ...]"""
    out, n = [], 0
    for offset, count, cart in spec:
        for _ in range(count):
            n += 1
            d = _d(offset) + timedelta(days=int(rng.integers(0, 3)))
            out.append(WebSession(
                session_id=f"{cid}-S{n:04d}", customer_id=cid, ts=_ts(d, rng),
                duration_s=int(np.clip(rng.lognormal(5.2, 0.6), 30, 2400)),
                pages_viewed=int(rng.integers(3, 12)),
                products_viewed=[str(s) for s in rng.choice(catalog.SKUS, size=2, replace=False)],
                added_to_cart=cart, checkout_started=cart and rng.random() < 0.6,
                converted=False,
            ))
    return out


def _ticket(
    cid: str, i: int, offset: int, category: TicketCategory, sentiment: float,
    text: str, resolved: bool, csat: int | None, rng: np.random.Generator,
) -> SupportTicket:
    created = _d(offset)
    return SupportTicket(
        ticket_id=f"{cid}-T{i:02d}", customer_id=cid, created_ts=_ts(created, rng),
        resolved_ts=_ts(created + timedelta(days=2), rng) if resolved else None,
        category=category,
        status=TicketStatus.RESOLVED if resolved else TicketStatus.OPEN,
        sentiment=sentiment, csat=csat, text=text,
    )


def build_archetypes(rng: np.random.Generator) -> dict:
    """Returns dict of lists keyed by table name, plus latent states."""
    customers, orders, lines, sessions, messages, tickets, latents = [], [], [], [], [], [], []

    def add(c, o, ln, s, m, t, lat):
        customers.append(c); orders.extend(o); lines.extend(ln)
        sessions.extend(s); messages.extend(m); tickets.extend(t); latents.append(lat)

    # ---------------------------------------------------------------- 1
    # TRUE POSITIVE, DIAGNOSABLE CAUSE.
    # Steady 32-day cadence, then a 90-day gap = 2.8x their personal median.
    # Two unresolved, angry tickets about a damaged shipment; the order that
    # triggered them was refunded. Correct action is SERVICE RECOVERY -- a
    # discount here would be insulting, not helpful.
    cid = "CUST-0001"
    c = _fixed_customer(cid, "service_recovery", 400)
    plan = [(o, [("WHEY-2LB", 1), ("CREA-350", 1)], None, "fulfilled")
            for o in (250, 218, 186, 154, 122, 90)]
    # The -90 order arrived damaged, so they reordered at -74. That one was
    # damaged too and got refunded. Last usable delivery is therefore 90 days
    # ago: 90 / 32-day median = 2.8x.
    plan.append((74, [("WHEY-2LB", 1)], None, "refunded"))
    o, ln = _orders_from_plan(cid, plan, rng)
    add(c, o, ln,
        _sessions_at(cid, rng, [(250, 2, True), (190, 2, True), (125, 3, True),
                                (95, 2, True), (70, 1, False), (30, 1, False)]),
        _campaign_stream(cid, rng, open_rate=0.38, click_rate=0.12),
        [_ticket(cid, 1, 85, TicketCategory.DAMAGED_SHIPMENT, -0.78,
                 "Box was crushed and the lid had popped off. Half the tub is gone. "
                 "I paid full price for this and I'd like it sorted out properly.",
                 False, None, rng),
         _ticket(cid, 2, 70, TicketCategory.DAMAGED_SHIPMENT, -0.88,
                 "Second tub in a row arrived with the seal broken and powder all "
                 "through the box. I sent photos two weeks ago and nobody has replied. "
                 "I've been a customer for over a year and this is how it goes?",
                 False, None, rng)],
        LatentState(customer_id=cid, still_training=True, brand_affinity_start=0.78,
                    brand_affinity_end=0.22, price_sensitivity=0.25, goal=Goal.BULKING,
                    latent_purchase_rate=1.0, dropout_p=0.05,
                    true_dropout_date=_d(74), satisfaction=0.12, alive_at_cutoff=False))

    # ---------------------------------------------------------------- 2
    # FALSE POSITIVE -- STILL STOCKED.
    # Usual cadence is 18 days on 1 lb tubs. Last order was a 5 lb tub (76
    # days of supply) 55 days ago, so the "3x median gap" is an artefact of
    # pack size. They have ~21 days of protein left. Correct action: NONE.
    cid = "CUST-0002"
    c = _fixed_customer(cid, "false_positive_stocked", 300)
    plan = [(o, [("WHEY-1LB", 1)], None, "fulfilled")
            for o in (199, 181, 163, 145, 127, 109, 91, 73)]
    plan.append((55, [("WHEY-5LB", 1), ("CREA-350", 1)], None, "fulfilled"))
    # Negative offset = AFTER the cutoff. They reorder on schedule once the
    # 5 lb tub runs down, which is what makes them a false positive.
    plan.append((-28, [("WHEY-5LB", 1)], None, "fulfilled"))
    o, ln = _orders_from_plan(cid, plan, rng)
    add(c, o, ln,
        _sessions_at(cid, rng, [(160, 2, True), (110, 2, True), (60, 3, True), (20, 1, False)]),
        _campaign_stream(cid, rng, open_rate=0.62, click_rate=0.34),
        [_ticket(cid, 1, 140, TicketCategory.PRODUCT_QUESTION, 0.30,
                 "Do you do a bigger size of the isolate? Going through the 1 lb "
                 "tubs pretty fast.", True, 5, rng)],
        LatentState(customer_id=cid, still_training=True, brand_affinity_start=0.80,
                    brand_affinity_end=0.82, price_sensitivity=0.30, goal=Goal.BULKING,
                    latent_purchase_rate=1.3, dropout_p=0.02,
                    true_dropout_date=None, satisfaction=0.85, alive_at_cutoff=True))

    # ---------------------------------------------------------------- 3
    # PRICE SENSITIVITY.
    # Browsing hard (9 sessions in 30 days, carting, never checking out),
    # no order in 52 days, and 85% of past orders used a promo code. They
    # are waiting for a sale. Correct action: value that is NOT a percentage
    # off -- bundle economics, cost per serving, loyalty pricing.
    cid = "CUST-0003"
    c = _fixed_customer(cid, "price_sensitive", 320)
    promo_plan = [(o, [("WHEY-2LB", 1)], "SAVE15", "fulfilled")
                  for o in (235, 205, 175, 145, 115)]
    promo_plan.append((85, [("WHEY-2LB", 1)], "SAVE20", "fulfilled"))
    promo_plan.insert(0, (265, [("PRE-25", 1)], None, "fulfilled"))  # 1 of 7 full price
    o, ln = _orders_from_plan(cid, promo_plan, rng)
    add(c, o, ln,
        _sessions_at(cid, rng, [(30, 3, True), (21, 3, True), (12, 3, True), (4, 3, True),
                                (150, 1, True), (95, 1, True)]),
        _campaign_stream(cid, rng, open_rate=0.40, click_rate=0.11),
        [_ticket(cid, 1, 34, TicketCategory.BILLING, -0.10,
                 "Do you offer a subscribe-and-save price? Paying full retail every "
                 "month adds up and I'm comparing against a couple of other brands "
                 "right now.", True, 3, rng)],
        LatentState(customer_id=cid, still_training=True, brand_affinity_start=0.60,
                    brand_affinity_end=0.45, price_sensitivity=0.94, goal=Goal.CUTTING,
                    latent_purchase_rate=0.9, dropout_p=0.08,
                    true_dropout_date=_d(85), satisfaction=0.48, alive_at_cutoff=False))

    # ---------------------------------------------------------------- 4
    # COOLDOWN SUPPRESSION.
    # Genuinely at risk and genuinely actionable -- but marketing emailed
    # them 6 days ago. The deterministic policy gate stops this one BEFORE
    # the agent runs. No LLM tokens are spent on a customer we are not
    # allowed to contact.
    cid = "CUST-0004"
    c = _fixed_customer(cid, "cooldown_suppressed", 280)
    plan = [(o, [("WHEY-2LB", 1), ("ELEC-30", 1)], None, "fulfilled")
            for o in (210, 175, 140, 105)]
    o, ln = _orders_from_plan(cid, plan, rng)
    add(c, o, ln,
        _sessions_at(cid, rng, [(200, 2, True), (140, 2, True), (100, 2, True), (45, 1, False)]),
        _campaign_stream(cid, rng, open_rate=0.30, click_rate=0.10, every=9,
                         last_send_days_before=6, targeted_last=True),
        [],
        LatentState(customer_id=cid, still_training=True, brand_affinity_start=0.55,
                    brand_affinity_end=0.20, price_sensitivity=0.40, goal=Goal.CUTTING,
                    latent_purchase_rate=0.95, dropout_p=0.10,
                    true_dropout_date=_d(105), satisfaction=0.50, alive_at_cutoff=False))

    # ---------------------------------------------------------------- 5
    # LEGITIMATE STOP.
    # Torn rotator cuff, out of the gym for months. High CSAT, zero
    # complaints, still opens the newsletter. Genuinely churned and
    # genuinely unsaveable. Correct action: NONE. Reaching out to sell
    # protein to someone who just told support they are injured is the
    # tasteless failure mode this whole project exists to avoid.
    cid = "CUST-0005"
    c = _fixed_customer(cid, "legitimate_stop", 360)
    plan = [(o, [("WHEY-2LB", 1), ("CREA-350", 1)], None, "fulfilled")
            for o in (215, 180, 145, 115, 85)]
    o, ln = _orders_from_plan(cid, plan, rng)
    add(c, o, ln,
        _sessions_at(cid, rng, [(210, 2, True), (150, 2, True), (88, 2, True), (55, 1, False)]),
        _campaign_stream(cid, rng, open_rate=0.58, click_rate=0.12),
        [_ticket(cid, 1, 150, TicketCategory.PRAISE, 0.90,
                 "Just wanted to say the isolate is the best I've used - no bloating "
                 "at all. Been on it since last spring.", True, 5, rng),
         _ticket(cid, 2, 78, TicketCategory.PRODUCT_QUESTION, 0.35,
                 "I've torn my rotator cuff and I'm out of the gym for at least three "
                 "months. Is there any point continuing the creatine while I'm not "
                 "training? Not a complaint about the product at all - I'll be back.",
                 True, 5, rng)],
        LatentState(customer_id=cid, still_training=False, training_stopped_on=_d(80),
                    brand_affinity_start=0.88, brand_affinity_end=0.80,
                    price_sensitivity=0.20, goal=Goal.MAINTENANCE,
                    latent_purchase_rate=1.0, dropout_p=0.03,
                    true_dropout_date=_d(85), satisfaction=0.92, alive_at_cutoff=False))

    return {
        "customers": customers, "orders": orders, "order_lines": lines,
        "sessions": sessions, "messages": messages, "tickets": tickets,
        "latents": latents,
    }


# ==========================================================================
# Assembly
# ==========================================================================

# Columns that must come back as real timestamps after the JSON round-trip.
_DATETIME_COLS = {
    "customers": ["signup_date"],
    "orders": ["ts"],
    "order_lines": [],
    "web_sessions": ["ts"],
    "message_events": ["ts"],
    "support_tickets": ["created_ts", "resolved_ts"],
    "labels": ["as_of"],
}


def _frame(models: list, model_cls, table: str) -> pd.DataFrame:
    if models:
        df = pd.DataFrame([m.model_dump(mode="json") for m in models])
    else:
        df = pd.DataFrame({k: pd.Series(dtype="object") for k in model_cls.model_fields})
    for col in _DATETIME_COLS.get(table, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _simulate_random_customer(idx: int, seed: int) -> tuple:
    """Draw a customer until they have enough pre-cutoff history to score.

    Customers whose entire order history fell before the observation window
    are not interesting (and not scoreable), so we redraw. This is a
    sampling convenience, not a modelling assumption -- the redraw touches
    only the latent parameters, never the observables.
    """
    cid = f"CUST-{idx + 1:04d}"
    for attempt in range(25):
        rng = np.random.default_rng([seed, idx, attempt])
        span = (cfg.SIGNUP_LATEST - cfg.SIGNUP_EARLIEST).days
        signup = cfg.SIGNUP_EARLIEST + timedelta(days=int(rng.integers(0, span)))
        customer = Customer(
            customer_id=cid, signup_date=signup,
            region=str(rng.choice(REGIONS)),
            acquisition_channel=str(rng.choice(CHANNELS)),
            archetype=None,
        )
        latent = draw_latent_state(cid, signup, rng)
        orders, lines, supply_log = _simulate_orders(customer, latent, rng)
        pre = [o for o in orders if o.ts.date() < cfg.AS_OF_DATE]
        if len(pre) >= 2:
            break
    sessions = _simulate_sessions(customer, latent, orders, supply_log, rng)
    messages = _simulate_messages(customer, latent, rng)
    tickets = _simulate_tickets(customer, latent, rng)
    return customer, orders, lines, sessions, messages, tickets, latent


def _personal_cycle_days(
    cid: str, orders: list[Order], lines_by_order: dict[str, list[OrderLine]]
) -> float:
    """The customer's own reorder cadence, from PRE-CUTOFF data only.

    Mirrors features.extract.cadence_features by design: the label window
    has to be derived the same way the model will later describe the
    customer. Kept as a local copy rather than an import so that the
    generator never depends on the feature pipeline (and vice versa).
    """
    mine = sorted(
        (o for o in orders
         if o.customer_id == cid and o.status == "fulfilled"
         and o.ts.date() < cfg.AS_OF_DATE),
        key=lambda o: o.ts,
    )
    if not mine:
        return float(cfg.LABEL_WINDOW_MIN_DAYS)

    dates = [o.ts.date() for o in mine]
    gaps = [float((b - a).days) for a, b in zip(dates, dates[1:])]
    if gaps:
        return float(np.median(gaps))

    # Single order: fall back to how long what they bought actually lasts.
    basket = [(ln.sku, ln.quantity) for ln in lines_by_order.get(mine[-1].order_id, [])
              if ln.sku in catalog.CATALOG]
    return float(catalog.basket_days_supply(basket)) or float(cfg.LABEL_WINDOW_MIN_DAYS)


def _build_labels(
    customers: list[Customer], orders: list[Order], lines: list[OrderLine]
) -> list[ChurnLabel]:
    lines_by_order: dict[str, list[OrderLine]] = {}
    for ln in lines:
        lines_by_order.setdefault(ln.order_id, []).append(ln)

    forward: dict[str, list[date]] = {}
    for o in orders:
        if o.status == "fulfilled" and o.ts.date() > cfg.AS_OF_DATE:
            forward.setdefault(o.customer_id, []).append(o.ts.date())

    labels = []
    for c in customers:
        cid = c.customer_id
        cycle = _personal_cycle_days(cid, orders, lines_by_order)
        window = int(np.clip(
            round(cycle * cfg.LABEL_CYCLE_MULTIPLE),
            cfg.LABEL_WINDOW_MIN_DAYS, cfg.LABEL_WINDOW_MAX_DAYS,
        ))
        deadline = cfg.AS_OF_DATE + timedelta(days=window)
        n = sum(1 for d in forward.get(cid, []) if d <= deadline)
        labels.append(ChurnLabel(
            customer_id=cid, as_of=cfg.AS_OF_DATE, window_days=window,
            cycle_days=round(cycle, 1), orders_in_window=n, churned=n == 0,
        ))
    return labels


def generate(n_customers: int | None = None, seed: int | None = None) -> dict[str, pd.DataFrame]:
    """Build the full synthetic cohort and write it to data/out/."""
    n = n_customers or cfg.N_CUSTOMERS
    seed = cfg.SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    bundle = build_archetypes(rng)
    customers = list(bundle["customers"])
    orders = list(bundle["orders"])
    lines = list(bundle["order_lines"])
    sessions = list(bundle["sessions"])
    messages = list(bundle["messages"])
    tickets = list(bundle["tickets"])
    latents = list(bundle["latents"])

    n_seeded = len(customers)
    for idx in range(n_seeded, n):
        c, o, ln, s, m, t, lat = _simulate_random_customer(idx, seed)
        customers.append(c); orders.extend(o); lines.extend(ln)
        sessions.extend(s); messages.extend(m); tickets.extend(t); latents.append(lat)

    labels = _build_labels(customers, orders, lines)

    frames = {
        "customers": _frame(customers, Customer, "customers"),
        "orders": _frame(orders, Order, "orders"),
        "order_lines": _frame(lines, OrderLine, "order_lines"),
        "web_sessions": _frame(sessions, WebSession, "web_sessions"),
        "message_events": _frame(messages, MessageEvent, "message_events"),
        "support_tickets": _frame(tickets, SupportTicket, "support_tickets"),
        "labels": _frame(labels, ChurnLabel, "labels"),
    }

    cfg.DATA_OUT.mkdir(parents=True, exist_ok=True)
    frames["customers"].to_parquet(cfg.CUSTOMERS_FILE, index=False)
    frames["orders"].to_parquet(cfg.ORDERS_FILE, index=False)
    frames["order_lines"].to_parquet(cfg.ORDER_LINES_FILE, index=False)
    frames["web_sessions"].to_parquet(cfg.SESSIONS_FILE, index=False)
    frames["message_events"].to_parquet(cfg.MESSAGES_FILE, index=False)
    frames["support_tickets"].to_parquet(cfg.TICKETS_FILE, index=False)
    frames["labels"].to_parquet(cfg.LABELS_FILE, index=False)

    # Held out. Deliberately JSON and deliberately not parquet, so it is
    # obvious at a glance that it is not part of the feature pipeline.
    with open(cfg.LATENT_TRUTH_FILE, "w") as fh:
        json.dump(
            {
                "_warning": (
                    "GROUND TRUTH -- generator output only. Importing this from "
                    "features/ or scoring/ is label leakage and smoke_test.py "
                    "will fail the build."
                ),
                "as_of": cfg.AS_OF_DATE.isoformat(),
                "states": [lat.model_dump(mode="json") for lat in latents],
            },
            fh,
            indent=2,
        )

    return frames
