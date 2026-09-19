"""Pydantic models for every entity in the simulated event stream.

Two families of model live here, and the split is load-bearing:

  OBSERVABLE   Customer, Order, OrderLine, WebSession, MessageEvent,
               SupportTicket, ChurnLabel
               -> written to parquet, read by features/extract.py

  HELD OUT     LatentState
               -> written to latent_truth.json, read by NOBODY in
                  features/ or scoring/. It exists so we can audit the
                  model afterwards, not so the model can use it.

The feature extractor imports the observable models. It does not import
LatentState, and smoke_test.py fails the build if that ever changes.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ==========================================================================
# Enums
# ==========================================================================


class Goal(str, Enum):
    BULKING = "bulking"
    CUTTING = "cutting"
    MAINTENANCE = "maintenance"


class MessageEventType(str, Enum):
    SENT = "sent"
    DELIVERED = "delivered"
    OPEN = "open"
    CLICK = "click"
    BOUNCE = "bounce"
    UNSUBSCRIBE = "unsubscribe"
    COMPLAINT = "complaint"


class TicketCategory(str, Enum):
    DAMAGED_SHIPMENT = "damaged_shipment"
    SHIPPING_DELAY = "shipping_delay"
    PRODUCT_QUESTION = "product_question"
    TASTE_MIXABILITY = "taste_mixability"
    BILLING = "billing"
    RETURN_REFUND = "return_refund"
    PRAISE = "praise"


class TicketStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"


# ==========================================================================
# Observable entities
# ==========================================================================


class Customer(BaseModel):
    """The customer record a real CRM would hold. No behavioural summary
    lives here -- that is the feature extractor's job."""

    customer_id: str
    signup_date: date
    region: str
    acquisition_channel: str
    # Only populated for the five seeded demo customers; None for the rest.
    # Used purely for demo navigation in the dashboard, never as a feature.
    archetype: str | None = None


class OrderLine(BaseModel):
    order_id: str
    customer_id: str
    sku: str
    quantity: int = Field(ge=1)
    unit_price: float
    line_total: float


class Order(BaseModel):
    order_id: str
    customer_id: str
    ts: datetime
    subtotal: float
    discount_amount: float = 0.0
    promo_code: str | None = None
    total: float
    status: Literal["fulfilled", "cancelled", "refunded"] = "fulfilled"
    n_lines: int = 1


class WebSession(BaseModel):
    session_id: str
    customer_id: str
    ts: datetime
    duration_s: int
    pages_viewed: int
    products_viewed: list[str] = Field(default_factory=list)
    added_to_cart: bool = False
    checkout_started: bool = False
    converted: bool = False


class MessageEvent(BaseModel):
    """One row per event. A single send produces several rows sharing a
    message_id (sent -> delivered -> open -> click)."""

    event_id: str
    message_id: str
    customer_id: str
    ts: datetime
    channel: Literal["email", "sms"] = "email"
    campaign_type: str
    event: MessageEventType


class SupportTicket(BaseModel):
    ticket_id: str
    customer_id: str
    created_ts: datetime
    resolved_ts: datetime | None = None
    category: TicketCategory
    status: TicketStatus
    # -1.0 (furious) .. +1.0 (delighted). A noisy read on the customer's
    # true satisfaction, not a direct copy of it.
    sentiment: float = Field(ge=-1.0, le=1.0)
    csat: int | None = Field(default=None, ge=1, le=5)
    text: str


class ChurnLabel(BaseModel):
    """Forward-looking outcome, observed strictly after AS_OF_DATE.

    `window_days` is per-customer: it scales with their own purchase
    cadence so that long-cycle and short-cycle customers are judged on
    equal terms. See config.LABEL_CYCLE_MULTIPLE.
    """

    customer_id: str
    as_of: date
    window_days: int
    cycle_days: float          # the cadence the window was derived from
    orders_in_window: int
    churned: bool


# ==========================================================================
# HELD OUT -- generator-only. Never a model input.
# ==========================================================================


class LatentState(BaseModel):
    """The hidden variables that drive behaviour.

    Observables are noisy functions of these. Nothing in features/ or
    scoring/ may import this class. It is written to latent_truth.json so
    we can check, after the fact, whether the model recovered the truth it
    was never shown.
    """

    customer_id: str

    still_training: bool
    training_stopped_on: date | None = None

    brand_affinity_start: float
    brand_affinity_end: float

    price_sensitivity: float

    goal: Goal
    goal_changed_on: date | None = None
    previous_goal: Goal | None = None

    # Buy-till-you-die style parameters (see research doc sec. 1).
    latent_purchase_rate: float   # lambda: baseline orders per year while alive
    dropout_p: float              # per-transaction chance of going dormant
    true_dropout_date: date | None = None

    # Ground-truth satisfaction, which drives ticket sentiment + CSAT.
    satisfaction: float

    # True state at the scoring cutoff. The honest answer to "is this
    # customer actually gone?" -- used only for post-hoc evaluation.
    alive_at_cutoff: bool
