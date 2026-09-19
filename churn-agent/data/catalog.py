"""Product catalog for a DTC sports nutrition brand.

The point of this file: every SKU has a *computable depletion date*. A
30-serving tub at 1 scoop/day runs out in 30 days. That turns "is this
customer lapsing?" from an arbitrary statistical threshold into a physical
question -- have they run out of the thing they were buying?

Different formats of the same product have wildly different cadences, which
is exactly what makes naive recency scoring fail (see the CUST-0002 demo
archetype: a long gap that means nothing because they bought a 5 lb tub).
"""

from __future__ import annotations

from pydantic import BaseModel

from data.schema import Goal


class Product(BaseModel):
    sku: str
    name: str
    category: str
    servings: int
    serving_size: str
    servings_per_day: float  # typical usage rate for a consistent trainee
    price: float
    margin: float            # gross margin fraction

    @property
    def expected_days_supply(self) -> float:
        """How long one unit lasts at typical usage. The whole point."""
        return self.servings / self.servings_per_day

    @property
    def unit_margin(self) -> float:
        return self.price * self.margin


CATALOG: dict[str, Product] = {
    p.sku: p
    for p in [
        Product(
            sku="WHEY-1LB", name="Whey Isolate, 1 lb", category="protein",
            servings=15, serving_size="30 g", servings_per_day=1.0,
            price=24.99, margin=0.38,
        ),
        Product(
            sku="WHEY-2LB", name="Whey Isolate, 2 lb", category="protein",
            servings=30, serving_size="30 g", servings_per_day=1.0,
            price=42.99, margin=0.42,
        ),
        Product(
            sku="WHEY-5LB", name="Whey Isolate, 5 lb", category="protein",
            servings=76, serving_size="30 g", servings_per_day=1.0,
            price=89.99, margin=0.45,
        ),
        Product(
            sku="CREA-350", name="Creatine Monohydrate, 350 g", category="creatine",
            servings=70, serving_size="5 g", servings_per_day=1.0,
            price=27.99, margin=0.55,
        ),
        Product(
            sku="PRE-25", name="Pre-Workout, 25 servings", category="preworkout",
            servings=25, serving_size="12 g", servings_per_day=1.0,
            price=34.99, margin=0.48,
        ),
        Product(
            sku="ELEC-30", name="Electrolyte Sticks, 30 ct", category="electrolytes",
            servings=30, serving_size="1 stick", servings_per_day=1.5,
            price=24.99, margin=0.50,
        ),
        Product(
            sku="MULTI-60", name="Daily Multivitamin, 60 ct", category="vitamins",
            servings=60, serving_size="2 capsules", servings_per_day=2.0,
            price=19.99, margin=0.52,
        ),
    ]
}

# Resulting expected_days_supply, for quick reference:
#   WHEY-1LB  15 | WHEY-2LB  30 | WHEY-5LB  76 | CREA-350 70
#   PRE-25    25 | ELEC-30   20 | MULTI-60  30

SKUS = list(CATALOG)

# Which SKUs a customer reaches for, by training goal. Weights are relative
# pick probabilities, not shares of wallet.
GOAL_AFFINITY: dict[Goal, dict[str, float]] = {
    Goal.BULKING: {
        "WHEY-5LB": 0.30, "WHEY-2LB": 0.25, "CREA-350": 0.22,
        "PRE-25": 0.15, "MULTI-60": 0.05, "ELEC-30": 0.03,
    },
    Goal.CUTTING: {
        "WHEY-1LB": 0.22, "WHEY-2LB": 0.24, "ELEC-30": 0.20,
        "MULTI-60": 0.16, "PRE-25": 0.15, "CREA-350": 0.03,
    },
    Goal.MAINTENANCE: {
        "MULTI-60": 0.30, "WHEY-2LB": 0.25, "CREA-350": 0.20,
        "WHEY-1LB": 0.15, "ELEC-30": 0.10,
    },
}


def get(sku: str) -> Product:
    return CATALOG[sku]


def expected_days_supply(sku: str, quantity: int = 1) -> float:
    return CATALOG[sku].expected_days_supply * quantity


def basket_days_supply(lines: list[tuple[str, int]]) -> float:
    """How long a whole basket lasts.

    Defined as the longest-lasting item in the order: that is the date the
    customer has a reason to come back. Buying a 2 lb whey (30 days) plus
    electrolytes (20 days) does not mean they reorder at day 20 -- they
    reorder when the anchor product runs out.
    """
    if not lines:
        return 0.0
    return max(expected_days_supply(sku, qty) for sku, qty in lines)
