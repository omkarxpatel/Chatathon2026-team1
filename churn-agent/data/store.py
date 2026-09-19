"""Loads the generated parquet files and slices them per customer.

This is the only place that touches the filesystem for event data, and it
is where the temporal firewall is enforced: `events_for(cid, as_of)` hands
back events strictly BEFORE as_of. A feature function physically cannot see
the future, because it is never given it.

Note what is absent: there is no loader for latent_truth.json here. The
audit helper for that lives in evaluate.py, outside the scoring path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

import config as cfg


@dataclass(frozen=True)
class CustomerEvents:
    """One customer's observable history, already truncated at `as_of`."""

    customer_id: str
    as_of: date
    signup_date: date
    orders: pd.DataFrame
    order_lines: pd.DataFrame
    sessions: pd.DataFrame
    messages: pd.DataFrame
    tickets: pd.DataFrame

    @property
    def fulfilled_orders(self) -> pd.DataFrame:
        return self.orders[self.orders["status"] == "fulfilled"]


class EventStore:
    def __init__(
        self,
        customers: pd.DataFrame,
        orders: pd.DataFrame,
        order_lines: pd.DataFrame,
        sessions: pd.DataFrame,
        messages: pd.DataFrame,
        tickets: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> None:
        self.customers = customers.set_index("customer_id", drop=False)
        self.orders = orders
        self.order_lines = order_lines
        self.sessions = sessions
        self.messages = messages
        self.tickets = tickets
        self.labels = labels.set_index("customer_id", drop=False)

        # Pre-group once; per-customer slicing then costs nothing.
        self._orders_by = dict(tuple(orders.groupby("customer_id"))) if len(orders) else {}
        self._lines_by = dict(tuple(order_lines.groupby("customer_id"))) if len(order_lines) else {}
        self._sessions_by = dict(tuple(sessions.groupby("customer_id"))) if len(sessions) else {}
        self._messages_by = dict(tuple(messages.groupby("customer_id"))) if len(messages) else {}
        self._tickets_by = dict(tuple(tickets.groupby("customer_id"))) if len(tickets) else {}

    # ------------------------------------------------------------------
    @classmethod
    def load(cls) -> EventStore:
        missing = [
            p.name for p in (
                cfg.CUSTOMERS_FILE, cfg.ORDERS_FILE, cfg.ORDER_LINES_FILE,
                cfg.SESSIONS_FILE, cfg.MESSAGES_FILE, cfg.TICKETS_FILE, cfg.LABELS_FILE,
            ) if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing generated data: {missing}. Run `python run_generate.py` first."
            )
        return cls(
            customers=pd.read_parquet(cfg.CUSTOMERS_FILE),
            orders=pd.read_parquet(cfg.ORDERS_FILE),
            order_lines=pd.read_parquet(cfg.ORDER_LINES_FILE),
            sessions=pd.read_parquet(cfg.SESSIONS_FILE),
            messages=pd.read_parquet(cfg.MESSAGES_FILE),
            tickets=pd.read_parquet(cfg.TICKETS_FILE),
            labels=pd.read_parquet(cfg.LABELS_FILE),
        )

    # ------------------------------------------------------------------
    @property
    def customer_ids(self) -> list[str]:
        return list(self.customers["customer_id"])

    def archetype(self, customer_id: str) -> str | None:
        value = self.customers.loc[customer_id, "archetype"]
        return None if pd.isna(value) else str(value)

    def label(self, customer_id: str) -> bool:
        return bool(self.labels.loc[customer_id, "churned"])

    def label_series(self) -> pd.Series:
        return self.labels["churned"].astype(bool)

    # ------------------------------------------------------------------
    @staticmethod
    def _before(df: pd.DataFrame | None, col: str, cutoff: datetime) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return df if df is not None else pd.DataFrame()
        return df[df[col] < cutoff].sort_values(col)

    def events_for(self, customer_id: str, as_of: date | None = None) -> CustomerEvents:
        """Everything observable about one customer, strictly before `as_of`."""
        as_of = as_of or cfg.AS_OF_DATE
        cutoff = pd.Timestamp(as_of)

        orders = self._before(self._orders_by.get(customer_id), "ts", cutoff)
        lines = self._lines_by.get(customer_id, pd.DataFrame(columns=self.order_lines.columns))
        if len(orders) and len(lines):
            lines = lines[lines["order_id"].isin(set(orders["order_id"]))]

        signup = self.customers.loc[customer_id, "signup_date"]
        return CustomerEvents(
            customer_id=customer_id,
            as_of=as_of,
            signup_date=pd.Timestamp(signup).date(),
            orders=orders,
            order_lines=lines,
            sessions=self._before(self._sessions_by.get(customer_id), "ts", cutoff),
            messages=self._before(self._messages_by.get(customer_id), "ts", cutoff),
            tickets=self._before(self._tickets_by.get(customer_id), "created_ts", cutoff),
        )
