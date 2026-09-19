"""The contract every risk scorer must satisfy.

Layer 1 of the architecture ends here. A scorer takes a feature vector and
returns a calibrated probability plus per-feature attributions. It does not
know what an LLM is, and no implementation in this package may make a
network call.

The protocol exists so BTYD / survival / gradient-boosting scorers can be
dropped in later without touching the policy gate, the agent, or the
dashboard. Only logistic regression is implemented today -- see README.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field

import config as cfg


class Band(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


_BAND_ORDER = {Band.LOW: 0, Band.MEDIUM: 1, Band.HIGH: 2}


def band_for(probability: float) -> Band:
    if probability < cfg.BAND_LOW_MAX:
        return Band.LOW
    if probability < cfg.BAND_MEDIUM_MAX:
        return Band.MEDIUM
    return Band.HIGH


def band_at_least(band: Band, minimum: str) -> bool:
    return _BAND_ORDER[band] >= _BAND_ORDER[Band(minimum)]


class Attribution(BaseModel):
    """One feature's contribution to this customer's log-odds.

    `contribution` is signed and in log-odds units: positive pushes toward
    churn. For logistic regression it is exact, not an approximation.
    """

    feature: str
    label: str
    value: float
    contribution: float

    @property
    def direction(self) -> str:
        return "increases risk" if self.contribution > 0 else "decreases risk"


class RiskResult(BaseModel):
    customer_id: str
    probability: float = Field(ge=0.0, le=1.0)
    band: Band
    attributions: list[Attribution]
    model_name: str

    def top(self, n: int = 5) -> list[Attribution]:
        return sorted(self.attributions, key=lambda a: -abs(a.contribution))[:n]


class FitReport(BaseModel):
    """Honest evaluation numbers. Reported on SYNTHETIC data -- see README."""

    model_name: str
    n_train: int
    n_test: int
    base_rate: float
    roc_auc: float
    pr_auc: float
    brier: float
    precision_at_k: float
    k: int
    lift_at_k: float
    top_coefficients: list[tuple[str, float]]

    def summary(self) -> str:
        return (
            f"{self.model_name}: PR-AUC {self.pr_auc:.3f} | ROC-AUC {self.roc_auc:.3f} | "
            f"Brier {self.brier:.3f} | precision@{self.k} {self.precision_at_k:.3f} "
            f"({self.lift_at_k:.2f}x base rate of {self.base_rate:.3f})"
        )


class RiskScorer(ABC):
    """Implement this to add a new scoring model."""

    name: str = "abstract"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> FitReport:
        ...

    @abstractmethod
    def score(self, customer_id: str, features: dict[str, float]) -> RiskResult:
        ...

    def score_batch(self, X: pd.DataFrame) -> list[RiskResult]:
        return [self.score(str(cid), row.to_dict()) for cid, row in X.iterrows()]
