"""Risk trajectory -- is this customer drifting toward the exit, or parked?

Layer 1, same as features/extract.py: arithmetic you could do in a
spreadsheet, no LLM anywhere.

WHAT THIS ANSWERS THAT THE RISK SCORE DOES NOT
    The score is a snapshot. Two customers at 0.55 are not the same
    customer if one has sat at 0.55 since January and the other was at
    0.25 in the spring. The first is a stable, slightly-lapsed buyer.
    The second is falling off a cliff and has not landed yet. Identical
    scores, opposite situations, opposite correct responses.

HOW IT WORKS
    Rebuild each customer's feature vector at a grid of earlier dates and
    run the SAME already-fitted model over each one. That gives a risk
    curve. The recency-weighted slope of that curve is `momentum`, in risk
    percentage points per 100 days.

    No second model. No new features fed to the scorer. That is deliberate
    and it is not laziness -- adding trend features to the model measurably
    makes it WORSE (PR-AUC 0.755 -> 0.742), because 94 churn events cannot
    support 13 predictors. Same events-per-variable argument that keeps
    MODEL_FEATURES at six. See SCORING.md.

WHY REWINDING IS NOT LEAKAGE
    store.events_for(cid, as_of) truncates the event streams before
    handing them over, so a rewound snapshot physically cannot see past
    its own cutoff -- the same firewall that protects the live features.
    This module reads no labels and no latent state; smoke_test.py parses
    its AST to enforce that.

COMMON-MODE DRIFT, AND WHY MOMENTUM IS CENTRED
    Rewinding has one honest problem. order_count_lifetime is cumulative,
    so a snapshot from 180 days ago necessarily shows fewer orders --
    cohort mean 0.90 then against 2.96 now. Its coefficient is negative,
    so every rewound snapshot is scored as riskier than it deserves and
    EVERY curve slopes gently downward. On the committed seed the cohort
    median slope is -4.7 points per 100 days.

    That is an artefact of looking backwards, not a cohort that is
    collectively recovering, and reading it as good news would be wrong.
    Because every customer is measured on the same calendar grid, the bias
    is common-mode and subtracting the cohort median removes it exactly.
    So `momentum` is the raw slope -- what actually happened to their
    number -- and `relative_momentum` is that slope against the cohort,
    which is what the state is classified on.

COLD STARTS AT THE LEFT EDGE
    A customer who had one order 180 days ago genuinely looked like a
    maximal risk at the time, and the curve says so -- 2 of 200 open at
    >95% on the committed seed. That is the model being honest about what
    it could see, not a bug, but it makes the oldest point a bad thing to
    quote in a headline. summary() deliberately reports the slope and
    today's risk instead of a then-and-now pair.

WHAT THE CURVE LOOKS LIKE, AND WHY IT IS BUMPY
    Healthy reorderers produce a SAWTOOTH: reorder_gap_ratio climbs as the
    tub empties, then drops to zero the day the next order lands. That is
    the signature of a working subscription, not of instability. A
    customer whose sawtooth flattens into a straight climb has stopped
    completing the cycle. That is what momentum is measuring, and it is
    why the slope is recency-weighted rather than a plain first-to-last
    difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import config as cfg
from features.extract import FEATURE_LABELS, extract_frame

if TYPE_CHECKING:  # avoids a features <-> scoring import cycle at runtime
    from data.store import EventStore
    from scoring.base import RiskScorer


class TrajectoryState(str, Enum):
    CLIMBING = "CLIMBING"      # risk rising -- the pattern past churners showed
    STABLE = "STABLE"          # flat, wherever it is flat
    RECOVERING = "RECOVERING"  # risk falling -- they are coming back


# Signals whose direction of travel we report in the UI. One per family, and
# only ones a person can act on: "they stopped clicking" is a sentence a
# retention manager can do something with, "promo_order_share fell 0.04" is
# not.
#
# +1 means an INCREASE is bad news. -1 means a DECREASE is bad news.
DRIFT_SIGNALS: dict[str, int] = {
    "reorder_gap_ratio": +1,
    "days_since_last_session": +1,
    "click_rate_90d": -1,
    "sessions_30d": -1,
    "open_rate_90d": -1,
}


@dataclass(frozen=True)
class SignalDrift:
    """One tracked signal's direction of travel over the lookback."""

    feature: str
    label: str
    earliest: float
    latest: float
    z_change: float   # movement in cohort standard deviations
    worsening: bool


@dataclass(frozen=True)
class Trajectory:
    """One customer's risk curve and what it is doing."""

    customer_id: str
    offsets: tuple[int, ...]          # days before as_of, oldest first
    curve: tuple[float, ...]          # risk probability at each offset
    momentum: float                   # raw slope, risk points per 100 days
    relative_momentum: float          # the same slope, cohort median removed
    state: TrajectoryState
    drifts: tuple[SignalDrift, ...]   # only signals that actually moved

    @property
    def risk_now(self) -> float:
        return self.curve[-1]

    @property
    def risk_then(self) -> float:
        return self.curve[0]

    @property
    def change(self) -> float:
        """Total change in risk over the lookback, in percentage points."""
        return (self.risk_now - self.risk_then) * 100.0

    @property
    def worsening_signals(self) -> list[SignalDrift]:
        return [d for d in self.drifts if d.worsening]

    def summary(self) -> str:
        """One line for the CLI and the card. Plain language on purpose.

        Quotes the slope and today's risk, NOT a then-and-now pair. The
        oldest point on the curve is the least trustworthy one -- see the
        cold-start note in the module docstring -- and putting it in a
        headline invites reading "100% -> 36%" as a recovery when it
        really means the customer had barely any history that far back.
        The chart shows the whole curve; the reader can see the shape.
        """
        if self.state is TrajectoryState.CLIMBING:
            return (f"Risk climbing {self.relative_momentum:+.0f} pts/100d "
                    f"faster than the cohort, now {self.risk_now:.0%}")
        if self.state is TrajectoryState.RECOVERING:
            return (f"Risk easing {abs(self.relative_momentum):.0f} pts/100d "
                    f"faster than the cohort, now {self.risk_now:.0%}")
        return f"Risk steady near {self.risk_now:.0%}"


# ==========================================================================
# the maths
# ==========================================================================


def _grid() -> list[int]:
    """Day offsets back from the as-of date, oldest first, always ending at 0."""
    offsets = list(range(cfg.TRAJECTORY_LOOKBACK_DAYS, -1, -cfg.TRAJECTORY_STEP_DAYS))
    if offsets[-1] != 0:
        offsets.append(0)
    return offsets


def _weights(offsets: np.ndarray) -> np.ndarray:
    """Exponential recency weights. A point HALFLIFE days older counts half.

    Without this, a reorder eighteen weeks ago moves the slope as much as
    one last week, and the number stops describing what the customer is
    doing now.
    """
    age = offsets.astype(float)
    return np.exp(-np.log(2.0) * age / cfg.TRAJECTORY_HALFLIFE_DAYS)


def momentum_of(curve: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Recency-weighted least-squares slope, in risk points per 100 days.

    curve: (n_customers, n_points) probabilities, oldest point first.
    """
    x = -offsets.astype(float)          # time axis, increasing toward today
    w = _weights(offsets)
    x_bar = (x * w).sum() / w.sum()
    y_bar = (curve * w).sum(axis=1) / w.sum()
    num = ((curve - y_bar[:, None]) * w * (x - x_bar)).sum(axis=1)
    den = (w * (x - x_bar) ** 2).sum()
    return num / den * 100.0 * 100.0    # prob/day -> risk POINTS per 100 days


def state_for(relative_momentum: float) -> TrajectoryState:
    """Classify on the COHORT-CENTRED slope, not the raw one."""
    if relative_momentum >= cfg.TRAJECTORY_CLIMBING_MIN:
        return TrajectoryState.CLIMBING
    if relative_momentum <= cfg.TRAJECTORY_RECOVERING_MAX:
        return TrajectoryState.RECOVERING
    return TrajectoryState.STABLE


# ==========================================================================
# entry point
# ==========================================================================


def build_curves(
    store: EventStore,
    scorer: RiskScorer,
    as_of: date | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Risk curve per customer, plus the raw signal frames behind it.

    Returns (curves indexed by customer_id with one column per offset,
    {offset: feature frame}). The scorer must already be fitted.
    """
    as_of = as_of or cfg.AS_OF_DATE
    offsets = _grid()

    frames = {
        off: extract_frame(store, as_of=as_of - timedelta(days=off)) for off in offsets
    }
    curves = pd.DataFrame(
        {off: scorer.probabilities(frames[off]) for off in offsets},
        index=frames[0].index,
    )[offsets]
    return curves, frames


def _drifts_for(
    cid: str, frames: dict[int, pd.DataFrame], offsets: list[int], spreads: dict[str, float]
) -> tuple[SignalDrift, ...]:
    """Which tracked signals moved, and was the movement bad news?"""
    out = []
    for feature, bad_direction in DRIFT_SIGNALS.items():
        earliest = float(frames[offsets[0]].at[cid, feature])
        latest = float(frames[offsets[-1]].at[cid, feature])
        if not (np.isfinite(earliest) and np.isfinite(latest)):
            continue
        change = latest - earliest
        z = change / spreads[feature] if spreads[feature] > 0 else 0.0
        if abs(z) < cfg.TRAJECTORY_DRIFT_MIN_Z:
            continue
        out.append(SignalDrift(
            feature=feature,
            label=FEATURE_LABELS.get(feature, feature),
            earliest=earliest, latest=latest, z_change=z,
            worsening=(change * bad_direction) > 0,
        ))
    # Biggest mover first -- that is the one worth a sentence in the UI.
    return tuple(sorted(out, key=lambda d: -abs(d.z_change)))


def build(
    store: EventStore,
    scorer: RiskScorer,
    as_of: date | None = None,
) -> dict[str, Trajectory]:
    """Trajectory for every customer in the cohort, keyed by customer_id."""
    curves, frames = build_curves(store, scorer, as_of)
    offsets = list(curves.columns)
    off_arr = np.asarray(offsets, dtype=float)
    moms = momentum_of(curves.to_numpy(dtype=float), off_arr)
    relative = moms - float(np.median(moms))   # see "common-mode drift" above

    # Cohort spread per signal, measured on today's frame, so "moved a lot"
    # means a lot RELATIVE TO THE COHORT rather than relative to its units.
    spreads = {
        f: float(np.nanstd(frames[offsets[-1]][f].to_numpy(dtype=float)))
        for f in DRIFT_SIGNALS
    }

    out: dict[str, Trajectory] = {}
    for i, cid in enumerate(curves.index):
        cid = str(cid)
        out[cid] = Trajectory(
            customer_id=cid,
            offsets=tuple(offsets),
            curve=tuple(float(v) for v in curves.iloc[i]),
            momentum=float(moms[i]),
            relative_momentum=float(relative[i]),
            state=state_for(float(relative[i])),
            drifts=_drifts_for(cid, frames, offsets, spreads),
        )
    return out
