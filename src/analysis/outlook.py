"""Where the cycle is now, and what the sample says about the next N days.

This is the descriptive half of the front page. It answers "what usually
happened from a day that looked like today" - not "what will happen". The
distinction is the whole module:

* every condition is backward-looking (days since a confirmed halving, price
  against its own 200-day average), so a day can be classified without
  knowing anything that came after it;
* the answer is a distribution, never a point;
* and it always arrives next to the unconditional distribution, because the
  only interesting question is whether conditioning changed anything.

The number that decides whether any of it is quotable is `effective_n`.
Matched days are not observations: 400 consecutive days share almost all of
their 90-day outcome, and treating them as 400 draws is how a coin flip turns
into a certainty. Only non-overlapping windows are counted, and with the
sample this project has, that count is usually small enough to embarrass the
percentiles it produces. Saying so is the point.

The calibrated price interval lives elsewhere - forecast/coverage.py builds it
and tests whether it kept its promise. Nothing here is a substitute for that;
this module supplies context, not intervals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from features.halving import CONFIRMED_HALVINGS

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

# Below this many non-overlapping windows the percentiles are noise dressed as
# numbers. The caller is told rather than silently handed them.
MIN_EFFECTIVE_N = 8

CYCLE_BUCKETS = (
    (0, 365, "first year after the halving"),
    (366, 730, "second year after the halving"),
    (731, 1095, "third year after the halving"),
    (1096, 10_000, "fourth year or later"),
)


@dataclass(frozen=True)
class BaseRate:
    """The distribution of forward returns over a set of matched days."""

    horizon: int
    label: str
    quantiles: dict
    share_positive: float
    matched_days: int
    effective_n: int
    # The matched forward returns themselves, so a caller can ask questions the
    # quantiles do not answer ("how often was it up at least 20%") without
    # re-deriving the match in the view layer, where it would drift.
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def share_above(self, threshold: float) -> float:
        """How often the forward return reached at least `threshold`."""
        if self.returns.empty:
            return float("nan")
        return float((self.returns >= threshold).mean())

    @property
    def usable(self) -> bool:
        return self.effective_n >= MIN_EFFECTIVE_N

    @property
    def median(self) -> float:
        return self.quantiles.get(0.50, float("nan"))

    def summary(self) -> str:
        state = "" if self.usable else " (too few independent windows to quote)"
        return (
            f"{self.label}: median {self.median:+.1%}, "
            f"{self.share_positive:.0%} positive, "
            f"{self.effective_n} independent windows{state}"
        )


@dataclass
class CycleOutlook:
    as_of: pd.Timestamp
    last_price: float
    days_since_halving: int
    cycle_label: str
    trend_label: str
    rates: list = field(default_factory=list)  # (conditional, unconditional) pairs
    paths: dict = field(default_factory=dict)


def forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Return from each day to `horizon` calendar rows later.

    The last `horizon` days have no outcome yet and drop out. Keeping them as
    zeros - which an early draft did through a careless fillna - pulls the
    median toward nothing and quietly shrinks every interval.
    """
    close = close.astype(float).sort_index()
    forward = close.shift(-horizon) / close - 1.0
    return forward.dropna()


def independent_count(dates: pd.DatetimeIndex, horizon: int) -> int:
    """How many of these days start windows that share no outcome.

    Greedy from the earliest: take a day, skip everything inside its horizon,
    take the next. This is a count of observations; `len(dates)` is a count of
    rows, and the two differ by a factor of `horizon` when the matched days
    are contiguous.
    """
    if len(dates) == 0:
        return 0
    ordered = pd.DatetimeIndex(dates).sort_values()
    kept, cutoff = 0, None
    for day in ordered:
        if cutoff is None or day >= cutoff:
            kept += 1
            cutoff = day + pd.Timedelta(days=horizon)
    return kept


def _bucket(days_since: float) -> str:
    for low, high, label in CYCLE_BUCKETS:
        if low <= days_since <= high:
            return label
    return "outside the cycle table"


def current_state(features: pd.DataFrame) -> dict:
    """The labels that describe today, using only what today knows."""
    last = features.iloc[-1]
    days_since = float(last.get("days_since_halving", float("nan")))
    above = float(last.get("ma_200_ratio", float("nan")))
    return {
        "as_of": features.index[-1],
        "days_since_halving": int(days_since) if np.isfinite(days_since) else -1,
        "cycle_label": _bucket(days_since) if np.isfinite(days_since) else "unknown",
        "trend_label": (
            "above its 200-day average" if np.isfinite(above) and above > 1
            else "below its 200-day average" if np.isfinite(above)
            else "trend unknown"
        ),
        "above_200": bool(np.isfinite(above) and above > 1),
    }


def _distribution(forward: pd.Series, dates, horizon: int, label: str) -> BaseRate:
    matched = forward.reindex(pd.DatetimeIndex(dates)).dropna()
    if matched.empty:
        return BaseRate(horizon, label, {q: float("nan") for q in QUANTILES},
                        float("nan"), 0, 0)
    return BaseRate(
        horizon=horizon,
        label=label,
        quantiles={q: float(matched.quantile(q)) for q in QUANTILES},
        share_positive=float((matched > 0).mean()),
        matched_days=int(len(matched)),
        effective_n=independent_count(matched.index, horizon),
        returns=matched,
    )


def base_rates(features: pd.DataFrame, horizon: int) -> tuple[BaseRate, BaseRate]:
    """Forward returns from days like today, and from every day.

    Returns (conditional, unconditional). Reading only the first one is the
    mistake this pairing exists to prevent: a conditional median means nothing
    until you know what the unconditional one was.
    """
    close = features["close"]
    forward = forward_returns(close, horizon)
    state = current_state(features)

    days_since = features.get("days_since_halving")
    above = features.get("ma_200_ratio")
    if days_since is None or above is None:
        whole = _distribution(forward, forward.index, horizon, "every day in the sample")
        return whole, whole

    same_phase = days_since.apply(_bucket) == state["cycle_label"]
    same_trend = (above > 1) == state["above_200"]
    matched = features.index[same_phase.fillna(False) & same_trend.fillna(False)]

    conditional = _distribution(
        forward, matched, horizon,
        f"days in the {state['cycle_label']}, price {state['trend_label']}",
    )
    unconditional = _distribution(
        forward, forward.index, horizon, "every day in the sample",
    )
    return conditional, unconditional


def cycle_paths(close: pd.Series, *, span_days: int = 1461) -> dict:
    """Each halving cycle as a price path normalised to 1.0 on its halving day.

    The point of drawing these together is not the shape they share - it is
    how far apart they are. Four paths that a reader can see diverging say
    more about n=4 than any confidence interval printed underneath.
    """
    close = close.astype(float).sort_index()
    paths = {}
    for halving in CONFIRMED_HALVINGS:
        if halving < close.index[0] or halving > close.index[-1]:
            continue
        window = close.loc[halving:halving + pd.Timedelta(days=span_days)]
        if window.empty:
            continue
        anchor = float(window.iloc[0])
        if anchor <= 0:
            continue
        normalised = window / anchor
        normalised.index = [(day - halving).days for day in window.index]
        paths[halving.strftime("%Y-%m-%d")] = normalised
    return paths
