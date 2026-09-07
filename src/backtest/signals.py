"""Turning a position series into something a person can read.

The backtest engine answers "what would this rule have earned". This module
answers the two questions someone actually asks in front of the chart: what
does the rule say right now, and what would have to happen for it to change
its mind.

The second question is the useful one, and it is the one a signal light
cannot express. A rule that is long today tells you nothing; a rule that is
long today and flips if the next close is below $54,300 tells you where you
stand. Where a rule cannot be expressed as a price - a halving window is a
calendar, a macro phase waits for a data release - the trigger says so
instead of inventing a level.

None of this decides anything. The numbers attached to every rule here are
its own backtest, and this project's finding is that those numbers do not
survive multiple-testing correction (see validation/) or a change of
analytical choices (see analysis/specification_curve.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from features.halving import CONFIRMED_HALVINGS

IN_MARKET = 1e-12  # positions below this count as flat


@dataclass(frozen=True)
class Trigger:
    """What would flip the rule, in the units the rule actually works in."""

    kind: str  # "price", "date", "external" or "none"
    text: str
    level: float | None = None
    direction: str = ""  # "above" or "below", for price triggers
    date: pd.Timestamp | None = None

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "text": self.text,
            "level": None if self.level is None or not np.isfinite(self.level) else float(self.level),
            "direction": self.direction,
            "date": None if self.date is None else pd.Timestamp(self.date).strftime("%Y-%m-%d"),
        }


@dataclass
class SignalReport:
    name: str
    side: str  # "long" or "flat"
    since: pd.Timestamp | None
    days: int
    entry_price: float | None
    last_price: float
    unrealized: float | None
    trigger: Trigger
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    closed_trades: int = 0
    win_rate_net: float = float("nan")


def trade_log(positions: pd.Series, close: pd.Series, *, cost_rate: float = 0.0) -> pd.DataFrame:
    """Every entry and exit, with what the round trip returned.

    `positions` must be the executed series that `run_backtest` returns, not
    the raw signal: the engine has already applied the execution lag, so a
    change here is the day the trade actually happened. Feeding the unlagged
    signal in would date every trade one day early and quietly improve every
    number in the table.

    The open trade at the end of the sample is kept, marked `open`, and priced
    at the last close. Dropping it would hide the position the rule is holding
    right now, which is the one thing someone reading this cares about.
    """
    positions = positions.reindex(close.index).fillna(0.0)
    in_market = positions.abs() > IN_MARKET
    if not in_market.any():
        return pd.DataFrame(
            columns=["entry_date", "entry_price", "exit_date", "exit_price",
                     "position", "days", "return_gross", "return_net", "open"]
        )

    # A block is a maximal run of in-market days; its edges are the trades.
    block = (in_market != in_market.shift(fill_value=False)).cumsum()
    rows = []
    for _, chunk in positions[in_market].groupby(block[in_market]):
        entry_date, exit_date = chunk.index[0], chunk.index[-1]
        entry_price = float(close.loc[entry_date])
        exit_price = float(close.loc[exit_date])
        size = float(chunk.abs().mean())
        still_open = exit_date == positions.index[-1] and in_market.iloc[-1]
        gross = (exit_price / entry_price - 1.0) * size
        rows.append({
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "position": size,
            "days": int((exit_date - entry_date).days),
            "return_gross": gross,
            # Both sides of the round trip are charged, which is why a rule
            # that flips often can be right about direction and still lose.
            "return_net": gross - 2.0 * cost_rate * size,
            "open": bool(still_open),
        })
    return pd.DataFrame(rows)


def trend_flip_level(close: pd.Series, *, fast: int = 50, slow: int = 200) -> float | None:
    """The next close at which the two moving averages meet.

    Exact for one day and one day only: tomorrow's fast window is the last
    `fast - 1` known closes plus tomorrow's, so the level solves

        (fast_sum + P) / fast == (slow_sum + P) / slow

    The day after, different closes drop out of each window and the level
    moves. Quoting it as a standing line on the chart would be wrong.

    The result can come out negative, and that is information rather than an
    error: it means no positive close crosses the averages tomorrow, because
    the gap between them is wider than one day of either window can close.
    The caller has to say so; returning None here would blur that case into
    "not enough history".
    """
    if len(close) < slow:
        return None
    fast_sum = float(close.iloc[-(fast - 1):].sum())
    slow_sum = float(close.iloc[-(slow - 1):].sum())
    level = (fast * slow_sum - slow * fast_sum) / (slow - fast)
    return float(level) if np.isfinite(level) else None


def _trend_trigger(close: pd.Series, meta: dict, side: str) -> Trigger:
    """Where the next close has to land for the crossover to change sides.

    The direction comes from today's averages, not from `side`. The executed
    position is a day behind the target by construction of the engine, and
    using it here would occasionally describe the wrong flip on the day after
    a cross.
    """
    fast, slow = int(meta.get("fast", 50)), int(meta.get("slow", 200))
    if len(close) < slow:
        return Trigger("none", f"not enough history for a {slow}-day average")

    level = trend_flip_level(close, fast=fast, slow=slow)
    wants_long = float(close.iloc[-fast:].mean()) > float(close.iloc[-slow:].mean())
    if level is None or level <= 0:
        return Trigger(
            "none",
            "no close can cross the averages tomorrow - the rule keeps its "
            + ("long" if wants_long else "flat") + " target whatever the price does",
        )
    if wants_long:
        return Trigger(
            "price",
            f"flips to flat if the next close is below ${level:,.0f}",
            level=level, direction="below",
        )
    return Trigger(
        "price",
        f"flips to long if the next close is above ${level:,.0f}",
        level=level, direction="above",
    )


def _halving_trigger(last_date: pd.Timestamp, meta: dict, side: str) -> Trigger:
    days_after = int(meta.get("days_after", 365))
    if side == "long":
        starts = [h for h in CONFIRMED_HALVINGS if h <= last_date]
        if starts:
            exit_date = max(starts) + pd.Timedelta(days=days_after)
            return Trigger(
                "date",
                f"holds until {exit_date:%Y-%m-%d}, then goes flat",
                date=exit_date,
            )
    # The rule only knows halvings that have happened. The next one is a
    # prediction, and this project does not put predictions in a signal.
    return Trigger(
        "date",
        "waits for the next halving; the rule uses confirmed ones only, "
        "so it has no date to give",
    )


def next_trigger(
    meta: dict, close: pd.Series, side: str, *, last_date: pd.Timestamp | None = None
) -> Trigger:
    """What has to happen for the rule to change its position."""
    kind = (meta or {}).get("kind", "")
    last_date = last_date if last_date is not None else close.index[-1]
    if kind == "trend":
        return _trend_trigger(close, meta, side)
    if kind == "halving":
        return _halving_trigger(last_date, meta, side)
    if kind == "macro":
        return Trigger(
            "external",
            "changes with the next M2 or rate release, not with the price",
        )
    if kind == "baseline":
        return Trigger("none", "always in the market - that is the point of a baseline")
    return Trigger("none", "no trigger defined for this rule")


def report(result, close: pd.Series, *, cost_rate: float = 0.0) -> SignalReport:
    """Everything the signal desk shows for one rule."""
    positions = result.positions.reindex(close.index).fillna(0.0)
    trades = trade_log(positions, close, cost_rate=cost_rate)
    last_price = float(close.iloc[-1])
    long_now = bool(positions.iloc[-1] > IN_MARKET)
    side = "long" if long_now else "flat"

    since, entry_price, unrealized = None, None, None
    if long_now and not trades.empty and bool(trades.iloc[-1]["open"]):
        since = pd.Timestamp(trades.iloc[-1]["entry_date"])
        entry_price = float(trades.iloc[-1]["entry_price"])
        unrealized = last_price / entry_price - 1.0
    elif not long_now and not trades.empty:
        # Flat since the day after the last exit.
        since = pd.Timestamp(trades.iloc[-1]["exit_date"])

    days = int((close.index[-1] - since).days) if since is not None else int(len(close))
    # Only closed round trips count: an open position has not been paid for
    # yet, and letting it into the win rate flatters whatever is running now.
    closed = trades[~trades["open"]] if not trades.empty else trades
    win_rate = float((closed["return_net"] > 0).mean()) if len(closed) else float("nan")
    return SignalReport(
        name=getattr(result, "name", "strategy"),
        side=side,
        since=since,
        days=days,
        entry_price=entry_price,
        last_price=last_price,
        unrealized=unrealized,
        trigger=next_trigger(getattr(result, "meta", {}), close, side),
        trades=trades,
        metrics=dict(getattr(result, "metrics", {}) or {}),
        closed_trades=int(len(closed)),
        win_rate_net=win_rate,
    )
