"""Strategies - each one a function returning the TARGET position for a day.

None of them shifts the signal in time; the execution lag is applied by the
engine (engine.run_backtest). Double-shifting is as wrong as not shifting.

The strategies are deliberately primitive. These are not investment proposals,
they are a way of checking whether a pattern found in the event study survives
contact with transaction costs and with the buy-and-hold baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.halving import CONFIRMED_HALVINGS


def buy_and_hold(index: pd.DatetimeIndex) -> pd.Series:
    """The baseline. Every strategy has to be compared against it."""
    return pd.Series(1.0, index=index, name="buy_and_hold")


def from_mask(mask: pd.Series, *, position: float = 1.0) -> pd.Series:
    """Hold `position` while the mask is true, nothing outside it."""
    return (mask.astype(float) * position).rename("mask_strategy")


def halving_window(
    index: pd.DatetimeIndex, *, days_after: int = 180, position: float = 1.0
) -> pd.Series:
    """In the market for N days after a halving, out of it otherwise.

    Uses only confirmed past halvings - on day t you know how many days have
    passed since the last halving, and nothing more.
    """
    index = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    flag = np.zeros(len(index))
    for halving in CONFIRMED_HALVINGS:
        window = (index >= halving) & (index <= halving + pd.Timedelta(days=days_after))
        flag = np.maximum(flag, window.astype(float))
    return pd.Series(flag * position, index=index, name=f"halving_{days_after}d")


def macro_regime(
    regime: pd.Series, *, long_labels: tuple[str, ...] = ("expanding_falling",)
) -> pd.Series:
    """Hold a position only in selected macro phases.

    Default: rising liquidity with falling rates. That is a hypothesis to be
    checked, not knowledge - the sample contains four cycles, so every phase
    has only a handful of independent episodes.
    """
    return regime.isin(long_labels).astype(float).rename("macro_regime")


def trend_following(
    close: pd.Series, *, fast: int = 50, slow: int = 200, position: float = 1.0
) -> pd.Series:
    """A classic trend filter - an honest yardstick for cyclical strategies.

    If the halving hypothesis cannot beat a moving-average crossover, then it is
    measuring trend rather than the halving cycle.
    """
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(float) * position
    signal[slow_ma.isna()] = 0.0
    return signal.rename(f"trend_{fast}_{slow}")


def combine(*signals: pd.Series, mode: str = "all") -> pd.Series:
    """Combine signals: `all` = every condition, `any` = logical or."""
    if not signals:
        raise ValueError("no signals to combine")
    frame = pd.concat(signals, axis=1).fillna(0.0)
    if mode == "all":
        combined = frame.min(axis=1)
    elif mode == "any":
        combined = frame.max(axis=1)
    else:
        raise ValueError("mode must be 'all' or 'any'")
    return combined.rename(f"combined_{mode}")
