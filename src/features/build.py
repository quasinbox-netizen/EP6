"""Assembling every feature into one daily frame.

Two rules run through this module:

1. The feature frame contains no forward returns. Targets (`fwd_return_*`) are
   attached by a separate function, so that an accidental `df.corr()` cannot
   mix predictors with the thing being predicted.
2. The builder takes `as_of` and reconstructs the state of knowledge on any
   given day - which is what lets the look-ahead test compare a point-in-time
   version against the full one (features/checks.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.events import days_since_event, event_flags
from features.halving import halving_features, halving_windows
from features.macro_phase import macro_phase

DEFAULT_EVENT_WINDOWS = [7, 30, 90]
DEFAULT_HALVING_WINDOWS = [30, 90, 180, 365]
DEFAULT_HORIZONS = [7, 30, 90, 180]


@dataclass
class FeatureInputs:
    """Raw inputs; each one carries an `available_from` column."""

    prices: pd.DataFrame
    macro: pd.DataFrame
    events: pd.DataFrame

    def as_of(self, day: str | pd.Timestamp | None) -> "FeatureInputs":
        """A copy containing only data published up to and including `day`."""
        if day is None:
            return self
        cutoff = pd.Timestamp(day).normalize()

        def trim(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            available = pd.to_datetime(df["available_from"]).dt.normalize()
            return df[available <= cutoff]

        return FeatureInputs(trim(self.prices), trim(self.macro), trim(self.events))


def price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Features derived from price alone. Every window looks strictly backwards."""
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date").set_index("date")

    close = df["close"].astype(float)
    out = pd.DataFrame(index=close.index)
    out.index.name = "date"
    out["close"] = close
    out["log_return"] = np.log(close).diff()

    for window in (7, 30, 90, 365):
        out[f"return_{window}d"] = close.pct_change(window)
    for window in (30, 90):
        out[f"volatility_{window}d"] = out["log_return"].rolling(window).std() * np.sqrt(365)
    for window in (50, 200):
        out[f"ma_{window}_ratio"] = close / close.rolling(window).mean()

    running_max = close.cummax()
    out["drawdown"] = close / running_max - 1.0
    out["days_since_ath"] = (
        close.groupby((close >= running_max).cumsum()).cumcount().astype(float)
    )
    return out


def build_features(
    inputs: FeatureInputs,
    *,
    as_of: str | pd.Timestamp | None = None,
    event_windows: list[int] | None = None,
    halving_window_list: list[int] | None = None,
    strict_halving: bool = False,
) -> pd.DataFrame:
    """Build the daily feature frame for the state of knowledge on `as_of`."""
    event_windows = event_windows or DEFAULT_EVENT_WINDOWS
    halving_window_list = halving_window_list or DEFAULT_HALVING_WINDOWS

    data = inputs.as_of(as_of)
    if data.prices.empty:
        return pd.DataFrame()

    frame = price_features(data.prices)
    index = frame.index

    frame = frame.join(halving_features(index, strict=strict_halving))
    frame = frame.join(halving_windows(index, halving_window_list, direction="after"))
    frame = frame.join(macro_phase(data.macro, index))
    frame = frame.join(event_flags(data.events, index, event_windows))
    frame = frame.join(days_since_event(data.events, index))
    return frame


def add_forward_returns(
    frame: pd.DataFrame, horizons: list[int] | None = None, price_column: str = "close"
) -> pd.DataFrame:
    """Attach targets: the return H days ahead. The last H days are NaN by definition."""
    horizons = horizons or DEFAULT_HORIZONS
    out = frame.copy()
    close = out[price_column]
    for horizon in horizons:
        out[f"fwd_return_{horizon}d"] = close.shift(-horizon) / close - 1.0
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Predictor columns - everything except targets and the raw price."""
    excluded = {"close"}
    return [
        c for c in frame.columns
        if not c.startswith("fwd_return_") and c not in excluded
    ]
