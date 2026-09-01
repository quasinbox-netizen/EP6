"""Zlozenie wszystkich cech w jedna ramke dzienna.

Dwie zasady, ktore przenikaja caly modul:

1. Ramka cech nie zawiera zwrotow przyszlych. Cele (`fwd_ret_*`) dokleja
   osobna funkcja, zeby przypadkowe `df.corr()` nie zmieszalo predyktorow
   z tym, co przewidujemy.
2. Builder przyjmuje `as_of` i odtwarza stan wiedzy z dowolnego dnia -
   dzieki temu test look-ahead moze porownac wersje punkt-w-czasie
   z wersja pelna (features/checks.py).
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
    """Surowe wejscia; kazde z kolumna `available_from`."""

    prices: pd.DataFrame
    macro: pd.DataFrame
    events: pd.DataFrame

    def as_of(self, day: str | pd.Timestamp | None) -> "FeatureInputs":
        """Kopia zawierajaca wylacznie dane opublikowane do dnia `day` wlacznie."""
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
    """Cechy z samej ceny. Kazde okno patrzy wylacznie wstecz."""
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
    """Buduje ramke cech dzienna dla stanu wiedzy z dnia `as_of`."""
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
    """Dokleja cele: zwrot za H dni do przodu. Ostatnie H dni to z definicji NaN."""
    horizons = horizons or DEFAULT_HORIZONS
    out = frame.copy()
    close = out[price_column]
    for horizon in horizons:
        out[f"fwd_return_{horizon}d"] = close.shift(-horizon) / close - 1.0
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Kolumny predyktorow - wszystko poza celami i surowa cena."""
    excluded = {"close"}
    return [
        c for c in frame.columns
        if not c.startswith("fwd_return_") and c not in excluded
    ]
