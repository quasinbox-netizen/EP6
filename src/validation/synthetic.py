"""Synthetic data generators - the proving ground for tests and calibration.

They serve two purposes:

1. Negative tests: on pure noise no method may find patterns more often than
   the significance level allows.
2. Power tests: when we INJECT a known effect, the method must find it -
   otherwise the absence of a result on real data means nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk_prices(
    n_days: int = 2000,
    *,
    start: str = "2013-01-01",
    seed: int = 0,
    mu: float = 0.0,
    sigma: float = 0.04,
    initial: float = 100.0,
) -> pd.DataFrame:
    """OHLCV bars from a geometric random walk - no pattern of any kind."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(mu, sigma, n_days)
    close = initial * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate([[initial], close[:-1]])
    noise = np.abs(rng.normal(0, sigma / 2, n_days))
    high = np.maximum(open_, close) * (1 + noise)
    low = np.minimum(open_, close) * (1 - noise)
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.lognormal(10, 1, n_days),
        }
    )


def inject_drift(
    prices: pd.DataFrame,
    anchor_dates: list[str] | pd.DatetimeIndex,
    *,
    window: int = 30,
    daily_drift: float = 0.004,
) -> pd.DataFrame:
    """Add constant drift in the `window` days following each anchor date.

    Used in power tests: a known, injected effect has to be detected.
    """
    out = prices.copy()
    dates = pd.to_datetime(pd.Index(anchor_dates))
    # pandas returns read-only views, hence copy=True: we mutate this array.
    log_returns = np.log(out["close"]).diff().fillna(0.0).to_numpy(copy=True)
    day_index = pd.DatetimeIndex(out["date"])
    for anchor in dates:
        mask = (day_index > anchor) & (day_index <= anchor + pd.Timedelta(days=window))
        log_returns[np.asarray(mask)] += daily_drift
    close = out["close"].iloc[0] * np.exp(np.cumsum(log_returns))
    scale = close / out["close"].to_numpy()
    for column in ("open", "high", "low", "close"):
        out[column] = out[column].to_numpy() * scale
    return out


def block_bootstrap(
    values: np.ndarray, *, block_length: int = 30, size: int | None = None, rng=None
) -> np.ndarray:
    """Block bootstrap - preserves autocorrelation and volatility clustering.

    A plain i.i.d. bootstrap overstates significance on financial series
    because it assumes days are independent, which prices are not.
    """
    rng = rng or np.random.default_rng()
    values = np.asarray(values)
    n = len(values)
    size = size or n
    if n == 0:
        return np.array([])
    block_length = max(1, min(block_length, n))
    n_blocks = int(np.ceil(size / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    # Circular bootstrap: blocks may wrap, so every day gets an equal chance.
    indices = (starts[:, None] + np.arange(block_length)[None, :]) % n
    return values[indices.reshape(-1)[:size]]
