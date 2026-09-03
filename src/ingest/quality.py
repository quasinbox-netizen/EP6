"""Quality control on raw data.

The report is a data structure, not a print - that way the same functions feed
the tests, the CLI and the dashboard. None of the checks modifies the data;
fixing it is a deliberate decision by the user, not a side effect of importing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    name: str
    rows: int
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None
    missing_days: list[pd.Timestamp] = field(default_factory=list)
    duplicate_dates: list[pd.Timestamp] = field(default_factory=list)
    non_positive: list[pd.Timestamp] = field(default_factory=list)
    ohlc_violations: list[pd.Timestamp] = field(default_factory=list)
    return_outliers: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    stale_runs: list[tuple[pd.Timestamp, int]] = field(default_factory=list)

    @property
    def problems(self) -> dict[str, int]:
        return {
            "missing_days": len(self.missing_days),
            "duplicate_dates": len(self.duplicate_dates),
            "non_positive": len(self.non_positive),
            "ohlc_violations": len(self.ohlc_violations),
            "return_outliers": len(self.return_outliers),
            "stale_runs": len(self.stale_runs),
        }

    @property
    def is_clean(self) -> bool:
        blocking = ("duplicate_dates", "non_positive", "ohlc_violations")
        return all(self.problems[k] == 0 for k in blocking)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "first_date": None if self.first_date is None else str(self.first_date.date()),
            "last_date": None if self.last_date is None else str(self.last_date.date()),
            **self.problems,
            "is_clean": self.is_clean,
        }

    def summary(self) -> str:
        parts = [f"{self.name}: {self.rows} rows"]
        if self.first_date is not None:
            parts.append(f"{self.first_date.date()} -> {self.last_date.date()}")
        parts.extend(f"{k}={v}" for k, v in self.problems.items() if v)
        return " | ".join(parts)


def check_prices(
    df: pd.DataFrame,
    *,
    name: str = "prices",
    outlier_sigma: float = 8.0,
    max_stale_days: int = 3,
) -> QualityReport:
    """Check for missing days, duplicates and outliers in OHLCV bars.

    The outlier threshold uses the median absolute deviation (MAD) rather than
    the standard deviation - a single 40% crash must not raise the threshold
    enough to hide itself.
    """
    if df.empty:
        return QualityReport(name=name, rows=0, first_date=None, last_date=None)

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data = data.sort_values("date")

    duplicates = data.loc[data.duplicated(subset="date", keep=False), "date"]
    unique = data.drop_duplicates(subset="date", keep="last").set_index("date")

    full_index = pd.date_range(unique.index.min(), unique.index.max(), freq="D")
    missing = full_index.difference(unique.index)

    price_cols = [c for c in ("open", "high", "low", "close") if c in unique.columns]
    values = unique[price_cols]
    non_positive = unique.index[(values <= 0).any(axis=1) | values.isna().any(axis=1)]

    if {"open", "high", "low", "close"} <= set(unique.columns):
        bad_ohlc = (
            (unique["high"] < unique["low"])
            | (unique["high"] < unique[["open", "close"]].max(axis=1))
            | (unique["low"] > unique[["open", "close"]].min(axis=1))
        )
        ohlc_violations = unique.index[bad_ohlc]
    else:
        ohlc_violations = pd.DatetimeIndex([])

    # Non-positive values are reported separately; here they must simply not
    # break the logarithm.
    safe_close = unique["close"].where(unique["close"] > 0)
    log_return = np.log(safe_close).diff()
    median = log_return.median()
    mad = (log_return - median).abs().median()
    scale = 1.4826 * mad if mad > 0 else log_return.std(ddof=0)
    if scale and np.isfinite(scale):
        z_score = (log_return - median).abs() / scale
        flagged = unique.index[z_score > outlier_sigma]
        outliers = [(ts, float(log_return.loc[ts])) for ts in flagged]
    else:
        outliers = []

    stale = _stale_runs(unique["close"], max_stale_days)

    return QualityReport(
        name=name,
        rows=len(unique),
        first_date=unique.index.min(),
        last_date=unique.index.max(),
        missing_days=list(missing),
        duplicate_dates=sorted(set(duplicates)),
        non_positive=list(non_positive),
        ohlc_violations=list(ohlc_violations),
        return_outliers=outliers,
        stale_runs=stale,
    )


def _stale_runs(series: pd.Series, max_stale_days: int) -> list[tuple[pd.Timestamp, int]]:
    """Find runs of identical closes longer than the threshold (a frozen feed)."""
    if series.empty:
        return []
    changed = series.ne(series.shift())
    group_id = changed.cumsum()
    runs = []
    for _, group in series.groupby(group_id):
        if len(group) > max_stale_days:
            runs.append((group.index[0], len(group)))
    return runs


def compare_sources(
    left: pd.DataFrame, right: pd.DataFrame, *, tolerance: float = 0.05
) -> pd.DataFrame:
    """Compare closes from two sources on their common days.

    Returns only the days where the relative difference exceeds the tolerance -
    usually a sign that one API broke or that the data has a split in it.
    """
    a = left.set_index(pd.to_datetime(left["date"]).dt.normalize())["close"]
    b = right.set_index(pd.to_datetime(right["date"]).dt.normalize())["close"]
    common = a.index.intersection(b.index)
    if common.empty:
        return pd.DataFrame(columns=["date", "left", "right", "rel_diff"])
    diff = (a.loc[common] - b.loc[common]).abs() / b.loc[common].abs()
    flagged = diff[diff > tolerance]
    return pd.DataFrame(
        {
            "date": flagged.index,
            "left": a.loc[flagged.index].to_numpy(),
            "right": b.loc[flagged.index].to_numpy(),
            "rel_diff": flagged.to_numpy(),
        }
    ).reset_index(drop=True)


def check_macro(df: pd.DataFrame, *, name: str = "macro") -> dict[str, Any]:
    """Checks on a macro series: date ordering, duplicates, publication sanity."""
    if df.empty:
        return {"name": name, "rows": 0, "is_clean": False, "detail": "no data"}
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["available_from"] = pd.to_datetime(data["available_from"])
    lag = (data["available_from"] - data["date"]).dt.days
    return {
        "name": name,
        "rows": len(data),
        "first_date": str(data["date"].min().date()),
        "last_date": str(data["date"].max().date()),
        "duplicates": int(data.duplicated(subset="date").sum()),
        "negative_lag": int((lag < 0).sum()),
        "median_lag_days": float(lag.median()),
        "max_lag_days": int(lag.max()),
        "nulls": int(data["value"].isna().sum()),
        "is_clean": bool(
            data.duplicated(subset="date").sum() == 0
            and (lag < 0).sum() == 0
            and data["value"].isna().sum() == 0
        ),
    }
