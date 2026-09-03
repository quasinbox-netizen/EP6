"""Halving-cycle features.

Halving dates are static data (block height 210,000 x n, confirmation time in
UTC). The date of the NEXT halving is a FORECAST - it follows from the protocol
but depends on mining speed, so it moves by roughly two weeks either way.

Time contract: `days_since_halving` is entirely backward-looking. By contrast,
`days_to_next_halving` and `cycle_progress` use the date of a future event.
That is defensible (the schedule was roughly known in advance) but it is not
free - which is why these functions have a `strict` mode that drops those
columns, and why the backtest does NOT use them by default.
"""
from __future__ import annotations

import pandas as pd

GENESIS = pd.Timestamp("2009-01-03")

# (date, block height, confirmed)
HALVINGS: list[tuple[pd.Timestamp, int, bool]] = [
    (pd.Timestamp("2012-11-28"), 210_000, True),
    (pd.Timestamp("2016-07-09"), 420_000, True),
    (pd.Timestamp("2020-05-11"), 630_000, True),
    (pd.Timestamp("2024-04-20"), 840_000, True),
    (pd.Timestamp("2028-04-20"), 1_050_000, False),  # projected
]

HALVING_DATES = pd.DatetimeIndex([d for d, _, _ in HALVINGS])
CONFIRMED_HALVINGS = pd.DatetimeIndex([d for d, _, confirmed in HALVINGS if confirmed])


def cycle_index(dates: pd.DatetimeIndex) -> pd.Series:
    """Cycle number: how many halvings have happened by that day (0 before the first)."""
    dates = pd.DatetimeIndex(dates)
    counts = [int((CONFIRMED_HALVINGS <= day).sum()) for day in dates]
    return pd.Series(counts, index=dates, name="cycle_index", dtype="int64")


def _previous_halving(day: pd.Timestamp) -> pd.Timestamp:
    past = CONFIRMED_HALVINGS[CONFIRMED_HALVINGS <= day]
    return past[-1] if len(past) else GENESIS


def _next_halving(day: pd.Timestamp) -> pd.Timestamp | None:
    future = HALVING_DATES[HALVING_DATES > day]
    return future[0] if len(future) else None


def halving_features(dates: pd.DatetimeIndex, *, strict: bool = False) -> pd.DataFrame:
    """Cycle features for the given days.

    strict=True keeps only backward-looking features (no knowledge of the next
    halving date) - that variant is fit for backtesting without caveats.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    previous = [_previous_halving(day) for day in dates]
    days_since = [(day - prev).days for day, prev in zip(dates, previous)]

    frame = pd.DataFrame(
        {
            "cycle_index": cycle_index(dates).to_numpy(),
            "days_since_halving": days_since,
        },
        index=dates,
    )
    frame.index.name = "date"

    if strict:
        return frame

    upcoming = [_next_halving(day) for day in dates]
    frame["days_to_next_halving"] = [
        float("nan") if nxt is None else (nxt - day).days for day, nxt in zip(dates, upcoming)
    ]
    span = [
        float("nan") if nxt is None else max((nxt - prev).days, 1)
        for prev, nxt in zip(previous, upcoming)
    ]
    frame["cycle_progress"] = [
        float("nan") if pd.isna(total) else since / total
        for since, total in zip(days_since, span)
    ]
    return frame


def halving_windows(
    dates: pd.DatetimeIndex, windows: list[int], *, direction: str = "after"
) -> pd.DataFrame:
    """Flags 0/1: does the day fall within N days after (or before) a halving?

    "Before" windows require knowing the future halving date - use them in a
    backtest only if you accept that assumption deliberately.
    """
    if direction not in {"after", "before"}:
        raise ValueError("direction must be 'after' or 'before'")
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    out = pd.DataFrame(index=dates)
    out.index.name = "date"
    for window in windows:
        flags = []
        for day in dates:
            if direction == "after":
                anchor = _previous_halving(day)
                hit = anchor in set(CONFIRMED_HALVINGS) and 0 <= (day - anchor).days <= window
            else:
                anchor = _next_halving(day)
                hit = anchor is not None and 0 < (anchor - day).days <= window
            flags.append(int(hit))
        out[f"halving_{direction}_{window}d"] = flags
    return out
