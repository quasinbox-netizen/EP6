"""Event flags - strictly backward-looking.

A flag turns on the day the market learned about the event (`available_from`)
and turns off N days later. There are no "before the event" flags: in a
backtest they would mean knowledge of the future. Preceding windows exist only
in the event-study module, which is a descriptive tool, not a signal.

The set of columns is FIXED and does not depend on how many events have
happened yet. If a column only appeared after the first event of its category,
a feature frame from 2014 would have a different shape than one from 2024, and
the look-ahead test could not tell "this column does not exist yet" apart from
"this value is wrong".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ingest.events import VALID_CATEGORIES


def _column_keys(events: pd.DataFrame, by: str, keys: list[str] | None) -> list[str]:
    if keys is not None:
        return sorted(keys)
    if by == "category":
        return sorted(VALID_CATEGORIES)
    return sorted(events[by].unique()) if not events.empty else []


def event_flags(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    windows: list[int],
    *,
    by: str = "category",
    keys: list[str] | None = None,
) -> pd.DataFrame:
    """Flags 0/1 for every category (or name) and every window length."""
    index = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    out = pd.DataFrame(index=index)
    out.index.name = "date"

    events = events.copy()
    if not events.empty:
        events["available_from"] = pd.to_datetime(events["available_from"]).dt.normalize()

    for key in _column_keys(events, by, keys):
        group = events[events[by] == key] if not events.empty else events
        anchors = (
            pd.DatetimeIndex(sorted(group["available_from"].unique()))
            if not group.empty
            else pd.DatetimeIndex([])
        )
        for window in windows:
            flag = np.zeros(len(index), dtype=int)
            for anchor in anchors:
                mask = (index >= anchor) & (index <= anchor + pd.Timedelta(days=window))
                flag |= mask.astype(int)
            out[f"event_{key}_{window}d"] = flag
    return out


def days_since_event(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    by: str = "category",
    keys: list[str] | None = None,
) -> pd.DataFrame:
    """Days since the last event of each category (NaN before the first one)."""
    index = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    out = pd.DataFrame(index=index)
    out.index.name = "date"

    events = events.copy()
    if not events.empty:
        events["available_from"] = pd.to_datetime(events["available_from"]).dt.normalize()

    for key in _column_keys(events, by, keys):
        group = events[events[by] == key] if not events.empty else events
        if group.empty:
            out[f"days_since_event_{key}"] = np.nan
            continue
        anchors = np.sort(group["available_from"].unique())
        positions = np.searchsorted(anchors, index.to_numpy(), side="right") - 1
        values = np.where(
            positions >= 0,
            (index.to_numpy() - anchors[positions.clip(min=0)]) / np.timedelta64(1, "D"),
            np.nan,
        )
        out[f"days_since_event_{key}"] = values
    return out
