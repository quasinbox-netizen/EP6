"""Flagi zdarzen - wylacznie wsteczne.

Flaga zapala sie w dniu, w ktorym rynek poznal zdarzenie (`available_from`),
i gasnie po N dniach. Nie ma tu flag "przed zdarzeniem": w backtescie
oznaczalyby wiedze o przyszlosci. Okna poprzedzajace istnieja tylko w module
event study, ktory jest narzedziem opisowym, nie sygnalem.

Zestaw kolumn jest STALY i nie zalezy od tego, ile zdarzen juz zaszlo.
Gdyby kolumna pojawiala sie dopiero po pierwszym zdarzeniu danej kategorii,
ramka cech z 2014 r. mialaby inny ksztalt niz ramka z 2024 r., a test
look-ahead nie odroznilby "kolumna jeszcze nie istnieje" od "wartosc jest zla".
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
    """Flagi 0/1 dla kazdej kategorii (lub nazwy) zdarzenia i kazdego okna."""
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
    """Ile dni minelo od ostatniego zdarzenia danej kategorii (NaN przed pierwszym)."""
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
