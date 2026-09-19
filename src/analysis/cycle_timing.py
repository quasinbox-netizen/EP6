"""The folk calendar: about 365 days down from a cycle top, about 1,064 up.

It is the most repeated timing claim about bitcoin, and it is repeated because
three cycles in a row roughly fit it. This module measures the fit on the
lab's own prices instead of quoting it, and says where the calendar would put
today. What it cannot do is make three examples into a law - the page that
shows this says so next to the numbers, and nothing here is scored as a
forecast.

How a top and a bottom are found, written down before anything is drawn so
the rule cannot be adjusted to the picture:

* top    - the highest close in the two years (730 days) after a halving;
* bottom - the lowest close between that top and the next halving.

Two years is the folk window as it is usually told. It is also a choice made
knowing roughly where the tops were, which is exactly why the page does not
test anything with it: a definition shaped around history cannot then be used
to show that history has a shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from features.halving import HALVINGS

TOP_WINDOW_DAYS = 730
FOLK_DOWN_DAYS = 365
FOLK_UP_DAYS = 1064


@dataclass(frozen=True)
class Lap:
    halving: pd.Timestamp
    top: pd.Timestamp
    top_price: float
    bottom: pd.Timestamp | None
    bottom_price: float | None
    # False while the next halving has not arrived: the lowest close so far is
    # only the lowest so far, and the page has to say "so far".
    bottom_final: bool

    @property
    def down_days(self) -> int | None:
        return None if self.bottom is None else int((self.bottom - self.top).days)


@dataclass(frozen=True)
class CycleTiming:
    laps: list
    as_of: pd.Timestamp
    last_price: float

    @property
    def current(self) -> Lap:
        return self.laps[-1]

    def up_days(self) -> list:
        """Bottom of one lap to the top of the next - the '1,064' half."""
        runs = []
        for before, after in zip(self.laps[:-1], self.laps[1:]):
            if before.bottom is not None and before.bottom_final:
                runs.append((before.bottom, after.top, int((after.top - before.bottom).days)))
        return runs

    def down_days(self) -> list:
        """Top to bottom, finished laps only - the '365' half."""
        return [(lap.top, lap.bottom, lap.down_days) for lap in self.laps
                if lap.bottom is not None and lap.bottom_final]

    @property
    def days_since_top(self) -> int:
        return int((self.as_of - self.current.top).days)

    @property
    def phase(self) -> str:
        """'down' until the current lap's bottom is behind us by the folk count.

        Deliberately crude: the lap is 'down' from its top until the folk
        calendar says the bottom should have come, and 'up' after that. The
        page labels it as the calendar's opinion, not the market's.
        """
        return "down" if self.days_since_top < FOLK_DOWN_DAYS else "up"

    @property
    def folk_bottom_date(self) -> pd.Timestamp:
        return self.current.top + pd.Timedelta(days=FOLK_DOWN_DAYS)

    @property
    def folk_next_top_date(self) -> pd.Timestamp:
        return self.folk_bottom_date + pd.Timedelta(days=FOLK_UP_DAYS)


def cycle_timing(close: pd.Series) -> CycleTiming | None:
    """Tops and bottoms per halving lap, from daily closes."""
    close = close.dropna().sort_index()
    if close.empty:
        return None
    halvings = [day for day, _, _ in HALVINGS]
    as_of = close.index[-1]
    laps = []
    for halving, following in zip(halvings, halvings[1:] + [None]):
        if halving > as_of:
            break
        window = close[(close.index >= halving)
                       & (close.index <= halving + pd.Timedelta(days=TOP_WINDOW_DAYS))]
        if window.empty:
            continue
        top = window.idxmax()
        end = following if following is not None and following <= as_of else None
        after = close[(close.index > top) & ((close.index < end) if end is not None else True)]
        bottom = after.idxmin() if not after.empty else None
        laps.append(Lap(
            halving=halving, top=top, top_price=float(close[top]),
            bottom=bottom,
            bottom_price=None if bottom is None else float(close[bottom]),
            bottom_final=end is not None,
        ))
    if not laps:
        return None
    return CycleTiming(laps=laps, as_of=as_of, last_price=float(close.iloc[-1]))
