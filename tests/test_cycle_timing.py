"""The folk calendar is measured on prices, and the measurement must not guess."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.cycle_timing import cycle_timing
from publish.trend import cycle_chart, trend_ribbon


def _laps() -> pd.Series:
    """A price with a known top and bottom inside every halving lap."""
    days = pd.date_range("2012-01-01", "2026-09-18", freq="D")
    price = pd.Series(100.0, index=days)
    marks = {
        "2013-12-04": 1000, "2015-01-14": 200,
        "2017-12-16": 20000, "2018-12-15": 3000,
        "2021-11-08": 67000, "2022-11-21": 16000,
        "2025-10-06": 125000, "2026-06-30": 58000,
    }
    anchors = pd.Series({pd.Timestamp(k): v for k, v in marks.items()})
    anchors[days[0]] = 100.0
    anchors[days[-1]] = 78000.0
    anchors = anchors.sort_index()
    logs = np.interp(days.asi8, anchors.index.asi8, np.log(anchors.to_numpy()))
    return pd.Series(np.exp(logs), index=days)


def test_tops_and_bottoms_land_on_the_known_dates():
    timing = cycle_timing(_laps())

    assert [f"{lap.top:%Y-%m-%d}" for lap in timing.laps] == [
        "2013-12-04", "2017-12-16", "2021-11-08", "2025-10-06"]
    assert [d for *_, d in timing.down_days()] == [406, 364, 378]
    assert [d for *_, d in timing.up_days()] == [1067, 1059, 1050]


def test_the_current_bottom_is_only_the_lowest_so_far():
    """Until the next halving, the lap's low can still be undercut."""
    timing = cycle_timing(_laps())

    assert not timing.current.bottom_final
    assert timing.days_since_top == 347
    assert f"{timing.folk_bottom_date:%Y-%m-%d}" == "2026-10-06"


def test_the_unfinished_fall_is_drawn_as_unfinished():
    close = _laps()
    svg = cycle_chart(close, cycle_timing(close))

    assert "267+ d" in svg
    assert "406 d" in svg


def test_the_ribbon_colours_the_days_it_was_given():
    close = _laps()
    ribbon = trend_ribbon(close, element_id="t", days=400)

    assert 'class="tr-line up"' in ribbon or 'class="tr-line down"' in ribbon
    assert 'data-slow="' in ribbon
    assert "flipped" in ribbon
