"""Control group - placebo test and difference in differences.

Two things must work for this module to mean anything:

1. The calendar. The NASDAQ has ~252 sessions a year, Bitcoin 365 days.
   Without alignment, "365 days after the halving" would mean different things
   for the two assets.
2. The pairing. Halvings fall in particular macro conditions that move both
   assets at once. The difference must be computed per event, not as the
   difference of two independent means.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.control import (
    compare_with_control,
    per_event_car,
    placebo_event_study,
    to_calendar,
)
from validation.synthetic import inject_drift, random_walk_prices


def trading_days_only(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep weekdays only - imitating an exchange against always-open Bitcoin."""
    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out[out["date"].dt.dayofweek < 5].reset_index(drop=True)


@pytest.fixture
def anchors() -> list[str]:
    """Four anchors that fit comfortably in a 3000-day sample from 2013-01-01.

    The last one must leave room for a full 365-day window, otherwise it drops
    out of the sample and the test measures something other than it thinks.
    """
    return ["2014-01-15", "2016-07-09", "2018-05-20", "2020-02-10"]


# --- calendar ------------------------------------------------------------


def test_to_calendar_fills_weekends_with_last_close():
    prices = trading_days_only(random_walk_prices(400, start="2020-01-01", seed=4))
    calendar = to_calendar(prices).set_index("date")

    friday = pd.Timestamp("2020-01-03")
    saturday = pd.Timestamp("2020-01-04")
    sunday = pd.Timestamp("2020-01-05")
    assert calendar.loc[saturday, "close"] == calendar.loc[friday, "close"]
    assert calendar.loc[sunday, "close"] == calendar.loc[friday, "close"]


def test_to_calendar_has_no_gaps_and_no_lookahead():
    prices = trading_days_only(random_walk_prices(500, start="2020-01-01", seed=5))
    calendar = to_calendar(prices).set_index("date")

    expected = pd.date_range(calendar.index.min(), calendar.index.max(), freq="D")
    assert calendar.index.equals(expected), "the calendar must be continuous"

    # Every calendar value comes from a session NO LATER than that day.
    sessions = prices.set_index(pd.to_datetime(prices["date"]))["close"]
    for day in calendar.index[::37]:
        past = sessions[sessions.index <= day]
        assert calendar.loc[day, "close"] == past.iloc[-1]


def test_without_calendar_alignment_most_events_are_lost(anchors):
    """Skipping the alignment costs events, not precision.

    On a sessions-only series "365 days after" means 365 ROWS, i.e. about 511
    calendar days. On top of that, an event falling on a weekend does not exist
    in the index at all. The result: one anchor out of four survives, and the
    study quietly becomes a description of a single case.
    """
    prices = trading_days_only(random_walk_prices(3000, start="2013-01-01", seed=6))
    raw = per_event_car(prices, anchors, post=365)
    aligned = per_event_car(to_calendar(prices), anchors, post=365)

    assert len(aligned) == 4
    assert len(raw) == 1, "three anchors are lost: two on weekends, one overruns the sample"
    survivor = raw.index[0]
    assert not np.isclose(raw.loc[survivor], aligned.loc[survivor]), (
        "even the surviving event has a different window"
    )


def test_events_land_on_calendar_days_even_when_market_was_closed():
    """A Saturday halving must not drop the event from the control sample."""
    prices = trading_days_only(random_walk_prices(1500, start="2015-01-01", seed=7))
    saturday = "2016-07-09"
    assert pd.Timestamp(saturday).dayofweek == 5
    assert per_event_car(prices, [saturday], post=90).empty
    assert len(per_event_car(to_calendar(prices), [saturday], post=90)) == 1


# --- difference in differences --------------------------------------------------


def test_shared_shock_leaves_no_difference(anchors):
    """When both assets get the same shock, the difference must be insignificant."""
    rng = np.random.default_rng(11)
    treatment = random_walk_prices(3000, start="2013-01-01", seed=21, sigma=0.02)
    control = random_walk_prices(3000, start="2013-01-01", seed=22, sigma=0.012)

    # A shared shock: the same drift over the same windows for both assets.
    treatment = inject_drift(treatment, anchors, window=180, daily_drift=0.004)
    control = inject_drift(control, anchors, window=180, daily_drift=0.004)
    _ = rng

    comparison = compare_with_control(
        treatment, trading_days_only(control), anchors, post=180, horizons=[180]
    )
    assert comparison.n_events == 4
    assert comparison.summary_at["difference_p_value"] > 0.05
    assert "indistinguishable" in comparison.verdict()


def test_effect_only_in_treatment_is_detected(anchors):
    """When ONLY the studied asset gets the drift, the difference must be detected."""
    treatment = random_walk_prices(3000, start="2013-01-01", seed=21, sigma=0.012)
    control = random_walk_prices(3000, start="2013-01-01", seed=22, sigma=0.012)
    treatment = inject_drift(treatment, anchors, window=180, daily_drift=0.006)

    comparison = compare_with_control(
        treatment, trading_days_only(control), anchors, post=180, horizons=[180]
    )
    assert comparison.summary_at["difference"] > 0.5
    assert comparison.summary_at["difference_p_value"] < 0.05
    assert "NOT common" in comparison.verdict()


def test_pairing_removes_a_common_time_effect():
    """Pairing removes the common market move; comparing means would keep it.

    Both assets get the same large shock, but a different one per event. The
    paired difference must be near zero despite a huge spread in CAR.
    """
    anchors = ["2014-01-15", "2016-07-09", "2018-05-20", "2020-02-10"]
    treatment = random_walk_prices(3000, start="2013-01-01", seed=31, sigma=0.01)
    control = random_walk_prices(3000, start="2013-01-01", seed=32, sigma=0.01)
    for anchor, drift in zip(anchors, (0.010, -0.006, 0.008, -0.004)):
        treatment = inject_drift(treatment, [anchor], window=180, daily_drift=drift)
        control = inject_drift(control, [anchor], window=180, daily_drift=drift)

    comparison = compare_with_control(
        treatment, trading_days_only(control), anchors, post=180, horizons=[180]
    )
    per_event = comparison.per_event
    # The spread of the CARs is large, the spread of the DIFFERENCES is not.
    assert per_event["BTC"].std() > 3 * per_event["difference"].std()
    assert comparison.summary_at["difference_p_value"] > 0.05


def test_table_reports_every_requested_horizon(anchors):
    treatment = random_walk_prices(3000, start="2013-01-01", seed=41)
    control = trading_days_only(random_walk_prices(3000, start="2013-01-01", seed=42))
    comparison = compare_with_control(
        treatment, control, anchors, post=365, horizons=[30, 90, 365]
    )
    assert list(comparison.table.index) == [30, 90, 365]
    assert (comparison.table["n_events"] == 4).all()


def test_horizons_longer_than_the_window_are_dropped(anchors):
    treatment = random_walk_prices(3000, start="2013-01-01", seed=41)
    control = trading_days_only(random_walk_prices(3000, start="2013-01-01", seed=42))
    comparison = compare_with_control(
        treatment, control, anchors, post=90, horizons=[30, 90, 365]
    )
    assert list(comparison.table.index) == [30, 90]


def test_single_event_reports_no_p_value():
    """With one event there is no spread - and we must not pretend there is."""
    treatment = random_walk_prices(2000, start="2013-01-01", seed=51)
    control = trading_days_only(random_walk_prices(2000, start="2013-01-01", seed=52))
    comparison = compare_with_control(
        treatment, control, ["2016-01-15"], post=180, horizons=[180]
    )
    assert comparison.n_events == 1
    assert np.isnan(comparison.summary_at["difference_p_value"])
    assert "too few events" in comparison.verdict()


def test_no_shared_events_gives_empty_table():
    treatment = random_walk_prices(500, start="2013-01-01", seed=61)
    control = trading_days_only(random_walk_prices(500, start="2013-01-01", seed=62))
    comparison = compare_with_control(
        treatment, control, ["2024-01-01"], post=90, horizons=[90]
    )
    assert comparison.table.empty
    assert comparison.n_events == 0


# --- placebo --------------------------------------------------------------


def test_placebo_on_control_finds_nothing_on_noise(anchors):
    """The method must not "find" a halving effect in an asset without halvings."""
    control = trading_days_only(random_walk_prices(3000, start="2013-01-01", seed=71))
    result = placebo_event_study(control, anchors, post=180, n_boot=1000)
    assert result.n_events == 4
    assert result.car_summary["p_value"] > 0.05


def test_paired_difference_test_is_calibrated():
    """On two independent noise series rejections must be ~5%, not 30%.

    Without this test, "the difference is insignificant" on real data would
    mean nothing: the method might simply never detect anything, or detect
    everything. The control is filtered to weekdays, so we also check that the
    calendar alignment itself introduces no systematic bias.
    """
    anchors = ["2014-01-15", "2016-07-09", "2018-05-20", "2020-02-10"]
    rejections = 0
    differences = []
    trials = 40
    for seed in range(trials):
        treatment = random_walk_prices(3000, start="2013-01-01", seed=7000 + seed)
        control = random_walk_prices(3000, start="2013-01-01", seed=9000 + seed)
        comparison = compare_with_control(
            treatment, trading_days_only(control), anchors, post=90, horizons=[90]
        )
        differences.append(comparison.summary_at["difference"])
        if comparison.summary_at["difference_p_value"] < 0.05:
            rejections += 1

    assert rejections <= 6, f"too many false discoveries: {rejections}/{trials}"
    mean_difference = float(np.mean(differences))
    standard_error = float(np.std(differences) / np.sqrt(trials))
    assert abs(mean_difference) < 4 * standard_error, (
        f"the calendar alignment shifts the difference by {mean_difference:+.3f}"
    )
