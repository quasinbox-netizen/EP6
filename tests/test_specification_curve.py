"""The specification curve, and the proof that it can detect anything at all.

A null result from a test with no power is not a finding, it is silence. The
power tests here plant an effect of known size and require the curve to call it
ROBUST; without them, "NOT ROBUST" on the real halvings would be unreadable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.specification import (
    ALPHA,
    ESTIMATION_WINDOWS,
    HORIZONS,
    RETURN_TYPES,
    Specification,
    build_grid,
    curve_statistics,
    permutation_test,
    run_curve,
    shifted_dates,
    verdict,
)

SERIES = ["alpha", "beta"]


def price_frame(n_days: int = 3000, seed: int = 7, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=n_days, freq="D")
    steps = rng.normal(drift, 0.03, size=len(dates))
    return pd.DataFrame({"date": dates, "close": 100 * np.exp(np.cumsum(steps))})


def plant_effect(frame: pd.DataFrame, event_dates, *, daily: float, days: int) -> pd.DataFrame:
    """Add `daily` to the log return on each of the `days` after every event."""
    out = frame.copy()
    log_price = np.log(out["close"].to_numpy())
    index = pd.DatetimeIndex(out["date"])
    bump = np.zeros(len(out))
    for event in pd.DatetimeIndex(event_dates):
        start = index.searchsorted(event)
        for offset in range(days):
            position = start + offset
            if position < len(bump):
                bump[position] += daily
    out["close"] = np.exp(log_price + np.cumsum(bump))
    return out


@pytest.fixture
def frames():
    return {name: price_frame(seed=i + 1) for i, name in enumerate(SERIES)}


@pytest.fixture
def events(frames):
    index = pd.DatetimeIndex(frames["alpha"]["date"])
    return pd.DatetimeIndex([index[600], index[1300], index[2000]])


def test_grid_size_and_no_duplicates():
    grid = build_grid(SERIES)
    # abnormal=True keeps every estimation window; abnormal=False collapses to one,
    # because without an abnormal return the window changes nothing.
    per_series = len(HORIZONS) * len(RETURN_TYPES) * (len(ESTIMATION_WINDOWS) + 1)
    assert len(grid) == len(SERIES) * per_series
    assert len(set(grid)) == len(grid)


def test_raw_specifications_are_not_counted_three_times():
    """The collapse is the point of the previous test; this says why it matters.

    Left uncollapsed, every raw specification would appear once per estimation
    window. That inflates the denominator with copies, so the share of
    significant specifications would fall without a single result changing.
    """
    raw = [s for s in build_grid(["alpha"]) if not s.abnormal]
    keys = [(s.horizon, s.return_type) for s in raw]
    assert len(keys) == len(set(keys))


def test_shifted_dates_land_on_real_days_and_keep_spacing_unless_wrapped(frames, events):
    """Spacing survives only while the block stays inside the index.

    The name says "unless wrapped" on purpose. A wrapped draw keeps the gaps
    circularly, not on the calendar, and on the real data that is most of the
    draws - so a test called "preserves spacing" would assert something the
    method does not promise.
    """
    index = pd.DatetimeIndex(frames["alpha"]["date"])
    positions = np.array([index.searchsorted(d) for d in events])
    for shift in (1, 137, 999):
        moved = shifted_dates(events, index, shift)
        assert set(moved).issubset(set(index))
        assert len(moved) == len(events)
        moved_positions = np.sort(np.array([index.searchsorted(d) for d in moved]))
        # Spacing survives unless the block wrapped around the end of the index.
        wrapped = ((positions + shift) >= len(index)).any() and not ((positions + shift) >= len(index)).all()
        if not wrapped:
            assert np.array_equal(np.diff(moved_positions), np.diff(np.sort(positions)))


def test_curve_on_noise_is_not_robust(frames, events):
    curve = run_curve(frames, events)
    stats = curve_statistics(curve)
    assert stats["n_specs"] == len(build_grid(SERIES))
    result = permutation_test(frames, events, stats, n_permutations=40, seed=1)
    assert "NOT ROBUST" in verdict(stats, result)


def test_curve_detects_a_planted_effect(frames, events):
    """Power: a real, large effect must survive the permutation test.

    0.4% per day for 90 days is roughly +43% cumulative - the order of size
    the halving is popularly claimed to produce. If the curve cannot see that,
    it cannot see anything, and its null result on the real data means nothing.
    """
    planted = {name: plant_effect(frame, events, daily=0.004, days=90)
               for name, frame in frames.items()}
    stats = curve_statistics(run_curve(planted, events))
    assert stats["median_car"] > 0.15, "the planted effect should dominate the curve"
    result = permutation_test(planted, events, stats, n_permutations=40, seed=2)
    assert "ROBUST" in verdict(stats, result)
    assert "NOT ROBUST" not in verdict(stats, result)


def test_permutation_p_value_is_never_zero(frames, events):
    """p=0 claims the observed arrangement is impossible, which it is not.

    The observed curve is one of the arrangements the null can produce, so it
    belongs in both numerator and denominator.
    """
    planted = {name: plant_effect(frame, events, daily=0.01, days=90)
               for name, frame in frames.items()}
    stats = curve_statistics(run_curve(planted, events))
    result = permutation_test(planted, events, stats, n_permutations=20, seed=3)
    assert result["median_p_value"] > 0
    assert result["median_p_value"] >= 1 / (result["n_permutations"] + 1)


def test_significance_flag_matches_alpha(frames, events):
    curve = run_curve(frames, events)
    usable = curve[np.isfinite(curve["p_value"])]
    assert (usable["significant"] == (usable["p_value"] < ALPHA)).all()


def test_specification_label_is_unique_per_specification():
    grid = build_grid(SERIES)
    assert len({s.label() for s in grid}) == len(grid)


def test_single_event_specifications_are_excluded(frames, events):
    """One event is an observation, not a specification result.

    A one-event window produces a CAR but no interval and no p-value. Left in,
    it would shift the median while being structurally unable to count as
    significant - so the shortest exchange history would drag the curve down
    without ever being able to lift it.
    """
    from analysis.specification import run_specification

    short = frames["alpha"].tail(400).reset_index(drop=True)
    only_one = pd.DatetimeIndex([pd.DatetimeIndex(short["date"])[200]])
    spec = Specification("alpha", 30, (-250, -31), False, "log")

    row = run_specification(short, only_one, spec)
    assert row["n_events"] <= 1
    assert not np.isfinite(row["car"])
    assert row["significant"] is False

    curve = pd.DataFrame([row])
    assert curve_statistics(curve)["n_specs"] == 0


def test_leading_nan_return_is_dropped(frames):
    """The first day has no return; nancumsum would score it as zero.

    An event window reaching the start of the series must not silently gain a
    free flat day.
    """
    from analysis.event_study import log_returns

    for return_type in RETURN_TYPES:
        series = log_returns(frames["alpha"], return_type=return_type)
        assert np.isfinite(series.iloc[0]), f"{return_type}: leading NaN survived"
        assert len(series) == len(frames["alpha"]) - 1


def test_event_without_a_baseline_is_dropped_not_zeroed():
    """An unusable event must reduce n, not enter the study as a flat zero.

    When the estimation window reaches back past the start of the series there
    is no baseline. Subtracting NaN makes the whole row NaN, and because the
    CAR is built with nancumsum, that row would otherwise be scored as an event
    where nothing happened - pulling the mean toward zero and inflating the
    spread with a fabricated observation.

    This is not a corner case. It fires on permutation draws that land near the
    start of the price history, which is precisely the null distribution every
    p-value in the curve is measured against.
    """
    from analysis.event_study import event_study

    frame = price_frame(n_days=900, seed=11)
    index = pd.DatetimeIndex(frame["date"])
    early, late = index[60], index[600]  # 60 days in: no room for a -250 window

    result = event_study(
        frame, [early, late], pre=30, post=90,
        abnormal=True, estimation_window=(-250, -31), n_boot=0,
    )
    assert result.n_events == 1, "the event without a baseline should be dropped"
    assert early in set(result.skipped_events)
    assert late in set(result.used_events)

    # And the surviving event must match a study run on it alone - proof that
    # the dropped one left no trace in the numbers.
    alone = event_study(
        frame, [late], pre=30, post=90,
        abnormal=True, estimation_window=(-250, -31), n_boot=0,
    )
    assert np.isclose(result.table.loc[90, "car"], alone.table.loc[90, "car"])


def test_permutation_reports_how_many_draws_wrapped(frames, events):
    """The wrapped share must be reported, not assumed away.

    A wrapped draw keeps the gaps between events only circularly. On the real
    data three quarters of the draws wrap, so a reader who is told the null
    "preserves spacing" would be misled about what the p-value was measured
    against. The number is therefore part of the output.
    """
    stats = curve_statistics(run_curve(frames, events))
    result = permutation_test(frames, events, stats, n_permutations=25, seed=5)
    assert "share_wrapped" in result
    assert 0.0 <= result["share_wrapped"] <= 1.0
    assert result["n_wrapped"] == pytest.approx(result["share_wrapped"] * 25, abs=1)


def test_events_spanning_most_of_the_index_wrap_often(frames):
    """Sanity check on the mechanism the previous test measures.

    Events spread across nearly the whole index leave few shifts that do not
    run off the end, so most draws must wrap. This pins the reason the real
    data wraps 76% of the time: it is the span of the halvings, not a bug.
    """
    index = pd.DatetimeIndex(frames["alpha"]["date"])
    wide = pd.DatetimeIndex([index[100], index[len(index) - 100]])
    stats = curve_statistics(run_curve(frames, wide))
    result = permutation_test(frames, wide, stats, n_permutations=25, seed=6)
    assert result["share_wrapped"] > 0.5


def test_permutation_shifts_along_the_longest_history(frames, events):
    """The null must span the same history the observed curve was measured on.

    If the shift ran along a short exchange history, the real event dates would
    fall outside it: searchsorted returns len(index), the modulo then places
    them somewhere unrelated, and the null explores only a fraction of the
    sample. Nothing raises - the p-value is simply computed against the wrong
    null. Dict ordering must not decide that.
    """
    short = frames["alpha"].tail(300).reset_index(drop=True)
    mixed = {"short": short, "alpha": frames["alpha"]}
    reversed_order = {"alpha": frames["alpha"], "short": short}

    stats = curve_statistics(run_curve(mixed, events))
    first = permutation_test(mixed, events, stats, n_permutations=12, seed=9)
    second = permutation_test(reversed_order, events, stats, n_permutations=12, seed=9)

    assert first["median_p_value"] == second["median_p_value"]
    assert first["share_wrapped"] == second["share_wrapped"]
