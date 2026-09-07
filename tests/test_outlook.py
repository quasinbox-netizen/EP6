"""Base rates and cycle laps - the descriptive half of the front page.

The failure mode this file guards is not a crash. It is a plausible-looking
median computed from three overlapping windows and printed as if it were a
finding, which is the exact mistake the rest of the project exists to catch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.outlook import (
    MIN_EFFECTIVE_N,
    base_rates,
    current_state,
    cycle_paths,
    forward_returns,
    independent_count,
)
from features.halving import CONFIRMED_HALVINGS
from validation.synthetic import random_walk_prices


@pytest.fixture
def features() -> pd.DataFrame:
    """Prices plus the two backward-looking columns the conditioning uses."""
    # Long enough to contain two halvings: a shorter window has no lap to
    # draw and no matched days to condition on.
    prices = random_walk_prices(2200, start="2019-01-01", seed=3)
    frame = prices.set_index("date")[["close"]]
    frame["ma_200_ratio"] = frame["close"] / frame["close"].rolling(200).mean()
    frame["days_since_halving"] = [
        (day - max(h for h in CONFIRMED_HALVINGS if h <= day)).days
        if (CONFIRMED_HALVINGS <= day).any() else np.nan
        for day in frame.index
    ]
    return frame


def test_forward_returns_drop_the_unfinished_tail():
    """The last h days have no outcome. Filling them in shrinks every interval."""
    close = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0],
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    forward = forward_returns(close, 2)

    assert len(forward) == 3
    assert forward.iloc[0] == pytest.approx(12.0 / 10.0 - 1)
    assert forward.index[-1] == close.index[2]


def test_independent_count_collapses_overlapping_days():
    """Thirty consecutive days are one 30-day observation, not thirty."""
    days = pd.date_range("2024-01-01", periods=30, freq="D")

    assert independent_count(days, 30) == 1
    assert independent_count(days, 10) == 3
    assert independent_count(days, 1) == 30


def test_independent_count_handles_gaps():
    """Two clusters a year apart are two observations, however dense each is."""
    days = pd.DatetimeIndex(
        list(pd.date_range("2024-01-01", periods=20, freq="D"))
        + list(pd.date_range("2025-01-01", periods=20, freq="D"))
    )
    assert independent_count(days, 90) == 2


def test_conditional_and_unconditional_arrive_together(features):
    """A conditional median alone is unreadable - the pair is the unit."""
    conditional, unconditional = base_rates(features, 30)

    assert conditional.horizon == unconditional.horizon == 30
    assert unconditional.matched_days >= conditional.matched_days
    assert unconditional.effective_n >= conditional.effective_n


def test_effective_n_is_far_below_the_matched_day_count(features):
    """The whole point of the guardrail: rows are not observations."""
    conditional, _ = base_rates(features, 90)

    assert conditional.matched_days > conditional.effective_n
    assert conditional.effective_n <= conditional.matched_days // 90 + 1


def test_thin_evidence_is_marked_unquotable(features):
    long_horizon, _ = base_rates(features, 365)

    if long_horizon.effective_n < MIN_EFFECTIVE_N:
        assert not long_horizon.usable
        assert "too few" in long_horizon.summary()


def test_state_uses_only_backward_looking_columns(features):
    state = current_state(features)

    assert state["as_of"] == features.index[-1]
    assert state["days_since_halving"] >= 0
    assert "200-day average" in state["trend_label"]


def test_cycle_paths_start_at_one_and_are_indexed_by_days(features):
    paths = cycle_paths(features["close"])

    assert paths
    for series in paths.values():
        assert series.iloc[0] == pytest.approx(1.0)
        assert series.index[0] == 0
        assert series.index.is_monotonic_increasing


def test_cycle_paths_skip_halvings_outside_the_sample():
    """A halving before the price series begins has no lap to draw."""
    close = pd.Series(
        np.linspace(100, 200, 400),
        index=pd.date_range("2025-01-01", periods=400, freq="D"),
    )
    paths = cycle_paths(close)

    assert all(pd.Timestamp(label) >= close.index[0] for label in paths)
