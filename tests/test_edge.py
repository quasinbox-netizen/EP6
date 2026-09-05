"""Does the edge test detect an edge, and does it refuse to invent one?

The power test matters most here. A test that never finds anything would pass
every "no edge" assertion in this file while being useless, so a rule with a
deliberately planted edge must be caught.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.edge import (
    count_round_trips,
    edge_test,
    episodes,
    fragility,
    strategy_returns,
    verdict,
)


def random_walk(n=2000, seed=1, drift=0.0005, sigma=0.03):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2012-01-02", periods=n, freq="D")
    return pd.Series(rng.normal(drift, sigma, n), index=index)


def test_buy_and_hold_cannot_have_timing_edge():
    """The validity check the whole method rests on.

    A constant position is unchanged by a circular shift, so every null draw
    equals the observed value and p must be exactly 1. If this ever returns
    anything else, the null is not doing what it claims.
    """
    returns = random_walk()
    positions = pd.Series(1.0, index=returns.index)
    result = edge_test(positions, returns, n_permutations=200)
    assert result.p_value == pytest.approx(1.0)
    assert result.observed == pytest.approx(result.null_mean)
    assert "NO EDGE" in verdict(result)


def test_a_planted_edge_is_detected():
    """Power. Without this, every "no edge" result in this file is meaningless.

    The rule is invested exactly on the days that were given an extra return,
    so its timing is genuinely informative and the test must say so.
    """
    returns = random_walk(n=2000, seed=7, drift=0.0)
    rng = np.random.default_rng(11)
    good_days = rng.random(len(returns)) < 0.30
    boosted = returns + np.where(good_days, 0.02, 0.0)

    # A position held on day i earns day i+1's return, so to capture a boosted
    # day j the position must already be on at j-1. That is shift(-1), not
    # shift(1) - getting it backwards misaligns the rule by two days and the
    # planted edge vanishes, which is how this test first failed.
    positions = pd.Series(good_days.astype(float), index=returns.index).shift(-1).fillna(0.0)

    result = edge_test(positions, boosted, n_permutations=500, seed=3)
    assert result.p_value < 0.01, result.summary()
    assert result.observed > result.null_p95
    assert "TIMING BEATS CHANCE" in verdict(result)


def test_an_arbitrary_rule_has_no_edge():
    returns = random_walk(n=2000, seed=13)
    rng = np.random.default_rng(17)
    positions = pd.Series((rng.random(len(returns)) < 0.3).astype(float), index=returns.index)
    result = edge_test(positions, returns, n_permutations=500, seed=5)
    assert result.p_value > 0.05, result.summary()


def test_positions_are_applied_to_the_next_days_return():
    """The classic backtest look-ahead, in one assertion.

    Multiplying a position by the same day's return credits the rule with a
    move it used to decide, and turns any rule that reacts to today's price
    into a money printer.
    """
    positions = np.array([0.0, 1.0, 0.0, 0.0])
    returns = np.array([0.10, 0.20, 0.30, 0.40])
    realised = strategy_returns(positions, returns)
    # Position taken on day 1 earns day 2's return, not day 1's.
    assert realised.tolist() == [0.0, 0.30, 0.0]


def test_costs_are_charged_on_turnover():
    positions = np.array([0.0, 1.0, 1.0, 0.0])
    returns = np.zeros(4)
    with_cost = strategy_returns(positions, returns, cost_rate=0.01)
    # Two changes inside the held window: in at index 1, out at index 3.
    assert with_cost.sum() < 0


# --- episodes and fragility ------------------------------------------------


def test_episodes_are_contiguous_holdings():
    positions = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=float)
    assert episodes(positions) == [(1, 3), (5, 6), (7, 10)]
    assert episodes(np.zeros(5)) == []


def test_round_trips_counts_entries_not_days():
    assert count_round_trips(np.array([0, 1, 1, 1, 0], dtype=float)) == 1
    assert count_round_trips(np.array([0, 1, 0, 1, 0], dtype=float)) == 2


def test_fragility_flags_a_result_carried_by_one_episode():
    """The failure this module was capable of producing.

    A permutation p-value is as precise as the number of draws and as strong as
    the number of episodes, and those are not the same thing. On the real
    halving rule p=0.0095 becomes p=0.1369 once the 2012 window is removed -
    one observation carrying the entire finding.
    """
    n = 1500
    index = pd.bdate_range("2012-01-02", periods=n, freq="D")
    rng = np.random.default_rng(21)
    returns = pd.Series(rng.normal(0.0, 0.02, n), index=index)

    positions = pd.Series(0.0, index=index)
    for start in (100, 500, 900):
        positions.iloc[start : start + 60] = 1.0
    # One episode is given a large edge; the other two are noise.
    returns.iloc[101 : 161] += 0.05

    result = fragility(positions, returns, n_permutations=400, seed=9)
    assert result.n_episodes == 3
    assert result.full_p_value < 0.05, "the planted episode should carry it"
    assert result.worst_p_value > 0.05, "removing that episode should destroy it"
    assert result.fragile


def test_a_broad_result_is_not_flagged_fragile():
    """Many episodes each contributing must not be called fragile."""
    n = 2500
    index = pd.bdate_range("2012-01-02", periods=n, freq="D")
    rng = np.random.default_rng(23)
    returns = pd.Series(rng.normal(0.0, 0.02, n), index=index)

    positions = pd.Series(0.0, index=index)
    for start in range(100, 2300, 200):
        positions.iloc[start : start + 40] = 1.0
        returns.iloc[start + 1 : start + 41] += 0.012

    result = fragility(positions, returns, n_permutations=400, seed=10)
    assert result.n_episodes >= 10
    assert result.full_p_value < 0.05
    assert not result.fragile, result.per_episode.to_string()


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="must align"):
        edge_test(pd.Series(np.zeros(100)), pd.Series(np.zeros(90)))


def test_too_short_a_sample_is_refused():
    with pytest.raises(ValueError, match="at least 50"):
        edge_test(pd.Series(np.zeros(10)), pd.Series(np.zeros(10)))


def test_p_value_is_never_zero():
    """p=0 claims the observed arrangement is impossible, which it is not."""
    returns = random_walk(n=800, seed=31, drift=0.0)
    positions = pd.Series((returns.shift(-1) > 0).astype(float).fillna(0.0))
    result = edge_test(positions, returns, n_permutations=200, seed=2)
    assert result.p_value > 0
    assert result.p_value >= 1 / (result.n_permutations + 1)
