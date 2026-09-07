"""The public paper portfolio.

A portfolio published in public is only worth publishing if it cannot quietly
cheat, and there are exactly two ways it would: earning a return on a position
it did not hold yet, and trading for free. Both are pinned here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestConfig, run_backtest
from backtest.paper import TRADE_THRESHOLD, run_paper, state, target_weights


@pytest.fixture
def close() -> pd.Series:
    """Ten days, +10% on day 5 and flat otherwise, so a return is easy to trace."""
    index = pd.date_range("2025-01-01", periods=10, freq="D")
    prices = [100.0] * 10
    for i in range(5, 10):
        prices[i] = 110.0
    return pd.Series(prices, index=index, name="close")


def test_the_days_return_goes_to_the_weight_held_coming_into_it(close):
    """Entering on the day of a jump must not earn that jump.

    This is the single most common way a paper account beats the market: the
    weight for day t is applied to the return of day t, and the portfolio
    collects a move it was not in for.
    """
    entering_on_the_jump = pd.Series(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], index=close.index, dtype=float)
    late = run_paper(close, entering_on_the_jump, capital=1000.0, cost_rate=0.0)

    # Day 5 is the +10% day; a portfolio that first holds on day 5 misses it.
    assert late.equity.iloc[-1] == pytest.approx(1000.0)


def test_a_position_held_through_the_jump_earns_it(close):
    already_in = pd.Series(1.0, index=close.index)
    run = run_paper(close, already_in, capital=1000.0, cost_rate=0.0)

    assert run.equity.iloc[-1] == pytest.approx(1100.0)


def test_every_change_in_weight_is_charged(close):
    weights = pd.Series([0, 1, 1, 0, 0, 1, 1, 0, 0, 0], index=close.index, dtype=float)
    free = run_paper(close, weights, capital=1000.0, cost_rate=0.0)
    charged = run_paper(close, weights, capital=1000.0, cost_rate=0.0025)

    assert charged.equity.iloc[-1] < free.equity.iloc[-1]
    assert (charged.trades["cost"] > 0).all()


def test_drift_below_the_threshold_is_not_reported_as_a_trade(close):
    """A rebalance band moves the size constantly; only decisions are printed.

    Entering the position is a decision and appears. The half-point wobbles
    after it are drift, and printing them would bury the entry among rows
    nobody came to read.
    """
    weights = pd.Series(
        [0.50, 0.505, 0.51, 0.515, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52],
        index=close.index)
    run = run_paper(close, weights, capital=1000.0, cost_rate=0.0)

    assert len(run.trades) == 1                       # the entry, and nothing else
    assert run.trades.iloc[0]["date"] == close.index[0]
    assert all(abs(weights.diff().dropna()) < TRADE_THRESHOLD)


def test_holding_cash_holds_its_value(close):
    run = run_paper(close, pd.Series(0.0, index=close.index), capital=1000.0)

    assert run.equity.nunique() == 1
    assert run.current_weight == 0.0
    assert run.units == 0.0


def test_the_benchmark_is_the_same_capital_over_the_same_days(close):
    run = run_paper(close, pd.Series(0.0, index=close.index), capital=1000.0,
                    start=close.index[2])

    assert run.hold.index[0] == close.index[2]
    assert run.hold.iloc[-1] == pytest.approx(1100.0)
    assert run.equity.index[0] == close.index[2]


def test_direction_can_only_switch_the_position_off_not_size_it_up(close):
    """A rule with no edge must not be able to make the position bigger."""
    size = pd.Series(0.5, index=close.index)
    weights = target_weights(pd.Series([1.0] * 10, index=close.index), size)

    assert (weights == 0.5).all()

    flat = target_weights(pd.Series([0.0] * 10, index=close.index), size)
    assert (flat == 0.0).all()


def test_units_are_what_a_live_price_revalues(close):
    weights = pd.Series(1.0, index=close.index)
    run = run_paper(close, weights, capital=1000.0, cost_rate=0.0)
    report = state(run, flip_level=90.0, flip_text="flips below 90")

    # units * last price is the whole portfolio, because it is fully invested.
    assert report["units"] * report["lastPrice"] == pytest.approx(report["equity"])
    assert report["cash"] == pytest.approx(0.0)
    assert report["flipLevel"] == 90.0


def test_an_empty_window_does_not_crash(close):
    run = run_paper(close, pd.Series(1.0, index=close.index),
                    start=close.index[-1] + pd.Timedelta(days=5))

    assert run.equity.empty
    assert state(run, flip_level=None)["equity"] == 0.0


def test_the_summary_names_the_benchmark(close):
    run = run_paper(close, pd.Series(1.0, index=close.index), capital=1000.0,
                    cost_rate=0.0)

    assert "against" in run.summary()
    assert "holding" in run.summary()


def test_the_portfolio_reproduces_the_engine_it_reports_against(close):
    """The invariant that catches a second lag, and did not exist before.

    run_paper applies the execution lag itself, so it must be handed the raw
    target - not `positions`, which the engine has already lagged. Feeding it
    the lagged series delays every entry by a day: on this fixture the
    portfolio missed the whole +10% move and ended flat while the engine
    ended up 10%. Costs are zero here so the two are directly comparable.
    """
    signal = pd.Series([0.0] * 10, index=close.index)
    signal.iloc[4:] = 1.0
    settings = BacktestConfig(fee_bps=0, slippage_bps=0, execution_lag_days=1,
                              initial_equity=1000.0)
    engine = run_backtest(close, signal, settings)

    paper = run_paper(close, engine.signal, capital=1000.0, cost_rate=0.0)

    assert paper.equity.iloc[-1] == pytest.approx(float(engine.equity.iloc[-1]))

    # And the mistake this replaces, stated so it cannot come back silently.
    double_lagged = run_paper(close, engine.positions, capital=1000.0, cost_rate=0.0)
    assert double_lagged.equity.iloc[-1] < paper.equity.iloc[-1]
