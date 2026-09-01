"""Faza 5 - backtest.

Baseline kup-i-trzymaj jest w kazdym istotnym tescie: strategia, ktora go
nie bije, jest gorsza niezaleznie od tego, jak dobrze wyglada sama.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import (
    BacktestConfig,
    compare,
    compute_metrics,
    excess_over_baseline,
    max_drawdown,
    run_backtest,
)
from backtest.strategies import (
    buy_and_hold,
    combine,
    from_mask,
    halving_window,
    macro_regime,
    trend_following,
)
from validation.synthetic import random_walk_prices

FREE = BacktestConfig(fee_bps=0.0, slippage_bps=0.0)


@pytest.fixture
def prices() -> pd.DataFrame:
    return random_walk_prices(1500, start="2016-01-01", seed=21)


@pytest.fixture
def close(prices) -> pd.Series:
    return prices.set_index("date")["close"]


# --- baseline i mechanika -------------------------------------------------


def test_buy_and_hold_tracks_the_asset(close):
    result = run_backtest(close, buy_and_hold(close.index), FREE, name="kup i trzymaj")
    asset_return = close.iloc[-1] / close.iloc[0] - 1
    # Bez kosztow kup-i-trzymaj to dokladnie zwrot aktywa: sygnal z dnia t
    # zbiera zwrot dnia t+1, wiec pierwszy dzien idzie na wejscie w pozycje.
    assert result.metrics["total_return"] == pytest.approx(asset_return, rel=1e-9)
    assert result.metrics["time_in_market"] == pytest.approx(1.0, abs=0.01)
    assert result.metrics["n_position_changes"] == 1


def test_flat_signal_earns_nothing(close):
    flat = pd.Series(0.0, index=close.index)
    result = run_backtest(close, flat, FREE)
    assert result.metrics["total_return"] == pytest.approx(0.0)
    assert result.metrics["total_cost"] == 0.0
    assert result.equity.nunique() == 1


def test_execution_lag_blocks_same_day_knowledge(close):
    """Sygnal "wiedzacy" dzisiejszy zwrot dziala tylko przy zerowym opoznieniu."""
    daily_return = close.pct_change()
    oracle = (daily_return > 0).astype(float)

    cheating = run_backtest(close, oracle, BacktestConfig(0, 0, execution_lag_days=0))
    honest = run_backtest(close, oracle, BacktestConfig(0, 0, execution_lag_days=1))

    assert cheating.metrics["total_return"] > 10.0, "kontrola: bez opoznienia to maszynka do pieniedzy"
    assert honest.metrics["total_return"] < cheating.metrics["total_return"] / 100
    assert honest.metrics["sharpe"] < 1.0


def test_signal_is_not_shifted_twice(close):
    """Pozycja dnia t ma byc sygnalem z t-1, nie z t-2."""
    signal = pd.Series(0.0, index=close.index)
    signal.iloc[10:20] = 1.0
    result = run_backtest(close, signal, FREE)
    assert result.positions.iloc[10] == 0.0
    assert result.positions.iloc[11] == 1.0
    assert result.positions.iloc[20] == 1.0
    assert result.positions.iloc[21] == 0.0


def test_short_positions_require_opt_in(close):
    signal = pd.Series(-1.0, index=close.index)
    long_only = run_backtest(close, signal, BacktestConfig(0, 0))
    with_shorts = run_backtest(close, signal, BacktestConfig(0, 0, allow_short=True))
    assert long_only.positions.min() == 0.0
    assert with_shorts.positions.min() == -1.0


# --- koszty ---------------------------------------------------------------


def test_costs_reduce_returns(close):
    signal = pd.Series(0.0, index=close.index)
    signal.iloc[::10] = 1.0  # duzy obrot

    free = run_backtest(close, signal, BacktestConfig(0, 0))
    cheap = run_backtest(close, signal, BacktestConfig(5, 5))
    expensive = run_backtest(close, signal, BacktestConfig(50, 50))

    assert free.metrics["total_return"] > cheap.metrics["total_return"]
    assert cheap.metrics["total_return"] > expensive.metrics["total_return"]
    assert expensive.metrics["total_cost"] > cheap.metrics["total_cost"]


def test_cost_is_charged_on_turnover_not_on_trade_count(close):
    """Zmiana pozycji o 0.1 kosztuje dziesiec razy mniej niz o 1.0."""
    config = BacktestConfig(fee_bps=10, slippage_bps=10)
    small = pd.Series(0.0, index=close.index)
    small.iloc[100] = 0.1
    big = pd.Series(0.0, index=close.index)
    big.iloc[100] = 1.0

    cost_small = run_backtest(close, small, config).costs.sum()
    cost_big = run_backtest(close, big, config).costs.sum()
    assert cost_big == pytest.approx(10 * cost_small, rel=1e-9)


def test_buy_and_hold_pays_entry_cost_once(close):
    config = BacktestConfig(fee_bps=10, slippage_bps=15)
    result = run_backtest(close, buy_and_hold(close.index), config)
    assert result.metrics["n_position_changes"] == 1
    assert result.costs.sum() == pytest.approx(config.cost_rate, rel=1e-9)


# --- metryki --------------------------------------------------------------


def test_max_drawdown_matches_hand_calculation():
    equity = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert max_drawdown(equity) == pytest.approx(-0.5)


def test_sharpe_of_constant_positive_returns_is_infinite_free():
    index = pd.date_range("2020-01-01", periods=365, freq="D")
    returns = pd.Series(0.001, index=index)
    equity = 100 * (1 + returns).cumprod()
    metrics = compute_metrics(equity, returns, pd.Series(1.0, index=index),
                              pd.Series(0.0, index=index))
    assert np.isnan(metrics["sharpe"]) or metrics["sharpe"] > 100


def test_metrics_report_more_than_total_return(close):
    result = run_backtest(close, buy_and_hold(close.index), FREE)
    required = {
        "total_return", "cagr", "sharpe", "sortino", "max_drawdown",
        "calmar", "win_rate", "time_in_market", "turnover_annual", "total_cost",
    }
    assert required <= set(result.metrics)
    assert result.metrics["max_drawdown"] <= 0


def test_win_rate_counts_only_days_in_market(close):
    signal = pd.Series(0.0, index=close.index)
    signal.iloc[500:600] = 1.0
    result = run_backtest(close, signal, FREE)
    assert 0.0 <= result.metrics["win_rate"] <= 1.0
    assert result.metrics["time_in_market"] == pytest.approx(100 / len(close), abs=0.01)


# --- strategie i porownanie ----------------------------------------------


def test_halving_strategy_is_out_of_market_between_cycles():
    index = pd.date_range("2019-01-01", "2021-12-31", freq="D")
    signal = halving_window(index, days_after=180)
    assert signal.loc[pd.Timestamp("2020-05-11")] == 1.0
    assert signal.loc[pd.Timestamp("2020-11-07")] == 1.0
    assert signal.loc[pd.Timestamp("2020-11-08")] == 0.0
    assert signal.loc[pd.Timestamp("2019-06-01")] == 0.0


def test_trend_following_waits_for_enough_history(close):
    signal = trend_following(close, fast=50, slow=200)
    assert (signal.iloc[:199] == 0).all()
    assert signal.isin([0.0, 1.0]).all()


def test_macro_regime_selects_only_listed_labels():
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    regime = pd.Series(
        ["expanding_falling"] * 5 + ["contracting_rising"] * 5, index=index
    )
    signal = macro_regime(regime)
    assert signal.iloc[:5].eq(1.0).all()
    assert signal.iloc[5:].eq(0.0).all()


def test_combine_all_requires_both_conditions():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    a = pd.Series([1.0, 1.0, 0.0, 0.0], index=index)
    b = pd.Series([1.0, 0.0, 1.0, 0.0], index=index)
    assert combine(a, b, mode="all").tolist() == [1.0, 0.0, 0.0, 0.0]
    assert combine(a, b, mode="any").tolist() == [1.0, 1.0, 1.0, 0.0]


def test_comparison_table_includes_baseline(close):
    baseline = run_backtest(close, buy_and_hold(close.index), FREE, name="kup i trzymaj")
    strategy = run_backtest(
        close, halving_window(close.index, days_after=180), FREE, name="halving 180d"
    )
    table = compare([baseline, strategy])
    assert list(table.index) == ["kup i trzymaj", "halving 180d"]
    assert "sharpe" in table.columns and "max_drawdown" in table.columns


def test_excess_over_baseline_flags_underperformance(close):
    baseline = run_backtest(close, buy_and_hold(close.index), FREE, name="baseline")
    weak = run_backtest(close, from_mask(pd.Series(False, index=close.index)), FREE)
    verdict = excess_over_baseline(weak, baseline)
    assert verdict["beats_baseline_return"] == (
        weak.metrics["total_return"] > baseline.metrics["total_return"]
    )
    assert "sharpe_difference" in verdict


def test_partial_exposure_strategy_has_lower_volatility(close):
    full = run_backtest(close, pd.Series(1.0, index=close.index), FREE)
    half = run_backtest(close, pd.Series(0.5, index=close.index), FREE)
    assert half.metrics["volatility"] < full.metrics["volatility"]
    assert half.metrics["max_drawdown"] > full.metrics["max_drawdown"]


def test_backtest_config_reads_project_config():
    from config import load_config

    config = BacktestConfig.from_config(load_config())
    assert config.fee_bps > 0 and config.slippage_bps > 0
    assert config.execution_lag_days >= 1, "zerowe opoznienie to handel po nieznanej cenie"
    assert config.periods_per_year == 365
