"""Position sizing, and the proof it does not see the future.

Sizing is where a backtest cheats most easily. A volatility estimate that
includes day t, used to size the position earning day t's return, quietly buys
small before every crash and produces a beautiful equity curve from nothing.
The look-ahead tests here are the point of the file; the rest is arithmetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.sizing import (
    TRADING_DAYS,
    conditional_volatility,
    realised_volatility,
    volatility_target_position,
)


def returns_with_regimes(calm=800, wild=400, seed=5):
    """Calm, then violent, then calm again - so sizing has something to react to."""
    rng = np.random.default_rng(seed)
    parts = [
        rng.normal(0, 0.01, calm),
        rng.normal(0, 0.06, wild),
        rng.normal(0, 0.01, calm),
    ]
    values = np.concatenate(parts)
    index = pd.bdate_range("2012-01-02", periods=len(values), freq="D")
    return pd.Series(values, index=index)


# --- the sizing formula ----------------------------------------------------


def test_position_is_target_over_forecast():
    daily = 0.60 / np.sqrt(TRADING_DAYS)  # exactly the target
    volatility = pd.Series([daily] * 10)
    position = volatility_target_position(volatility, target_annual_volatility=0.60)
    assert position.iloc[0] == pytest.approx(1.0)


def test_a_calmer_market_gets_a_bigger_position():
    calm = 0.30 / np.sqrt(TRADING_DAYS)
    wild = 1.20 / np.sqrt(TRADING_DAYS)
    volatility = pd.Series([calm, wild])
    position = volatility_target_position(
        volatility, target_annual_volatility=0.60, max_leverage=10.0
    )
    assert position.iloc[0] > position.iloc[1]
    assert position.iloc[0] == pytest.approx(2.0)
    assert position.iloc[1] == pytest.approx(0.5)


def test_leverage_cap_is_enforced():
    """Without the cap the formula asks for 3x in a quiet week.

    A backtest that allows that is testing a different and far riskier strategy
    than the one most readers assume they are reading about.
    """
    tiny = 0.05 / np.sqrt(TRADING_DAYS)
    position = volatility_target_position(
        pd.Series([tiny]), target_annual_volatility=0.60, max_leverage=1.0
    )
    assert position.iloc[0] == 1.0


def test_zero_or_missing_volatility_does_not_produce_infinite_leverage():
    volatility = pd.Series([0.0, np.nan, 0.01])
    position = volatility_target_position(volatility, max_leverage=1.0)
    assert np.isfinite(position).all()
    assert (position <= 1.0).all()
    assert position.iloc[0] == 0.0 and position.iloc[1] == 0.0


# --- look-ahead ------------------------------------------------------------


def test_conditional_volatility_never_uses_a_future_return():
    """The check the whole module rests on.

    Every value is recomputed from a truncated series ending at its own date.
    If any of them changes, the full-sample version was reading ahead.
    """
    returns = returns_with_regimes(calm=500, wild=200, seed=9)
    window = 300
    full = conditional_volatility(returns, window=window, refit_every=1000)

    for date in list(full.index)[::80]:
        truncated = returns.loc[:date]
        partial = conditional_volatility(truncated, window=window, refit_every=1000)
        assert partial.index[-1] == date
        assert partial.iloc[-1] == pytest.approx(full.loc[date], rel=1e-9), (
            f"{date}: value changed when future data was removed"
        )


def test_volatility_forecast_is_for_the_next_day_not_today():
    """The value at t must be sigma for t+1.

    Sized on today's own volatility, the position would be small on the day of
    a crash rather than the day after - which is exactly the free lunch a
    backtest must not be given.
    """
    from forecast.volatility import SCALE, fit_garch, update_state

    returns = returns_with_regimes(calm=400, wild=100, seed=11)
    window = 300
    series = conditional_volatility(returns, window=window, refit_every=1000)

    last_date = series.index[-1]
    position = returns.index.get_loc(last_date)
    parameters = fit_garch(returns.iloc[:window])
    state = update_state(parameters, returns.iloc[window : position + 1])
    expected = np.sqrt(
        state.omega + state.alpha * state.last_shock**2 + state.beta * state.last_variance
    ) / SCALE
    assert series.iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_realised_volatility_is_lagged():
    """The cheap comparison must be causal too, or it wins by cheating."""
    returns = returns_with_regimes(calm=200, wild=100, seed=13)
    realised = realised_volatility(returns, span=20)
    # The value at t is built from returns strictly before t.
    manual = returns.ewm(span=20, min_periods=20).std()
    assert realised.iloc[50] == pytest.approx(manual.iloc[49])


# --- behaviour -------------------------------------------------------------


def test_sizing_reacts_to_a_volatility_regime_change():
    """The premise: a violent stretch must shrink the position.

    If it does not, the forecast is not tracking volatility and the whole
    exercise is decoration.
    """
    returns = returns_with_regimes(calm=800, wild=400, seed=17)
    volatility = conditional_volatility(returns, window=600, refit_every=200)
    position = volatility_target_position(volatility, target_annual_volatility=0.60)

    # Measured in the SETTLED part of the violent stretch, not from its first
    # day. The forecast needs a few days of large returns before it reflects
    # them, so averaging across the ramp-up understates the reaction - an
    # earlier version did that and read 0.77 where the settled level is far
    # lower.
    settled = position.loc[returns.index[950] : returns.index[1150]].mean()
    calm_again = position.loc[returns.index[1400] :].mean()
    assert settled < calm_again * 0.7, f"settled {settled:.2f} vs calm {calm_again:.2f}"
    # And the direction alone, over the whole stretch, must still hold.
    assert position.loc[returns.index[800] : returns.index[1150]].mean() < calm_again


def test_a_short_sample_is_refused_rather_than_guessed():
    returns = returns_with_regimes(calm=100, wild=50)
    with pytest.raises(ValueError, match="need at least"):
        conditional_volatility(returns, window=1000)


def test_the_series_starts_after_the_window_not_at_the_beginning():
    returns = returns_with_regimes(calm=500, wild=100, seed=19)
    series = conditional_volatility(returns, window=400, refit_every=500)
    assert series.index[0] == returns.index[400]
    assert len(series) == len(returns) - 400


# --- rebalance band --------------------------------------------------------


def test_band_holds_until_the_target_drifts_far_enough():
    from backtest.sizing import apply_rebalance_band

    target = pd.Series([1.00, 1.05, 1.08, 1.20, 1.19, 0.90])
    held = apply_rebalance_band(target, band=0.10)
    # Starts at the first target, ignores small drifts, moves on the big one.
    assert held.tolist() == [1.00, 1.00, 1.00, 1.20, 1.20, 0.90]


def test_a_zero_band_changes_nothing():
    from backtest.sizing import apply_rebalance_band

    target = pd.Series([0.4, 0.9, 0.2, 0.7])
    assert apply_rebalance_band(target, band=0.0).tolist() == target.tolist()


def test_a_wider_band_trades_less():
    """The whole point of the band: turnover is what it buys.

    Vol targeting retraded 8.9 times a year on the real data and paid 24.5% of
    capital in costs, against a strategy whose entire benefit is a smoother
    ride. The band exists to make that affordable.
    """
    from backtest.sizing import apply_rebalance_band

    rng = np.random.default_rng(29)
    target = pd.Series(rng.uniform(0.2, 1.0, 2000))

    def turnover(band):
        held = apply_rebalance_band(target, band=band)
        return float(held.diff().abs().sum())

    assert turnover(0.30) < turnover(0.10) < turnover(0.0)


def test_band_never_invents_a_position_outside_the_targets():
    """It may only hold a level the target actually asked for at some point."""
    from backtest.sizing import apply_rebalance_band

    rng = np.random.default_rng(31)
    target = pd.Series(rng.uniform(0.0, 1.0, 500))
    held = apply_rebalance_band(target, band=0.15)
    assert set(np.round(held.unique(), 12)).issubset(set(np.round(target.unique(), 12)))


def test_sized_position_anticipates_larger_moves():
    """The claim the whole module rests on, stated as a measurement.

    If the position is not smaller before bigger moves, the forecast carries no
    information and the sizing is astrology with a GARCH in it. On the real
    data this correlation is -0.186.
    """
    returns = returns_with_regimes(calm=900, wild=500, seed=37)
    volatility = conditional_volatility(returns, window=600, refit_every=200)
    position = volatility_target_position(volatility, target_annual_volatility=0.60)

    future_move = returns.reindex(position.index).abs()
    correlation = position.shift(1).corr(future_move)
    assert correlation < -0.10, f"correlation {correlation:+.3f} - no information"
