"""How much to hold, from the one thing here that is predictable.

Everything else in this project answers "which way" and the answer is always
that nobody knows. This module does not ask that question. It asks how much of
the asset to hold, and it can answer because volatility - unlike direction - is
forecastable: `forecast.coverage` shows the model's 90% interval containing the
outcome 87.6% of the time across 403 non-overlapping windows.

The rule is one line. Hold

    position = target volatility / forecast volatility

so that a quiet market gets a large position and a violent one a small
position, and the RISK taken stays roughly constant instead of the quantity
held. It contains no view about where the price is going, which is what makes
it usable in a project that established there is no such view to be had.

WHAT THIS IS NOT
----------------
It is not a way to earn more. Scaling a position down in a volatile market
scales its returns down too, and if volatility carried no information the
long-run effect would be a wash. The reason to expect anything at all is that
volatility is persistent, so today's estimate says something about tomorrow -
and `backtest.edge` is used here to check whether that "something" is worth
anything after costs, rather than assuming the well-known result transfers.

THE LOOK-AHEAD THIS INVITES
---------------------------
Sizing is the easiest place in a backtest to cheat without noticing. A
volatility estimate that includes day t, used to size the position that earns
day t's return, quietly buys small before crashes. Every estimate here is
strictly causal: parameters come from a trailing window, the state is rolled
forward with returns up to t, and what comes out is the conditional volatility
for t+1. The backtest engine then applies its own execution lag on top, so the
position is held before the return it earns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast.volatility import SCALE, fit_garch, update_state

DEFAULT_WINDOW = 1460
DEFAULT_REFIT_EVERY = 30
TRADING_DAYS = 365  # crypto trades every day


def conditional_volatility(
    returns: pd.Series,
    *,
    window: int = DEFAULT_WINDOW,
    refit_every: int = DEFAULT_REFIT_EVERY,
) -> pd.Series:
    """One-day-ahead conditional volatility, as known at each date.

    The value at date t is sigma for t+1, built from returns up to and
    including t. Parameters are refitted every `refit_every` days; the variance
    recursion is rolled forward daily in between, because the parameters age
    slowly and the state does not.

    Returned in daily units. Multiply by sqrt(365) for the annualised figure.
    """
    returns = pd.Series(returns).dropna().astype(float).sort_index()
    # The real requirement, not a round number: one full window to fit on plus
    # at least one day to forecast for. An earlier version demanded window + 10
    # and refused a series that could legitimately produce a value, which broke
    # the look-ahead test - it truncates the series to each date in turn, and
    # the earliest of those has exactly window + 1 returns.
    if window < 100:
        raise ValueError(f"need at least 100 returns to fit GARCH, window is {window}")
    if len(returns) <= window:
        raise ValueError(
            f"need at least {window + 1} returns for a {window}-day window, "
            f"got {len(returns)}"
        )

    values = []
    dates = []
    parameters = None
    last_refit = None

    for position in range(window, len(returns)):
        if (position - window) % refit_every == 0 or parameters is None:
            parameters = fit_garch(returns.iloc[position - window : position])
            last_refit = position
        state = update_state(parameters, returns.iloc[last_refit : position + 1])
        # update_state leaves last_variance as sigma^2 for the day just past and
        # last_shock as that day's return, so the next step of the recursion is
        # the forecast for tomorrow - which is the number the position needs.
        forward = (
            state.omega
            + state.alpha * state.last_shock**2
            + state.beta * state.last_variance
        )
        values.append(np.sqrt(forward) / SCALE)
        dates.append(returns.index[position])

    return pd.Series(values, index=pd.DatetimeIndex(dates), name="conditional_volatility")


def volatility_target_position(
    volatility: pd.Series,
    *,
    target_annual_volatility: float = 0.60,
    max_leverage: float = 1.0,
    min_position: float = 0.0,
) -> pd.Series:
    """Target position from a volatility forecast.

    `target_annual_volatility` defaults to 60%, which is deliberately high: BTC
    has run at 60-100% annualised for most of its history, so a conventional
    15% target would hold a few percent of the asset almost always and the
    result would say more about the cap than about the method.

    `max_leverage` defaults to 1.0 - no borrowing. The formula would happily
    ask for 3x in a calm week, and a backtest that allows it is testing a
    different, much riskier strategy than the one most readers will assume.
    """
    volatility = pd.Series(volatility).astype(float)
    annualised = volatility * np.sqrt(TRADING_DAYS)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = target_annual_volatility / annualised
    return raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(min_position, max_leverage)


def apply_rebalance_band(position: pd.Series, band: float = 0.10) -> pd.Series:
    """Hold the current size until the target drifts more than `band` away.

    Vol targeting retrades on every wobble in the forecast, and on this data
    that costs more than the sizing is worth: 8.9 times annual turnover, 24.5%
    of capital in fees and slippage, against a strategy whose whole benefit is
    a smoother ride. The band is the standard answer - move only when the gap
    between what is held and what is wanted is big enough to be worth paying
    for.

    A band is not free. It leaves the position wrong by up to `band` most of
    the time, so it trades accuracy of sizing for cost of sizing, and whether
    that is a good trade is a question for the backtest rather than for taste.
    """
    target = pd.Series(position).astype(float)
    if band <= 0:
        return target

    held = np.empty(len(target))
    current = float(target.iloc[0]) if len(target) else 0.0
    for i, wanted in enumerate(target.to_numpy()):
        if abs(wanted - current) > band:
            current = float(wanted)
        held[i] = current
    return pd.Series(held, index=target.index, name=target.name)


def realised_volatility(returns: pd.Series, span: int = 30) -> pd.Series:
    """Trailing realised volatility - the cheap alternative, for comparison.

    Included so the GARCH version has to earn its cost. If an exponentially
    weighted standard deviation sizes positions just as well, the honest
    conclusion is to use that instead, and this makes the comparison runnable
    rather than a matter of opinion.
    """
    returns = pd.Series(returns).astype(float)
    return returns.ewm(span=span, min_periods=span).std().shift(1)
