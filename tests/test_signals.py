"""The signal desk: reading a position series back as trades and triggers.

Every test here is a way the reading can be wrong while the backtest stays
right. Dating a trade one day early, counting an open position as a win, or
quoting a crossover level that does not actually cross - none of those touch
the equity curve, and all of them change what someone does on a Monday.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestConfig, run_backtest
from backtest.signals import next_trigger, report, trade_log, trend_flip_level
from config import REPO_ROOT
from features.halving import CONFIRMED_HALVINGS
from pipeline import LabData, strategy_signals
from validation.synthetic import random_walk_prices

DESK = REPO_ROOT / "dashboard" / "signal_desk.html"
APP = REPO_ROOT / "dashboard" / "app.py"


@pytest.fixture
def close() -> pd.Series:
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    return pd.Series([100.0, 110.0, 120.0, 130.0, 140.0, 150.0], index=index, name="close")


# --- reading the trades --------------------------------------------------------


def test_trade_dates_come_from_the_executed_positions(close):
    """The entry is the day the position exists, not the day the rule decided.

    `run_backtest` has already applied the execution lag, so a change in the
    series it returns is the trade. Re-shifting here would date every entry a
    day early and hand the strategy a return it never earned.
    """
    positions = pd.Series([0, 0, 1, 1, 0, 0], index=close.index, dtype=float)
    log = trade_log(positions, close)

    assert len(log) == 1
    assert log.loc[0, "entry_date"] == close.index[2]
    assert log.loc[0, "exit_date"] == close.index[3]
    assert log.loc[0, "entry_price"] == 120.0
    assert log.loc[0, "return_gross"] == pytest.approx(130.0 / 120.0 - 1)


def test_open_trade_is_kept_and_flagged(close):
    positions = pd.Series([0, 0, 0, 1, 1, 1], index=close.index, dtype=float)
    log = trade_log(positions, close)

    assert len(log) == 1
    assert bool(log.loc[0, "open"]) is True
    assert log.loc[0, "exit_price"] == 150.0  # marked at the last close


def test_costs_are_charged_on_both_sides(close):
    positions = pd.Series([0, 1, 1, 0, 0, 0], index=close.index, dtype=float)
    log = trade_log(positions, close, cost_rate=0.0025)

    round_trip = log.loc[0, "return_gross"] - log.loc[0, "return_net"]
    assert round_trip == pytest.approx(2 * 0.0025)


def test_win_rate_ignores_the_open_position(close):
    """An open trade has not been paid for yet.

    Letting it into the win rate would make whatever is running right now
    count as a result, and it flatters exactly the number a reader is most
    tempted by.
    """
    positions = pd.Series([1, 0, 0, 1, 1, 1], index=close.index, dtype=float)
    result = run_backtest(close, positions.shift(-1).fillna(0.0), BacktestConfig(
        fee_bps=0, slippage_bps=0, execution_lag_days=1, initial_equity=1.0))
    signal = report(result, close)

    assert signal.side == "long"
    assert signal.closed_trades == len(signal.trades[~signal.trades["open"]])
    assert signal.closed_trades < len(signal.trades)


# --- the trigger ---------------------------------------------------------------


def test_flip_level_actually_makes_the_averages_meet():
    """The level is checked by using it, not by re-deriving the algebra.

    Append the quoted price to the series and the two moving averages have to
    land on the same number; anything else means the formula is quoting a
    crossing that would not happen.
    """
    prices = random_walk_prices(400, start="2020-01-01", seed=5)
    close = prices.set_index("date")["close"]
    level = trend_flip_level(close, fast=50, slow=200)
    assert level is not None

    extended = pd.concat([close, pd.Series([level], index=[close.index[-1] + pd.Timedelta(days=1)])])
    fast = extended.rolling(50).mean().iloc[-1]
    slow = extended.rolling(200).mean().iloc[-1]
    assert fast == pytest.approx(slow, rel=1e-9)


def test_an_unreachable_level_is_reported_as_unreachable():
    """A negative level means no price crosses tomorrow, not a missing answer.

    Quoting it as a dollar figure would put a negative trigger on the panel;
    swallowing it as "not enough history" would claim the wrong reason. The
    seed here produces exactly that case with a 50/200 pair.
    """
    close = random_walk_prices(400, start="2020-01-01", seed=11).set_index("date")["close"]
    assert trend_flip_level(close, fast=50, slow=200) < 0

    trigger = next_trigger({"kind": "trend", "fast": 50, "slow": 200}, close, "long")
    assert trigger.kind == "none"
    assert trigger.level is None
    assert "no close can cross" in trigger.text


def test_flip_level_needs_a_full_slow_window():
    short = pd.Series(np.arange(50.0), index=pd.date_range("2020-01-01", periods=50))
    assert trend_flip_level(short, fast=50, slow=200) is None


def test_halving_trigger_gives_the_exit_date_while_holding():
    last = pd.Timestamp(CONFIRMED_HALVINGS[-1]) + pd.Timedelta(days=10)
    close = pd.Series([1.0], index=[last])
    trigger = next_trigger({"kind": "halving", "days_after": 365}, close, "long", last_date=last)

    assert trigger.kind == "date"
    assert trigger.date == pd.Timestamp(CONFIRMED_HALVINGS[-1]) + pd.Timedelta(days=365)


def test_halving_trigger_refuses_to_date_an_unconfirmed_halving():
    """The rule uses confirmed halvings, so out of the window it has no date.

    Filling one in would mean the panel predicting a halving the research side
    of the project deliberately refuses to assume.
    """
    last = pd.Timestamp(CONFIRMED_HALVINGS[-1]) + pd.Timedelta(days=2000)
    close = pd.Series([1.0], index=[last])
    trigger = next_trigger({"kind": "halving", "days_after": 365}, close, "flat", last_date=last)

    assert trigger.date is None


def test_baseline_has_no_trigger():
    close = pd.Series([1.0], index=[pd.Timestamp("2024-01-01")])
    assert next_trigger({"kind": "baseline"}, close, "long").kind == "none"


# --- the bundle the dashboard reads --------------------------------------------


@pytest.fixture
def noise_data() -> LabData:
    prices = random_walk_prices(1500, start="2019-01-01", seed=5)
    features = prices.set_index("date")[["close"]]
    empty = pd.DataFrame()
    return LabData(prices=prices, macro=empty, events=empty, features=features)


def test_every_strategy_reports_a_position_and_a_trigger(noise_data):
    bundle = strategy_signals(noise_data)
    assert bundle["strategies"]

    for item in bundle["strategies"]:
        signal = item["report"]
        assert signal.side in {"long", "flat"}
        assert signal.trigger.kind in {"price", "date", "external", "none"}
        assert len(item["positions"]) == len(noise_data.features)


def test_trend_rule_quotes_a_price_trigger(noise_data):
    bundle = strategy_signals(noise_data)
    trend = next(i["report"] for i in bundle["strategies"] if i["report"].name == "trend 50/200")

    assert trend.trigger.kind == "price"
    assert trend.trigger.level > 0
    assert trend.trigger.direction in {"above", "below"}


def test_signals_agree_with_the_backtest_tab(noise_data):
    """Both come from `run_strategies`, and this is what keeps them there.

    If the desk ever grew its own copy of the strategies, the panel and the
    equity curve could disagree while both looked right.
    """
    from pipeline import run_strategies

    table, _ = run_strategies(noise_data)
    bundle = strategy_signals(noise_data)
    names = [item["report"].name for item in bundle["strategies"]]

    assert names == list(table.index)
    for item in bundle["strategies"]:
        assert item["report"].metrics["sharpe"] == pytest.approx(
            table.loc[item["report"].name, "sharpe"], nan_ok=True
        )


# --- the panel file ------------------------------------------------------------


def test_desk_markers_match_the_dashboard():
    markup = DESK.read_text(encoding="utf-8")
    source = APP.read_text(encoding="utf-8")
    for marker in ("<!--DESK:START-->", "<!--DESK:END-->",
                   "<!--DATA:START-->", "<!--DATA:END-->"):
        assert markup.count(marker) == 1
        assert marker in source


def test_desk_ships_valid_placeholder_data():
    """The file has to open on its own, which means the demo payload parses."""
    markup = DESK.read_text(encoding="utf-8")
    demo = markup.split("<!--DATA:START-->", 1)[1].split("<!--DATA:END-->", 1)[0]
    parsed = json.loads(demo)

    assert parsed["strategies"] == []  # a placeholder, not a fixture with numbers


def test_desk_says_what_it_is_not():
    """The panel is a signal light; the label is not decoration.

    A page that shows entries, exits and a live price reads as advice unless
    it says otherwise, and this project's own result is that the rules have no
    edge to advise from.
    """
    markup = DESK.read_text(encoding="utf-8").lower()
    assert "not advice" in markup
