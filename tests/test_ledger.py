"""The prediction ledger - the only record here that is not a backtest.

What these tests protect is not arithmetic. It is the property that makes the
ledger worth keeping at all: a claim is written before its outcome exists, is
never edited afterwards, and is scored only once its window has actually
closed. Every test below is one way that property could be lost while the
numbers still looked fine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.ledger import (
    CLAIM_DIRECTION,
    CLAIM_INTERVAL,
    append,
    build_entries as build,
    drop_forming_bar,
    load,
    score,
    scoreboard,
    verdict,
)


@pytest.fixture
def close() -> pd.Series:
    """A year of daily prices, rising steadily from 100 to about 200."""
    index = pd.date_range("2025-01-01", periods=365, freq="D")
    return pd.Series(np.linspace(100.0, 200.0, 365), index=index, name="close")


@pytest.fixture
def intervals() -> pd.DataFrame:
    return pd.DataFrame([
        {"level": 0.5, "low": 95.0, "high": 115.0},
        {"level": 0.9, "low": 80.0, "high": 140.0},
    ])


# --- writing -------------------------------------------------------------------


def test_one_row_per_level_plus_the_direction_call(intervals):
    entries = build(
        as_of=pd.Timestamp("2025-01-10"), price=100.0, horizon=30,
        intervals=intervals, p_up=0.6, n_effective=12,
    )

    assert len(entries) == 3
    assert (entries["target_date"] == pd.Timestamp("2025-02-09")).all()
    assert set(entries["claim"]) == {CLAIM_INTERVAL, CLAIM_DIRECTION}


def test_recording_twice_on_the_same_data_replaces_rather_than_duplicates(tmp_path, intervals):
    """Running the command daily must not inflate the record with copies.

    The key is (as_of, horizon, claim, level): a forecast that has not changed
    because the data has not changed is one claim, not two.
    """
    path = tmp_path / "predictions.csv"
    entries = build(as_of=pd.Timestamp("2025-01-10"), price=100.0, horizon=30,
                    intervals=intervals)

    append(path, entries)
    stored = append(path, entries)

    assert len(stored) == 2
    assert len(load(path)) == 2


def test_a_new_origin_adds_rows(tmp_path, intervals):
    path = tmp_path / "predictions.csv"
    append(path, build(as_of=pd.Timestamp("2025-01-10"), price=100.0,
                       horizon=30, intervals=intervals))
    append(path, build(as_of=pd.Timestamp("2025-01-11"), price=101.0,
                       horizon=30, intervals=intervals))

    assert len(load(path)) == 4


# --- scoring -------------------------------------------------------------------


def test_an_open_claim_is_not_scored(close):
    """Partial credit for a window still running is how a record becomes a pitch."""
    entries = build(as_of=close.index[-5], price=float(close.iloc[-5]),
                    horizon=30, intervals=pd.DataFrame(
                        [{"level": 0.5, "low": 1.0, "high": 1e9}]))
    scored = score(entries, close)

    assert not bool(scored["matured"].iloc[0])
    assert pd.isna(scored["hit"].iloc[0])
    assert pd.isna(scored["realised"].iloc[0])


def test_a_matured_interval_is_scored_against_the_price_on_its_target_date(close):
    origin = close.index[0]
    target_price = float(close.loc[origin + pd.Timedelta(days=30)])
    entries = build(
        as_of=origin, price=float(close.iloc[0]), horizon=30,
        intervals=pd.DataFrame([
            {"level": 0.5, "low": target_price - 1, "high": target_price + 1},
            {"level": 0.9, "low": 0.0, "high": target_price - 5},
        ]),
    )
    scored = score(entries, close)

    assert bool(scored["matured"].all())
    assert bool(scored.loc[scored["level"] == 0.5, "hit"].iloc[0])
    assert not bool(scored.loc[scored["level"] == 0.9, "hit"].iloc[0])


def test_scoring_never_reaches_back_before_the_target(close):
    """The outcome is the first price at or after the target, never before it.

    Taking the nearest price instead would let a claim be settled by a day
    inside its own window - which is a backtest wearing the ledger's clothes.
    """
    origin = close.index[0]
    entries = build(as_of=origin, price=float(close.iloc[0]), horizon=30,
                    intervals=pd.DataFrame([{"level": 0.5, "low": 0.0, "high": 1e9}]))
    scored = score(entries, close)

    target = origin + pd.Timedelta(days=30)
    assert scored["price_at_target"].iloc[0] == pytest.approx(float(close.loc[target]))
    assert scored["price_at_target"].iloc[0] > float(close.loc[target - pd.Timedelta(days=1)])


def test_a_target_far_past_the_series_stays_open(close):
    entries = build(as_of=close.index[-1], price=float(close.iloc[-1]), horizon=365,
                    intervals=pd.DataFrame([{"level": 0.5, "low": 0.0, "high": 1e9}]))
    scored = score(entries, close)

    assert not bool(scored["matured"].iloc[0])


# --- the scoreboard ------------------------------------------------------------


def test_interval_scoreboard_compares_delivery_with_the_promise(close):
    rows = []
    for offset in range(0, 200, 40):
        origin = close.index[offset]
        target_price = float(close.loc[origin + pd.Timedelta(days=30)])
        # Half the claims are honest, half are deliberately too narrow.
        wide = offset % 80 == 0
        rows.append(build(
            as_of=origin, price=float(close.iloc[offset]), horizon=30,
            intervals=pd.DataFrame([{
                "level": 0.5,
                "low": target_price - (5 if wide else 100),
                "high": target_price + (5 if wide else 50),
            }]),
        ))
    board = scoreboard(score(pd.concat(rows, ignore_index=True), close))

    row = board.iloc[0]
    assert row["promised"] == 0.5
    assert 0.0 <= row["delivered"] <= 1.0
    assert row["n"] == 5


def test_direction_is_scored_against_the_base_rate_not_a_coin_flip(close):
    """In a series that rose in most windows, beating 50% proves nothing."""
    entries = pd.concat([
        build(as_of=close.index[offset], price=float(close.iloc[offset]),
              horizon=30, p_up=0.6)
        for offset in range(0, 120, 30)
    ], ignore_index=True)
    board = scoreboard(score(entries, close), base_rate=0.95)

    row = board[board["claim"] == CLAIM_DIRECTION].iloc[0]
    # Every window rose here, so a 0.60 call is worse than a 0.95 baseline.
    assert row["delivered"] == 1.0
    assert row["brier"] > row["baseline_brier"]


def test_verdict_refuses_to_call_a_thin_record_a_record(close):
    entries = build(as_of=close.index[0], price=float(close.iloc[0]), horizon=30,
                    intervals=pd.DataFrame([{"level": 0.5, "low": 0.0, "high": 1e9}]))
    board = scoreboard(score(entries, close))

    assert "not a score" in verdict(board, open_claims=3)


def test_verdict_says_so_when_nothing_has_matured():
    assert "Nothing has matured" in verdict(pd.DataFrame(), open_claims=7)


# --- the origin ----------------------------------------------------------------


def test_todays_unfinished_bar_is_not_an_origin(close):
    """A claim anchored to a price that is still moving is not reproducible.

    Two runs a few hours apart recorded intervals tens of dollars apart before
    this rule existed, for what was supposed to be the same forecast, and
    neither price was the settled close the coverage walk was calibrated on.
    """
    today = close.index[-1]
    trimmed, dropped = drop_forming_bar(close, today=today)

    assert dropped == today
    assert trimmed.index[-1] == close.index[-2]


def test_a_settled_series_is_left_alone(close):
    trimmed, dropped = drop_forming_bar(close, today=close.index[-1] + pd.Timedelta(days=1))

    assert dropped is None
    assert trimmed.index[-1] == close.index[-1]


def test_an_empty_series_is_handled():
    trimmed, dropped = drop_forming_bar(pd.Series(dtype=float))

    assert trimmed.empty
    assert dropped is None
