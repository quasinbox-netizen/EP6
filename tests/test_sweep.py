"""The band sweep, and the statistics that stop it being read as a shopping list.

The danger in publishing a sweep is that a reader - or a later contributor -
takes the argmax and adopts it. Every test here pins something that makes that
harder: the adopted band is always priced, the noise measure is stated next to
the ranking, and the region where the rule stops trading is identified rather
than left to look like skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestConfig
from backtest.sweep import DEFAULT_BANDS, DEFAULT_TARGETS, band_sweep, target_sweep


@pytest.fixture(scope="module")
def close() -> pd.Series:
    """A wandering price with enough days for the metrics to be defined."""
    index = pd.date_range("2018-01-01", periods=900, freq="D")
    steps = np.random.default_rng(11).normal(0.0005, 0.03, len(index))
    return pd.Series(1000.0 * np.exp(np.cumsum(steps)), index=index, name="close")


@pytest.fixture(scope="module")
def target(close) -> pd.Series:
    """An unbanded position that moves every day, so the band has work to do."""
    wobble = pd.Series(
        np.random.default_rng(12).uniform(0.2, 1.0, len(close)), index=close.index)
    return wobble.rolling(5).mean().bfill()


@pytest.fixture(scope="module")
def sweep(close, target):
    return band_sweep(close, target, BacktestConfig(), adopted=0.30,
                      bands=tuple(round(b, 4) for b in np.arange(0.0, 0.61, 0.05)))


def test_the_band_in_use_is_always_priced(close, target):
    """A sweep that cannot price the value in use answers a different question.

    The adopted band need not land on the grid - `--band 0.07` against a grid
    of five-point steps does not - and the row for it must exist anyway, or the
    page has to report the nearest neighbour as though it were the real one.
    """
    off_grid = band_sweep(close, target, BacktestConfig(), adopted=0.07,
                          bands=(0.0, 0.10, 0.30))

    assert 0.07 in off_grid.table.index
    assert off_grid.rank >= 1


def test_the_rank_counts_from_the_best(sweep):
    ranks = {
        band: int((sweep.sharpe > sweep.sharpe.loc[band]).sum()) + 1
        for band in sweep.table.index
    }

    assert ranks[sweep.best_band] == 1
    assert sweep.rank == ranks[sweep.adopted]


def test_the_noise_is_reported_beside_the_ranking(sweep):
    """The ranking is only publishable because this number is published with it."""
    assert sweep.standard_error > 0
    assert "standard error" in sweep.summary()
    assert f"{sweep.rank} of {len(sweep.table)}" in sweep.summary()


def test_the_standard_error_is_the_optimistic_one(sweep):
    """Lo's iid formula, stated as such because it understates the real spread.

    Using the optimistic figure makes the "differences are small against the
    noise" argument harder to make, not easier. A wider estimate would flatter
    the conclusion, which is the wrong direction for a claim like this one.
    """
    days = int(sweep.table["days"].loc[sweep.adopted])
    expected = np.sqrt((1 + float(sweep.sharpe.loc[sweep.adopted]) ** 2 / 2) / days)

    assert sweep.standard_error == pytest.approx(expected)


def test_a_band_wider_than_the_position_ever_moves_is_flagged(close, target):
    """The plateau on the right of the chart, named rather than left to impress.

    Past that width the weight never changes again, so the "strategy" is a
    constant holding wearing a strategy's name - and a constant holding has
    buy-and-hold's Sharpe by construction. Unflagged, that plateau reads as the
    sweep finding something.
    """
    wide = band_sweep(close, target, BacktestConfig(), adopted=0.30,
                      bands=(0.0, 0.30, 0.95))

    assert wide.frozen_from is not None
    assert wide.frozen_from <= 0.95
    frozen = wide.table.loc[wide.frozen_from]
    assert frozen["n_position_changes"] <= 1
    assert "stops trading" in wide.summary()


def test_the_default_grid_reaches_past_the_point_of_stopping():
    """Cropping the grid before the plateau would hide why scores rise."""
    assert min(DEFAULT_BANDS) == 0.0
    assert max(DEFAULT_BANDS) >= 0.60
    assert len(DEFAULT_BANDS) == 61


def test_every_row_differs_by_the_band_and_nothing_else(sweep):
    assert sweep.table.index.is_monotonic_increasing
    assert sweep.table.index.is_unique
    assert sweep.table["days"].nunique() == 1


@pytest.fixture(scope="module")
def volatility(close) -> pd.Series:
    """A forecast that spends part of the history calm and part of it violent.

    The calm stretch matters: it is where the formula asks for more than the
    no-borrowing cap allows, which is the regime the target sweep is about.
    """
    daily = pd.Series(
        np.random.default_rng(13).uniform(0.01, 0.06, len(close)), index=close.index)
    return daily.rolling(10).mean().bfill()


@pytest.fixture(scope="module")
def targets(close, volatility):
    return target_sweep(close, volatility, BacktestConfig(), adopted=0.60, band=0.30,
                        targets=tuple(round(t, 4) for t in np.arange(0.10, 2.01, 0.10)))


def test_the_target_sweep_records_where_the_position_is_pinned(targets):
    """The column the site was missing, and the reason this sweep exists.

    A target that spends most days at the no-borrowing cap is not sizing
    anything on those days - it wants more than it may hold, so it holds
    everything and stops listening to the forecast. Sharpe cannot show that
    and drawdown only hints at it; this counts it.
    """
    assert "at_the_cap" in targets.table.columns
    assert targets.at_the_cap is not None
    assert 0.0 <= targets.at_the_cap <= 1.0
    # More target means more days wanting more than the cap allows.
    assert targets.table["at_the_cap"].is_monotonic_increasing


def test_a_target_large_enough_becomes_buy_and_hold(targets):
    """Named, because otherwise the flat right-hand end reads as a result."""
    assert targets.pinned_from is not None
    pinned = targets.table.loc[targets.pinned_from]

    assert pinned["at_the_cap"] >= 0.99
    assert "IS buy-and-hold" in targets.summary()


def test_the_band_sweep_has_no_cap_column_and_says_nothing_about_it(sweep):
    """A band sweep does not vary the cap, so it must not claim to measure it."""
    assert sweep.at_the_cap is None
    assert sweep.pinned_from is None
    assert "no-borrowing cap" not in sweep.summary()


def test_the_summary_names_the_parameter_it_swept(sweep, targets):
    assert "bands from" in sweep.summary()
    assert "targets from" in targets.summary()


def test_the_target_in_use_is_priced_even_off_the_grid(close, volatility):
    off_grid = target_sweep(close, volatility, BacktestConfig(), adopted=0.63,
                            band=0.30, targets=(0.30, 0.60, 1.20))

    assert 0.63 in off_grid.table.index


def test_the_target_grid_runs_past_both_degenerate_ends():
    """Stopping at the defensible values would show a curve without its limits."""
    assert min(DEFAULT_TARGETS) <= 0.10
    assert max(DEFAULT_TARGETS) >= 2.00
    assert 0.60 in DEFAULT_TARGETS          # the value in use is always priced
