"""Disclosure: the corrections log, and the provenance of the fitted constants.

These tests are unusual in that they are not about arithmetic. They exist
because the project shipped a seven-point restatement of its own headline with
no notice on the page, and described two constants it fitted on this dataset as
though they came from outside it. Neither failure was a bug; both were things
the code had no reason to keep doing and no reason to stop.

So the reasons live here. A correction removed from the log, or a provenance
note quietly dropped from the page, fails a test rather than passing review.
"""
from __future__ import annotations

import pandas as pd
import pytest

from publish.disclosure import (
    CORRECTIONS, Correction, corrections_for, corrections_html, provenance_html,
    sweep_html, target_sweep_html,
)


@pytest.fixture
def sizing() -> pd.DataFrame:
    return pd.DataFrame([{
        "as_of": "2026-09-03", "price": 81270.37,
        "forecast_annual_volatility": 0.5428, "median_annual_volatility": 0.5995,
        "target_position": 1.0, "target_annual_volatility": 0.60, "band": 0.30,
        "position": 0.816,
    }])


@pytest.fixture
def comparison() -> pd.DataFrame:
    return pd.DataFrame([
        {"strategy": "buy and hold", "sharpe": 1.1356},
        {"strategy": "vol target, band 0%", "sharpe": 1.0950},
        {"strategy": "vol target, band 30%", "sharpe": 1.1316},
        {"strategy": "vol target, EWMA band 10%", "sharpe": 1.1527},
    ])


def test_the_lag_restatement_is_on_the_record():
    """The correction that started this file.

    The published figure moved by 6.8 points when the double execution lag was
    fixed. If this entry ever disappears, the page goes back to showing a
    number that improved for reasons the reader cannot see.
    """
    entries = corrections_for("trader.html")

    assert any("-14.5%" in item.was and "-7.7%" in item.now for item in entries)


def test_a_correction_carries_the_number_it_replaces(sizing):
    """A correction without the old figure cannot be checked by anyone."""
    for item in CORRECTIONS:
        assert item.was and item.now and item.was != item.now
        assert len(item.why) > 80          # a reason, not a shrug


def test_the_notice_prints_both_figures():
    html = corrections_html("trader.html")

    assert "-14.5%" in html and "-7.7%" in html
    assert "2026-09-07" in html


def test_a_page_with_nothing_to_correct_says_nothing():
    """An empty corrections box would read as an admission on every page."""
    assert corrections_html("index.html") == ""


def test_the_corrections_log_is_append_only():
    """Frozen records, so an entry cannot be softened in place.

    Rewriting history here would be worse than never having disclosed: it
    would leave a log that looks complete and is not.
    """
    with pytest.raises(Exception):
        CORRECTIONS[0].was = "something kinder"

    assert isinstance(CORRECTIONS, tuple)
    assert all(isinstance(item, Correction) for item in CORRECTIONS)


def test_both_constants_are_named_as_fitted(sizing, comparison):
    """The point of the section: neither number came from outside the data."""
    html = provenance_html(sizing, comparison)

    assert "chosen with this dataset in front of us" in html
    assert "60%" in html and "59.95%" in html      # the target, and the median it is
    assert "30%" in html
    assert "selection in sample" in html


def test_the_variant_that_beats_the_adopted_one_is_named(sizing, comparison):
    """The reason for publishing the whole comparison rather than a verdict.

    The adopted band is not the best row in its own table. A reader who is
    only shown the winner cannot tell the difference between a value that won
    and a value that was kept.
    """
    html = provenance_html(sizing, comparison)

    assert "EWMA band 10%" in html
    assert "1.1527" in html


def test_nothing_is_claimed_when_the_run_did_not_record_it(comparison):
    """The target is read out of the saved row, never written into the page."""
    assert provenance_html(pd.DataFrame(), comparison) == ""

    older = pd.DataFrame([{"median_annual_volatility": 0.5995, "band": 0.30}])
    html = provenance_html(older, comparison)

    assert "59.95%" in html                        # what the run did record
    assert "The volatility target</b>" in html    # named, but not given a figure


def test_a_comparison_the_adopted_row_wins_names_no_rival(sizing):
    swept = pd.DataFrame([
        {"strategy": "buy and hold", "sharpe": 9.0},          # never a rival
        {"strategy": "vol target, band 30%", "sharpe": 1.2},
        {"strategy": "vol target, band 10%", "sharpe": 1.1},
    ])
    html = provenance_html(sizing, swept)

    assert "scores higher" not in html


@pytest.fixture
def sweep() -> pd.DataFrame:
    """A grid whose best value is far from the adopted one, and a frozen tail."""
    return pd.DataFrame([
        {"band": 0.00, "sharpe": 1.0950, "n_position_changes": 900, "days": 4035},
        {"band": 0.30, "sharpe": 1.1316, "n_position_changes": 40, "days": 4035},
        {"band": 0.42, "sharpe": 1.1886, "n_position_changes": 12, "days": 4035},
        {"band": 0.52, "sharpe": 1.1356, "n_position_changes": 1, "days": 4035},
    ])


def test_the_sweep_refuses_its_own_best_value(sweep):
    """The section must not read as a recommendation to move the band.

    A published argmax is an invitation, and the whole point of the sweep is
    that this particular argmax is worth nothing. So the refusal is part of the
    text and is pinned here.
    """
    html = sweep_html(sweep, adopted=0.30)

    assert "Nothing here was changed in response" in html
    assert "precisely the mistake the sweep exposes" in html
    assert "turnover is a cost, not a score" in html


def test_the_sweep_prices_the_band_actually_in_use(sweep):
    """It reported the sweep's FIRST row as the band in use, which was 0%.

    Subtracting the adopted value from the index and taking idxmin returns the
    difference rather than the label, so the page confidently described a band
    the portfolio does not use, with that band's score attached.
    """
    html = sweep_html(sweep, adopted=0.30)

    assert "The band in use (30%) scores <b>1.1316</b>" in html
    assert "The best (42%) scores <b>1.1886</b>" in html
    assert "it ranks <b>3</b>" in html          # two of the four score higher


def test_the_frozen_tail_is_named_as_such(sweep):
    html = sweep_html(sweep, adopted=0.30)

    assert "From <b>52%</b>" in html
    assert "inherits buy-and-hold's Sharpe by construction" in html


def test_no_sweep_no_section():
    assert sweep_html(pd.DataFrame(), adopted=0.30) == ""


@pytest.fixture
def target_grid() -> pd.DataFrame:
    return pd.DataFrame([
        {"target": 0.30, "sharpe": 1.2151, "at_the_cap": 0.0136,
         "max_drawdown": -0.4870, "days": 4035},
        {"target": 0.60, "sharpe": 1.1316, "at_the_cap": 0.5016,
         "max_drawdown": -0.6638, "days": 4035},
        {"target": 1.60, "sharpe": 1.1172, "at_the_cap": 0.9901,
         "max_drawdown": -0.8319, "days": 4035},
    ])


def test_the_target_section_says_the_score_is_the_wrong_criterion(target_grid):
    """The band's problem was noise; this one's is a category error.

    Sharpe is built to be indifferent to position size, and position size is
    the only thing the target sets. Ranking it on Sharpe is not a close call
    made badly - it is the wrong measurement, and the section has to say so
    rather than repeat the band's argument.
    """
    html = target_sweep_html(target_grid, adopted=0.60)

    assert "dial on how much risk to carry" in html
    assert "indifferent to how large a position is" in html
    assert "The target did not move either" in html


def test_the_section_publishes_the_share_of_days_at_the_cap(target_grid):
    """The number that was missing from the site entirely."""
    html = target_sweep_html(target_grid, adopted=0.60)

    assert "On 50.2% of days the position is pinned" in html
    assert "From a target of <b>160%</b>" in html


def test_the_drawdown_range_is_given_as_what_the_target_chooses(target_grid):
    html = target_sweep_html(target_grid, adopted=0.60)

    assert "-49% to -83%" in html
    assert "at the target in use it is -66%" in html


def test_a_grid_without_the_cap_column_claims_nothing_about_it(target_grid):
    html = target_sweep_html(target_grid.drop(columns=["at_the_cap"]), adopted=0.60)

    assert "pinned at the" not in html
    assert "dial on how much risk to carry" in html      # the rest still stands


def test_no_target_sweep_no_section():
    assert target_sweep_html(pd.DataFrame(), adopted=0.60) == ""


def test_the_turnover_claim_is_read_from_the_sweep_not_written_into_the_page():
    """It said "roughly 8.9x to 1.1x" in prose, and the data had moved to 8.75x.

    Harmless on the day it was written and wrong within a week of the daily
    refresh being switched on, which is the shape of every stale number this
    module exists to prevent. The figures now come out of the same table the
    chart is drawn from.
    """
    grid = pd.DataFrame([
        {"band": 0.00, "sharpe": 1.0, "turnover_annual": 12.5, "n_position_changes": 900},
        {"band": 0.30, "sharpe": 1.1, "turnover_annual": 2.4, "n_position_changes": 40},
    ])
    html = sweep_html(grid, adopted=0.30)

    assert "from 12.5x, with no band at all, to 2.4x" in html
    assert "8.9x" not in html


def test_a_sweep_without_turnover_still_defends_the_band():
    grid = pd.DataFrame([
        {"band": 0.00, "sharpe": 1.0, "n_position_changes": 900},
        {"band": 0.30, "sharpe": 1.1, "n_position_changes": 40},
    ])
    html = sweep_html(grid, adopted=0.30)

    assert "it cuts turnover sharply" in html
    assert "turnover is a cost, not a score" in html
