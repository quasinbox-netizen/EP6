"""The record of where each constant placed, run after run.

This log exists to make an argument the sweeps can only assert. A sweep says
"ranking these constants measures nothing" using statistics computed inside one
run, and a reader is entitled to distrust a snapshot reasoning about its own
reliability. The log says the same thing by keeping the ranking and watching it
move, which needs no standard error to believe.

That only works if the log cannot be tidied. Every test here defends one way it
could quietly become flattering: losing an old row, gaining a duplicate,
drawing a line through too few points, or presenting a ratio without the counts
behind it.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from backtest.rank_history import (
    MIN_OBSERVATIONS, load, movement, placement, record,
)


@dataclass
class FakeSweep:
    """The three things `record` reads off a Sweep, without running one."""

    parameter: str = "band"
    adopted: float = 0.30
    rank_value: int = 34
    size: int = 61
    best_value: float = 0.42
    score: float = 1.1316

    @property
    def sharpe(self):
        return pd.Series({self.adopted: self.score, self.best_value: 1.1886})

    @property
    def best_band(self):
        return self.best_value

    @property
    def rank(self):
        return self.rank_value

    @property
    def table(self):
        # The filler labels start at 1.0 so they cannot collide with `adopted`.
        # A real sweep indexes on a sorted set and cannot repeat a value; a
        # fixture that did would hand `.loc` a Series and fail for a reason
        # that has nothing to do with what is being tested.
        return pd.DataFrame(
            {"days": [4035] * self.size},
            index=pd.Index([self.adopted] + [1.0 + i for i in range(self.size - 1)]),
        )


@pytest.fixture
def log(tmp_path):
    return tmp_path / "rank_history.csv"


def test_one_row_per_parameter_per_day(log):
    record(log, FakeSweep(), as_of="2026-09-07", recorded="2026-09-07")
    record(log, FakeSweep(parameter="target", adopted=0.60, rank_value=14, size=39),
           as_of="2026-09-07", recorded="2026-09-07")
    history = load(log)

    assert len(history) == 2
    assert set(history["parameter"]) == {"band", "target"}


def test_running_twice_in_a_day_replaces_rather_than_duplicates(log):
    """A day contributes one observation, not one per time somebody typed it.

    Without this a morning of debugging would stack ten identical points on
    the chart and make a flat stretch look like a busy one.
    """
    record(log, FakeSweep(rank_value=34), as_of="2026-09-07", recorded="2026-09-07")
    record(log, FakeSweep(rank_value=61), as_of="2026-09-07", recorded="2026-09-07")
    history = load(log)

    assert len(history) == 1
    assert int(history.iloc[0]["rank"]) == 61        # the later answer stands


def test_a_new_day_never_disturbs_an_older_one(log):
    """The whole point: yesterday's answer survives today's.

    A log that overwrote history would show a constant sitting exactly where it
    was put, every day, and prove the opposite of what happened.
    """
    record(log, FakeSweep(rank_value=34), as_of="2026-09-07", recorded="2026-09-07")
    record(log, FakeSweep(rank_value=61), as_of="2026-09-08", recorded="2026-09-08")
    history = load(log)

    assert list(history["rank"].astype(int)) == [34, 61]
    assert list(history["recorded"]) == ["2026-09-07", "2026-09-08"]


def test_a_figure_taken_from_a_page_is_labelled_as_one(log):
    """Recorded and reconstructed are different claims and must look different."""
    record(log, FakeSweep(), as_of="2026-09-03", recorded="2026-09-07",
           source="published")

    assert load(log).iloc[0]["source"] == "published"

    with pytest.raises(ValueError):
        record(log, FakeSweep(), as_of="2026-09-03", recorded="2026-09-09",
               source="looked about right")


def test_the_share_of_the_grid_carries_the_counts_that_made_it(log):
    """34th of 61 and 34th of 39 are different facts wearing the same number."""
    record(log, FakeSweep(rank_value=34, size=61),
           as_of="2026-09-07", recorded="2026-09-07")
    shares = placement(load(log))

    assert shares.iloc[0]["placement"] == pytest.approx(34 / 61)
    assert {"rank", "of"} <= set(shares.columns)


def test_movement_says_nothing_until_there_is_movement_to_see(log):
    """Two points asserting a wander would be the sin the sweeps warn against."""
    for day in range(MIN_OBSERVATIONS - 1):
        record(log, FakeSweep(rank_value=30 + day),
               as_of=f"2026-09-0{day + 1}", recorded=f"2026-09-0{day + 1}")

    assert movement(load(log), "band") is None


def test_movement_reports_the_span_the_rank_has_covered(log):
    for day, rank in enumerate((34, 12, 61), start=1):
        record(log, FakeSweep(rank_value=rank, best_value=0.40 + day / 100),
               as_of=f"2026-09-0{day}", recorded=f"2026-09-0{day}")
    travel = movement(load(log), "band")

    assert travel["observations"] == 3
    assert (travel["best_rank"], travel["worst_rank"]) == (12, 61)
    assert (travel["first_rank"], travel["last_rank"]) == (34, 61)
    assert travel["best_value_moved"] == 3      # a different winner every run


def test_an_absent_log_reads_as_empty_rather_than_raising(tmp_path):
    history = load(tmp_path / "never_written.csv")

    assert history.empty
    assert movement(history, "band") is None
