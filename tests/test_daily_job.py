"""The scheduled job, whose failures are all silent by construction.

Nobody watches this run. Every mistake it can make looks like a normal morning
from the outside: a stale number republished with today's date on it, a push
that stopped happening, a working tree that quietly fills with regenerated
files. So the properties that keep it honest are pinned here rather than left
to whoever next edits the STEPS tuple.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import daily_record  # noqa: E402


def order() -> list:
    return [name for name, _ in daily_record.STEPS]


def arguments(step: str) -> list:
    return next(args for name, args in daily_record.STEPS if name == step)


def test_the_sizing_step_refreshes_rather_than_re_reading_the_cache():
    """A bare `sizing` on a schedule is a no-op dressed as an update.

    The volatility forecast is cached on disk. Without `--refresh` the command
    reads yesterday's file and prints yesterday's size, however often it runs -
    so scheduling it without the flag would have looked like a fix, published a
    daily timestamp beside an unchanged number, and shown nothing on the page
    to say which day the number was really from.
    """
    assert "sizing" in order()
    assert "--refresh" in arguments("sizing")


def test_the_position_is_recomputed_before_anything_publishes_it():
    """Ordering is the whole correctness property of this file.

    `paper` sizes the portfolio from the volatility forecast and `publish`
    reads `sizing_today.csv`. Both would happily run against the previous
    day's files and say nothing about it.
    """
    steps = order()

    assert steps.index("ingest") < steps.index("sizing")
    assert steps.index("sizing") < steps.index("paper")
    assert steps.index("sizing") < steps.index("publish")
    assert steps[-1] == "publish"


def test_the_sizing_tables_are_committed_with_the_pages_that_show_them():
    """A published chart whose source table is not in the repository.

    The sweeps are charted on the agent page and the CSVs behind them are
    version-controlled so a reader can check the chart without running
    anything. Regenerating them daily without committing them would break that
    promise on day one and leave the tree permanently dirty besides.
    """
    for name in ("sizing_today", "sizing_sweep", "sizing_target_sweep"):
        assert any(name in path for path in daily_record.SIZING_ARTEFACTS)


@pytest.mark.parametrize("path", daily_record.SIZING_ARTEFACTS)
def test_every_staged_path_is_one_git_will_accept(path: str):
    """An ignored path in the staging list breaks the push, not just that file.

    `git add -- <ignored>` exits non-zero, and commit_and_push returns at the
    first failing step - so one forgotten negation in .gitignore would stop the
    site being deployed at all, every morning, with the reason buried in a log.
    """
    finished = subprocess.run(
        ["git", "check-ignore", "--", path],
        cwd=ROOT, capture_output=True, text=True,
    )

    assert finished.returncode != 0, f"{path} is ignored and would break the push"


def test_the_job_stages_named_paths_and_never_the_whole_tree():
    """An unattended `git add -A` would commit whatever was lying around.

    This job pushes to a public repository on a timer. Everything it stages has
    to be something it produced itself and nothing it merely found.
    """
    source = (ROOT / "scripts" / "daily_record.py").read_text(encoding="utf-8")

    assert "add -A" not in source and "add ." not in source
    for path in daily_record.SIZING_ARTEFACTS:
        assert path.startswith("data/processed/")


def test_a_failed_refit_does_not_stop_the_day_being_recorded():
    """Blocking steps are the ones whose absence would misrepresent the day.

    A GARCH refit that fails on a bad morning should leave yesterday's sizing
    in place and let the ledger record anyway. It must not join the set that
    makes the whole job report failure.
    """
    source = (ROOT / "scripts" / "daily_record.py").read_text(encoding="utf-8")
    blocking = source.split("blocking = [", 1)[1].split("]", 1)[0]

    assert "sizing" not in blocking
    assert "ledger" in blocking and "publish" in blocking
