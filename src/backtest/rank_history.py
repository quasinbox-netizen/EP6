"""Where the two constants placed in their own sweeps, run after run.

The sweeps argue that ranking these constants on this sample measures nothing.
They argue it with statistics computed inside one run - the spread between
neighbouring values, the standard error of a single Sharpe - and a reader is
entitled to find that unconvincing, because it is one snapshot reasoning about
its own reliability.

This is the same claim made by measurement instead. Every run records where the
value in use placed, and the record is kept. If the ranking meant something, a
value would sit in roughly the same place as days are added to the end of an
eleven-year sample. It does not: the band went from the middle of its grid to
the bottom of it when the price stitch changed underneath it, and that is one
observation of a series this file exists to accumulate.

WHAT THE KEY IS, AND WHY. A row is identified by the day it was RECORDED, not
by the last day of data it used. Those come apart exactly when it matters: two
runs over the same price window can disagree, because the window itself can be
restated underneath them - which is what happened when the stitch began
preferring one exchange's history to another's. Keying on the recording date
keeps both answers, in the order they were believed, and that disagreement is
the most informative thing the log can hold. Re-running on the same day
replaces that day's row, because a day should contribute one observation rather
than one per time somebody typed the command.

Rows are never removed. A log that drops its embarrassing entries is a log that
proves nothing, and the embarrassing entries are the evidence here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = [
    "recorded", "as_of", "parameter", "adopted", "sharpe", "rank", "of",
    "best", "best_sharpe", "days", "source",
]

# How a row got here. `run` is the normal case: a sweep computed it. `published`
# marks a figure taken from a page this project had already put in public -
# real, checkable, and not produced by the recorder, which is a difference the
# reader should be able to see rather than take on trust.
SOURCES = ("run", "published")

# Below this many distinct days there is no wandering to show, only points.
MIN_OBSERVATIONS = 3


def load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path).reindex(columns=COLUMNS)


def record(
    path: str | Path,
    sweep,
    *,
    as_of,
    recorded,
    source: str = "run",
) -> pd.DataFrame:
    """Add one observation, replacing today's row for the same parameter."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    path = Path(path)
    history = load(path)
    row = {
        "recorded": str(pd.Timestamp(recorded).date()),
        "as_of": str(pd.Timestamp(as_of).date()),
        "parameter": sweep.parameter,
        "adopted": round(float(sweep.adopted), 4),
        "sharpe": round(float(sweep.sharpe.loc[sweep.adopted]), 6),
        "rank": int(sweep.rank),
        "of": int(len(sweep.table)),
        "best": round(float(sweep.best_band), 4),
        "best_sharpe": round(float(sweep.sharpe.max()), 6),
        "days": int(sweep.table["days"].loc[sweep.adopted]),
        "source": source,
    }
    same_day = (
        (history["recorded"] == row["recorded"])
        & (history["parameter"] == row["parameter"])
    ) if not history.empty else pd.Series(dtype=bool)
    if not history.empty:
        history = history.loc[~same_day]
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    history = history.sort_values(["recorded", "parameter"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(path, index=False)
    return history


def placement(history: pd.DataFrame) -> pd.DataFrame:
    """Rank as a share of its own grid, so two grids of different size compare.

    A rank of 34 means one thing out of 61 and another out of 39. The share is
    what the chart plots; the raw rank travels alongside it so a reader is
    never shown a ratio without the counts that made it.
    """
    if history.empty:
        return history.assign(placement=pd.Series(dtype=float))
    frame = history.copy()
    frame["placement"] = frame["rank"].astype(float) / frame["of"].astype(float)
    return frame


def movement(history: pd.DataFrame, parameter: str) -> dict | None:
    """How far this parameter's placing has travelled, or None if too few days.

    Returns nothing rather than a shrug when there is not enough to look at. A
    chart of two points asserting that something wanders would be claiming from
    noise the same way the sweeps say not to.
    """
    rows = history.loc[history["parameter"] == parameter]
    if rows["recorded"].nunique() < MIN_OBSERVATIONS:
        return None
    ranked = rows.sort_values("recorded")
    first, last = ranked.iloc[0], ranked.iloc[-1]
    return {
        "observations": int(rows["recorded"].nunique()),
        "from": str(ranked["recorded"].iloc[0]),
        "to": str(ranked["recorded"].iloc[-1]),
        "best_rank": int(ranked["rank"].min()),
        "worst_rank": int(ranked["rank"].max()),
        "of": int(last["of"]),
        "first_rank": int(first["rank"]),
        "last_rank": int(last["rank"]),
        "best_value_moved": int(ranked["best"].nunique()),
    }
