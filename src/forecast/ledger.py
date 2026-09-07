"""What the app said, on the day it said it, and what happened next.

Every other number in this project is a backtest: the rule is applied to data
that already exists, and the analyst who wrote the rule had seen it. That is
unavoidable and the reason the whole validation directory exists. This module
is the one thing that escapes it - a claim written down before its outcome
existed cannot be tuned to that outcome afterwards.

Three rules hold the escape open:

1. The file is APPEND-ONLY in meaning. A claim is stored with the date it was
   made and the last close it was made from, and is never edited afterwards.
   Re-recording the same (as_of, horizon, claim, level) replaces the row
   rather than adding a second one - running the command twice on the same
   data is not two predictions - but a claim whose `as_of` has passed is
   frozen.

2. Scoring is NOT stored. `score()` joins the ledger against today's prices
   every time it is asked, so there is no cached verdict to drift away from
   the data. A stored score is a number nobody re-checks.

3. A row is scored only when its target date has arrived. Partial credit for
   a window still open is how a track record becomes a marketing document.

The scoreboard compares each claim against the baseline it has to beat:
intervals against the coverage they promised, direction calls against the
base rate of the sample (Bitcoin rose in most windows, so 50% is not the bar).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CLAIM_INTERVAL = "interval"
CLAIM_DIRECTION = "direction"

COLUMNS = [
    "made_on", "as_of", "horizon", "target_date", "price_at_origin",
    "claim", "level", "low", "high", "p_up", "n_effective", "note",
]
KEY = ["as_of", "horizon", "claim", "level"]


def drop_forming_bar(close: pd.Series, *, today: pd.Timestamp | None = None):
    """Cut the bar for today, which has not finished forming.

    A claim has to be anchored to a price that will still mean the same thing
    tomorrow. Today's daily bar does not: it moves until the day closes, so
    two runs a few hours apart record different intervals for what is supposed
    to be the same forecast, and neither is the settled close the coverage
    walk was calibrated on.

    Returns (series, dropped_day) so the caller can say what it ignored rather
    than quietly shifting the origin by a day.
    """
    close = pd.Series(close).sort_index()
    if close.empty:
        return close, None
    cutoff = pd.Timestamp(today) if today is not None else pd.Timestamp(
        datetime.now(timezone.utc).date()
    )
    if close.index[-1] < cutoff:
        return close, None
    return close.loc[close.index < cutoff], close.index[-1]


def empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return empty_ledger()
    frame = pd.read_csv(path)
    for column in ("made_on", "as_of", "target_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
    return frame.reindex(columns=COLUMNS)


def build_entries(
    *,
    as_of: pd.Timestamp,
    price: float,
    horizon: int,
    intervals: pd.DataFrame | None = None,
    p_up: float | None = None,
    n_effective: int | None = None,
    note: str = "",
    made_on: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Turn today's forecast into rows that can be checked later.

    `intervals` is the frame `price_interval` produces - one row per level
    that passed its coverage test. Levels that were withheld are not recorded:
    the ledger holds claims that were made, and a withheld level was not one.
    """
    as_of = pd.Timestamp(as_of).normalize()
    made_on = pd.Timestamp(made_on or pd.Timestamp.today()).normalize()
    target = as_of + pd.Timedelta(days=horizon)

    rows = []
    if intervals is not None and not intervals.empty:
        for row in intervals.itertuples():
            rows.append({
                "made_on": made_on, "as_of": as_of, "horizon": horizon,
                "target_date": target, "price_at_origin": float(price),
                "claim": CLAIM_INTERVAL, "level": float(row.level),
                "low": float(row.low), "high": float(row.high),
                "p_up": np.nan, "n_effective": np.nan, "note": note,
            })
    if p_up is not None and np.isfinite(p_up):
        rows.append({
            "made_on": made_on, "as_of": as_of, "horizon": horizon,
            "target_date": target, "price_at_origin": float(price),
            "claim": CLAIM_DIRECTION, "level": np.nan,
            "low": np.nan, "high": np.nan, "p_up": float(p_up),
            "n_effective": np.nan if n_effective is None else int(n_effective),
            "note": note,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def append(path: str | Path, entries: pd.DataFrame) -> pd.DataFrame:
    """Add today's claims, replacing any recorded for the same origin.

    Replacement rather than merge, for the same reason `store_events` replaces:
    the key identifies one claim, and two rows for it would be silently
    averaged by everything downstream.
    """
    path = Path(path)
    ledger = load(path)
    if entries.empty:
        return ledger

    combined = pd.concat([ledger, entries], ignore_index=True)
    # keep="last" so the newly built rows win over anything already stored
    # for the same origin.
    combined = combined.drop_duplicates(subset=KEY, keep="last")
    combined = combined.sort_values(["as_of", "horizon", "claim", "level"])
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def score(ledger: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Attach the outcome to every claim whose target date has arrived.

    The price on the target date is taken with a backward fill of at most a
    few days, because a target can land on a day the series does not have.
    Rows without an outcome stay in the frame with empty results - dropping
    them would hide how much of the record is still open.
    """
    if ledger.empty:
        return ledger.assign(price_at_target=[], realised=[], hit=[], matured=[])

    close = close.astype(float).sort_index()
    scored = ledger.copy()
    targets = pd.DatetimeIndex(scored["target_date"])
    positions = close.index.searchsorted(targets)
    outcome, matured = [], []
    for target, position in zip(targets, positions):
        if position >= len(close):
            outcome.append(np.nan)
            matured.append(False)
            continue
        found = close.index[position]
        # searchsorted lands on the first day at or after the target; a gap of
        # more than a week means the series simply has no outcome there.
        usable = (found - target).days <= 7
        outcome.append(float(close.iloc[position]) if usable else np.nan)
        matured.append(bool(usable))

    scored["price_at_target"] = outcome
    scored["matured"] = matured
    scored["realised"] = scored["price_at_target"] / scored["price_at_origin"] - 1.0
    scored["hit"] = np.where(
        scored["matured"] & (scored["claim"] == CLAIM_INTERVAL),
        (scored["price_at_target"] >= scored["low"])
        & (scored["price_at_target"] <= scored["high"]),
        np.nan,
    )
    return scored


def scoreboard(scored: pd.DataFrame, *, base_rate: float | None = None) -> pd.DataFrame:
    """One row per promise, with what it promised and what it delivered.

    Intervals are scored against their own level. Direction calls are scored
    with the Brier score against the base rate rather than against 50%: in a
    series that rose in most windows, beating a coin flip is not evidence of
    anything.
    """
    if scored.empty:
        return pd.DataFrame(columns=["claim", "horizon", "level", "n", "promised",
                                     "delivered", "brier", "baseline_brier"])

    matured = scored[scored["matured"].astype(bool)]
    rows = []
    for (claim, horizon, level), group in matured.groupby(
        ["claim", "horizon", "level"], dropna=False
    ):
        row = {
            "claim": claim, "horizon": int(horizon),
            "level": float(level) if pd.notna(level) else np.nan,
            "n": int(len(group)),
        }
        if claim == CLAIM_INTERVAL:
            row["promised"] = float(level)
            row["delivered"] = float(group["hit"].astype(float).mean())
            row["brier"] = np.nan
            row["baseline_brier"] = np.nan
        else:
            went_up = (group["realised"] > 0).astype(float)
            row["promised"] = float(group["p_up"].mean())
            row["delivered"] = float(went_up.mean())
            row["brier"] = float(((group["p_up"] - went_up) ** 2).mean())
            row["baseline_brier"] = (
                float(((base_rate - went_up) ** 2).mean())
                if base_rate is not None else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(board: pd.DataFrame, open_claims: int) -> str:
    """One line, phrased so a thin record cannot be read as a good one."""
    if board.empty:
        return (
            f"Nothing has matured yet: {open_claims} claims are still open. "
            "A track record starts the day the first one is settled, not the "
            "day it is written."
        )
    settled = int(board["n"].sum())
    if settled < 20:
        return (
            f"{settled} settled claims, {open_claims} still open. Too few to "
            "read as a record - this is a start, not a score."
        )
    intervals = board[board["claim"] == CLAIM_INTERVAL]
    kept = int((intervals["delivered"] >= intervals["promised"] - 0.1).sum())
    return (
        f"{settled} settled claims, {open_claims} open. "
        f"{kept} of {len(intervals)} interval levels are within 10 points of "
        "what they promised."
    )
