"""Registry of historical events - a hand-maintained CSV.

Events are the only dataset in this project without an API behind it. The file
data/raw/events.csv is version-controlled and deliberately short: each row is
an event that was publicly known at the moment it happened.

Columns:
    name           - identifier (unique together with the date),
    date           - the day of the event (UTC),
    category       - halving | regulation | macro | market_structure | credit_event,
    description    - one sentence of context,
    available_from - when the market learned about it (defaults to date),
    source         - link or citation; FILL THIS IN before drawing conclusions.

Methodological warning: a list of events chosen after the fact is inherently
biased - we remember the ones that were followed by something. Treat event
study results on a hand-made list as hypotheses, not proof, which is why
analysis/event_study.py always reports a confidence interval and the number of
events.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from storage import log_ingest, upsert_events

REQUIRED_COLUMNS = ["name", "date", "category"]
VALID_CATEGORIES = {
    "halving",
    "regulation",
    "macro",
    "market_structure",
    "credit_event",
    "cycle_extreme",
}


def load_events_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "available_from" in df.columns:
        available = pd.to_datetime(df["available_from"], errors="coerce")
        df["available_from"] = available.fillna(df["date"]).dt.normalize()
    else:
        df["available_from"] = df["date"]
    for column in ("description", "source"):
        if column not in df.columns:
            df[column] = None

    unknown = sorted(set(df["category"]) - VALID_CATEGORIES)
    if unknown:
        raise ValueError(
            f"{path}: unknown categories {unknown}; allowed: {sorted(VALID_CATEGORIES)}"
        )
    duplicates = df.duplicated(subset=["name", "date"])
    if duplicates.any():
        raise ValueError(f"{path}: duplicate events: {df.loc[duplicates, 'name'].tolist()}")
    if (df["available_from"] < df["date"]).any():
        raise ValueError(f"{path}: available_from is earlier than date")
    return df.sort_values("date").reset_index(drop=True)


def store_events(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        log_ingest(conn, "events", "manual", 0, status="empty")
        return 0
    rows = upsert_events(conn, df)
    log_ingest(
        conn,
        "events",
        "manual",
        rows,
        first_date=str(df["date"].min().date()),
        last_date=str(df["date"].max().date()),
    )
    return rows
