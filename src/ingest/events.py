"""Rejestr zdarzen historycznych - recznie prowadzony CSV.

Zdarzenia sa jedynym zbiorem w projekcie, ktory nie ma zrodla API. Plik
data/raw/events.csv jest wersjonowany i celowo krotki: kazdy wiersz to
zdarzenie, ktore w momencie wystapienia bylo publicznie znane.

Kolumny:
    name           - identyfikator (unikalny razem z data),
    date           - dzien zdarzenia (UTC),
    category       - halving | regulation | macro | market_structure | credit_event,
    description    - jedno zdanie kontekstu,
    available_from - kiedy rynek sie o tym dowiedzial (domyslnie = date),
    source         - link lub odnosnik do zrodla; UZUPELNIJ przed wnioskowaniem.

Ostrzezenie metodologiczne: lista zdarzen wybranych po fakcie jest z natury
obciazona (pamietamy te, po ktorych cos sie stalo). Wyniki event study na
recznej liscie traktuj jako hipotezy, nie dowody - dlatego
analysis/event_study.py zawsze raportuje przedzial ufnosci i liczbe zdarzen.
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
        raise ValueError(f"{path}: brakuje kolumn {missing}")

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
            f"{path}: nieznane kategorie {unknown}; dozwolone: {sorted(VALID_CATEGORIES)}"
        )
    duplicates = df.duplicated(subset=["name", "date"])
    if duplicates.any():
        raise ValueError(f"{path}: zduplikowane zdarzenia: {df.loc[duplicates, 'name'].tolist()}")
    if (df["available_from"] < df["date"]).any():
        raise ValueError(f"{path}: available_from wczesniejsze niz date")
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
