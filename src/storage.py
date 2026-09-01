"""Lokalna baza SQLite: surowe ceny, makro, zdarzenia i log pobran.

Kontrakt czasowy - kluczowy dla calego projektu:

* `date`           - data OBSERWACJI (dzien, ktorego dotyczy wartosc),
* `available_from` - pierwszy dzien, w ktorym wartosc byla publicznie znana.

Dla cen roznica wynosi jeden dzien (bar dzienny D jest kompletny dopiero
o 00:00 UTC dnia D+1). Dla makro potrafi wynosic tygodnie - M2 za dany
miesiac publikowane jest ~30 dni po jego zakonczeniu. Kazde zapytanie
analityczne filtruje po `available_from`, nigdy po `date`; to jedyne
miejsce, w ktorym zapobiegamy look-ahead bias na poziomie danych.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol         TEXT NOT NULL,
    source         TEXT NOT NULL,
    date           TEXT NOT NULL,
    open           REAL,
    high           REAL,
    low            REAL,
    close          REAL NOT NULL,
    volume         REAL,
    available_from TEXT NOT NULL,
    ingested_at    TEXT NOT NULL,
    PRIMARY KEY (symbol, source, date)
);

CREATE TABLE IF NOT EXISTS macro (
    series         TEXT NOT NULL,
    source         TEXT NOT NULL,
    date           TEXT NOT NULL,
    value          REAL,
    available_from TEXT NOT NULL,
    ingested_at    TEXT NOT NULL,
    PRIMARY KEY (series, source, date)
);

CREATE TABLE IF NOT EXISTS events (
    name           TEXT NOT NULL,
    date           TEXT NOT NULL,
    category       TEXT NOT NULL,
    description    TEXT,
    available_from TEXT NOT NULL,
    source         TEXT,
    PRIMARY KEY (name, date)
);

CREATE TABLE IF NOT EXISTS ingest_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name  TEXT NOT NULL,
    key         TEXT NOT NULL,
    rows        INTEGER NOT NULL,
    first_date  TEXT,
    last_date   TEXT,
    status      TEXT NOT NULL,
    detail      TEXT,
    ingested_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);
CREATE INDEX IF NOT EXISTS idx_macro_avail ON macro(series, available_from);
"""

PRICE_COLUMNS = (
    "symbol", "source", "date", "open", "high", "low", "close",
    "volume", "available_from", "ingested_at",
)
MACRO_COLUMNS = ("series", "source", "date", "value", "available_from", "ingested_at")
EVENT_COLUMNS = ("name", "date", "category", "description", "available_from", "source")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _placeholders(n: int) -> str:
    return ",".join(["?"] * n)


def _rows_from_frame(df: pd.DataFrame, columns: Sequence[str]) -> list[tuple]:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError("brakuje kolumn: " + ", ".join(missing))
    out = df.loc[:, list(columns)].copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return [tuple(None if pd.isna(v) else v for v in row)
            for row in out.itertuples(index=False, name=None)]


def upsert_prices(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Wstawia/aktualizuje bary. Idempotentne - ponowne pobranie nie duplikuje."""
    if df.empty:
        return 0
    df = df.copy()
    df["ingested_at"] = utc_now()
    rows = _rows_from_frame(df, PRICE_COLUMNS)
    conn.executemany(
        "INSERT INTO prices (" + ",".join(PRICE_COLUMNS) + ") "
        "VALUES (" + _placeholders(len(PRICE_COLUMNS)) + ") "
        "ON CONFLICT(symbol, source, date) DO UPDATE SET "
        "open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, "
        "volume=excluded.volume, available_from=excluded.available_from, "
        "ingested_at=excluded.ingested_at",
        rows,
    )
    return len(rows)


def upsert_macro(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["ingested_at"] = utc_now()
    rows = _rows_from_frame(df, MACRO_COLUMNS)
    conn.executemany(
        "INSERT INTO macro (" + ",".join(MACRO_COLUMNS) + ") "
        "VALUES (" + _placeholders(len(MACRO_COLUMNS)) + ") "
        "ON CONFLICT(series, source, date) DO UPDATE SET "
        "value=excluded.value, available_from=excluded.available_from, "
        "ingested_at=excluded.ingested_at",
        rows,
    )
    return len(rows)


def upsert_events(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = _rows_from_frame(df, EVENT_COLUMNS)
    conn.executemany(
        "INSERT INTO events (" + ",".join(EVENT_COLUMNS) + ") "
        "VALUES (" + _placeholders(len(EVENT_COLUMNS)) + ") "
        "ON CONFLICT(name, date) DO UPDATE SET "
        "category=excluded.category, description=excluded.description, "
        "available_from=excluded.available_from, source=excluded.source",
        rows,
    )
    return len(rows)


def log_ingest(
    conn: sqlite3.Connection,
    table_name: str,
    key: str,
    rows: int,
    first_date: str | None = None,
    last_date: str | None = None,
    status: str = "ok",
    detail: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO ingest_log "
        "(table_name, key, rows, first_date, last_date, status, detail, ingested_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (table_name, key, rows, first_date, last_date, status, detail, utc_now()),
    )


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("date", "available_from"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def read_prices(
    conn: sqlite3.Connection,
    symbol: str,
    source: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM prices WHERE symbol = ?"
    params: list[object] = [symbol]
    if source:
        query += " AND source = ?"
        params.append(source)
    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)
    df = pd.read_sql_query(query + " ORDER BY date", conn, params=params)
    return _parse_dates(df) if not df.empty else df


def read_macro(
    conn: sqlite3.Connection,
    series: str | Iterable[str] | None = None,
    as_of: str | None = None,
) -> pd.DataFrame:
    """Odczyt makro. `as_of` filtruje po dacie PUBLIKACJI, nie obserwacji.

    Dzieki temu read_macro(conn, as_of="2020-03-01") zwraca dokladnie ten
    zbior informacji, jaki byl dostepny 1 marca 2020 - takze dla serii
    publikowanych z opoznieniem.
    """
    query = "SELECT * FROM macro WHERE 1=1"
    params: list[object] = []
    if series is not None:
        names = [series] if isinstance(series, str) else list(series)
        query += " AND series IN (" + _placeholders(len(names)) + ")"
        params.extend(names)
    if as_of:
        query += " AND available_from <= ?"
        params.append(as_of)
    df = pd.read_sql_query(query + " ORDER BY series, date", conn, params=params)
    return _parse_dates(df) if not df.empty else df


def read_events(
    conn: sqlite3.Connection,
    category: str | None = None,
    as_of: str | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM events WHERE 1=1"
    params: list[object] = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if as_of:
        query += " AND available_from <= ?"
        params.append(as_of)
    df = pd.read_sql_query(query + " ORDER BY date", conn, params=params)
    return _parse_dates(df) if not df.empty else df


def table_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Krotki przeglad zawartosci bazy - uzywany przez CLI i dashboard."""
    rows = []
    # Ceny grupujemy po symbolu I zrodle - inaczej wszystkie aktywa kontrolne
    # zlewaja sie w jeden wiersz "yahoo" i nie widac, czy ktoregos brakuje.
    for table, key in (
        ("prices", "symbol || ':' || source"),
        ("macro", "series"),
        ("events", "category"),
    ):
        query = (
            "SELECT " + key + " AS key, COUNT(*) AS rows, MIN(date) AS first_date, "
            "MAX(date) AS last_date FROM " + table + " GROUP BY " + key
        )
        part = pd.read_sql_query(query, conn)
        part.insert(0, "table", table)
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
