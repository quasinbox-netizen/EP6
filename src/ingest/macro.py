"""Fetching macro data with publication dates under control.

Three channels:

* FRED (needs a free FRED_API_KEY) - M2, industrial production, the Fed funds
  rate. We fetch FIRST RELEASES (output_type=4), so every observation carries
  the date it actually entered circulation rather than a revised value known
  only today. That distinction decides whether a backtest is honest - the
  revised M2 figure for March 2020 was not known until 2021.
* Yahoo Finance (no key) - market series (DXY, SPX, yields). Known the same day
  after the close, hence available_from = date + 1.
* Manual CSV files in data/raw/manual/ - for data without a free API, such as
  ISM PMI, whose licence does not permit distribution through FRED.

A note on scale: the Yahoo yield series (^TNX, ^IRX) have used different
quoting conventions over the years. The features built in
features/macro_phase.py use changes and z-scores, which are scale-invariant.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import secret
from ingest.http import FetchError, get_json
from storage import log_ingest, upsert_macro

MACRO_COLUMNS = ["date", "value", "available_from"]

# FRED returns these instead of numbers when an observation is missing.
_FRED_MISSING = {".", "", None}

# The date FRED uses as "the beginning of time" in vintage queries. It also
# comes back in responses for series with no vintage history in ALFRED.
FRED_NO_VINTAGE_SENTINEL = "1776-07-04"

# Fragment of the FRED message when a series has more vintages than JSON allows.
TOO_MANY_VINTAGES = "maximum number of vintage dates"


class MissingCredentials(RuntimeError):
    """A required API key is missing."""


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=MACRO_COLUMNS)


def parse_fred_observations(
    observations: list[dict],
    *,
    publication_lag_days: int = 30,
    use_vintages: bool = True,
    max_extra_lag_days: int = 120,
) -> pd.DataFrame:
    """Turn a FRED response into a series carrying first-publication dates.

    Split out of `fetch_fred` so it can be tested without network or key.

    Two cases where `realtime_start` is NOT the publication date:

    1. The 1776-07-04 sentinel - a series with no vintage history in ALFRED.
    2. An absurdly long lag (longer than `publication_lag_days +
       max_extra_lag_days`). This happens for two reasons and both must be
       rejected the same way: the observation is older than the vintage archive
       (ALFRED keeps M2SL vintages from roughly the 1990s, so a 1960
       observation receives the archive start date - which does not mean we
       learned about 1960 M2 thirty-five years later), or the series went
       through a methodology revision and the entry comes from a later
       recalculation.

    In both cases we fall back to the fixed publication lag from the config.
    One hard rule always applies: publication cannot precede observation.
    """
    rows = []
    for obs in observations:
        raw_value = obs.get("value")
        if raw_value in _FRED_MISSING:
            continue
        date = pd.Timestamp(obs["date"]).normalize()
        fallback = date + pd.Timedelta(days=publication_lag_days)
        realtime = obs.get("realtime_start")

        usable = (
            use_vintages
            and realtime
            and realtime != FRED_NO_VINTAGE_SENTINEL
        )
        if usable:
            vintage = pd.Timestamp(realtime).normalize()
            implausible = (vintage - date).days > publication_lag_days + max_extra_lag_days
            available = fallback if implausible else vintage
            source = "fallback" if implausible else "vintage"
        else:
            available = fallback
            source = "fallback"

        rows.append(
            {
                "date": date,
                "value": float(raw_value),
                "available_from": max(date, available),
                "lag_source": source,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return _empty()
    df = df.sort_values("date").drop_duplicates(subset="date", keep="first")
    df.attrs["vintage_rows"] = int((df["lag_source"] == "vintage").sum())
    df.attrs["fallback_rows"] = int((df["lag_source"] == "fallback").sum())
    return df.reset_index(drop=True)


def fetch_fred(
    series_id: str,
    *,
    publication_lag_days: int = 30,
    use_vintages: bool = True,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch a FRED series together with its first-publication dates.

    `use_vintages=True` asks for first releases (output_type=4); realtime_start
    is then the date the value entered circulation. For the edge cases see
    `parse_fred_observations`.
    """
    key = api_key or secret("FRED_API_KEY")
    if not key:
        raise MissingCredentials(
            "FRED_API_KEY is missing - copy .env.example to .env and paste a free key "
            "from https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    base = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={key}&file_type=json"
    )
    vintage_params = (
        f"&realtime_start={FRED_NO_VINTAGE_SENTINEL}"
        "&realtime_end=9999-12-31&output_type=4"
    )

    vintages_used = use_vintages
    try:
        payload = get_json(base + vintage_params if use_vintages else base)
    except FetchError as exc:
        if not (use_vintages and TOO_MANY_VINTAGES in str(exc)):
            raise
        # FRED serves first releases only up to 2000 vintage dates. The series
        # exceeding that limit are the DAILY ones (DFF has over 5000) - exactly
        # those published the next day and essentially never revised. For them
        # the fixed lag from the config is faithful, so we fetch the plain
        # series instead of dropping it.
        vintages_used = False
        payload = get_json(base)

    observations = payload.get("observations", [])
    if not observations:
        raise FetchError(f"fred: empty response for {series_id}")
    series = parse_fred_observations(
        observations,
        publication_lag_days=publication_lag_days,
        use_vintages=vintages_used,
    )
    series.attrs["vintages_requested"] = bool(use_vintages)
    series.attrs["vintages_available"] = bool(vintages_used)
    return series


def fetch_yahoo_series(ticker: str, start: str = "2009-01-01") -> pd.DataFrame:
    """Daily closes of a market series. Known after the close, hence D+1."""
    period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    period2 = int(datetime.now(timezone.utc).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )
    payload = get_json(url)
    result = payload.get("chart", {}).get("result")
    if not result:
        raise FetchError(f"yahoo: no data for {ticker}")
    block = result[0]
    closes = block["indicators"]["quote"][0]["close"]
    rows = [
        {
            "date": pd.Timestamp(datetime.fromtimestamp(int(ts), tz=timezone.utc).date()),
            "value": float(close),
        }
        for ts, close in zip(block["timestamp"], closes)
        if close is not None
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return _empty()
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
    df["available_from"] = df["date"] + pd.Timedelta(days=1)
    return df.reset_index(drop=True)


def load_manual_csv(path: str | Path, default_lag_days: int = 30) -> pd.DataFrame:
    """Load a hand-maintained series.

    Format: observation_date,value[,available_from]. Without an
    available_from column we assume the fixed publication lag.
    """
    df = pd.read_csv(path)
    columns = {c.lower().strip(): c for c in df.columns}
    date_col = columns.get("observation_date") or columns.get("date")
    value_col = columns.get("value")
    if not date_col or not value_col:
        raise ValueError(
            f"{path}: columns observation_date (or date) and value are required"
        )
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col]).dt.normalize(),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
        }
    ).dropna(subset=["value"])
    if "available_from" in columns:
        out["available_from"] = pd.to_datetime(df[columns["available_from"]]).dt.normalize()
    else:
        out["available_from"] = out["date"] + pd.Timedelta(days=default_lag_days)
    out["available_from"] = out[["date", "available_from"]].max(axis=1)
    return out.sort_values("date").reset_index(drop=True)


def store_macro(
    conn: sqlite3.Connection, df: pd.DataFrame, series: str, source: str
) -> int:
    if df.empty:
        log_ingest(conn, "macro", f"{series}:{source}", 0, status="empty")
        return 0
    out = df.copy()
    out["series"] = series
    out["source"] = source
    rows = upsert_macro(conn, out)
    log_ingest(
        conn,
        "macro",
        f"{series}:{source}",
        rows,
        first_date=str(out["date"].min().date()),
        last_date=str(out["date"].max().date()),
    )
    return rows
