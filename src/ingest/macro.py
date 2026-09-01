"""Pobieranie danych makro z kontrola daty publikacji.

Trzy kanaly:

* FRED (wymaga darmowego klucza FRED_API_KEY) - M2, produkcja przemyslowa,
  stopa Fed. Pobieramy PIERWSZE PUBLIKACJE (output_type=4), wiec kazda
  obserwacja ma prawdziwa date wejscia do obiegu, a nie zrewidowana wartosc
  znana dopiero dzis. To rozroznienie decyduje o tym, czy backtest jest
  uczciwy - zrewidowane M2 dla marca 2020 poznalismy w 2021 r.
* Yahoo Finance (bez klucza) - serie rynkowe (DXY, SPX, rentownosci).
  Znane tego samego dnia po zamknieciu, wiec available_from = date + 1.
* Reczne CSV w data/raw/manual/ - dla danych bez darmowego API
  (np. ISM PMI, ktorego licencja nie pozwala na dystrybucje przez FRED).

Uwaga o skali: serie rentownosci z Yahoo (^TNX, ^IRX) miewaly rozne
konwencje kwotowania na przestrzeni lat. Cechy budowane w
features/macro_phase.py uzywaja zmian i z-score, ktore sa odporne na skale.
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

# FRED zwraca te wartosci zamiast liczb, gdy obserwacja nie istnieje.
_FRED_MISSING = {".", "", None}

# Data, ktorej FRED uzywa jako "poczatek czasu" w zapytaniach o wersje.
# Wraca tez w odpowiedzi dla serii bez historii wersji w ALFRED.
FRED_NO_VINTAGE_SENTINEL = "1776-07-04"

# Fragment komunikatu FRED, gdy seria ma wiecej wersji, niz format JSON udzwignie.
TOO_MANY_VINTAGES = "maximum number of vintage dates"


class MissingCredentials(RuntimeError):
    """Brakuje klucza API wymaganego przez zrodlo."""


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=MACRO_COLUMNS)


def parse_fred_observations(
    observations: list[dict],
    *,
    publication_lag_days: int = 30,
    use_vintages: bool = True,
    max_extra_lag_days: int = 120,
) -> pd.DataFrame:
    """Zamienia odpowiedz FRED na szereg z data pierwszej publikacji.

    Wydzielone z `fetch_fred`, zeby dalo sie testowac bez sieci i bez klucza.

    Dwa przypadki, w ktorych `realtime_start` NIE jest data publikacji:

    1. Sentinel 1776-07-04 - seria bez historii wersji w ALFRED.
    2. Opoznienie absurdalnie duze (dluzsze niz `publication_lag_days +
       max_extra_lag_days`). Zdarza sie to z dwoch powodow i oba trzeba
       odrzucic tak samo: obserwacja jest starsza niz archiwum wersji
       (ALFRED trzyma wersje M2SL mniej wiecej od lat 90., wiec obserwacja
       z 1960 r. dostaje date poczatku archiwum - to nie znaczy, ze o M2
       z 1960 r. dowiedzielismy sie 35 lat pozniej), albo seria przeszla
       rewizje metodologii i wpis pochodzi z pozniejszego przeliczenia.

    W obu wracamy do stalego opoznienia publikacji z konfiguracji.
    Zawsze obowiazuje twarda zasada: publikacja nie moze poprzedzac obserwacji.
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
    """Pobiera serie z FRED wraz z data pierwszej publikacji.

    `use_vintages=True` prosi o pierwsze publikacje (output_type=4); pole
    realtime_start jest wtedy data wejscia danej do obiegu. Szczegoly
    obslugi przypadkow brzegowych: `parse_fred_observations`.
    """
    key = api_key or secret("FRED_API_KEY")
    if not key:
        raise MissingCredentials(
            "brak FRED_API_KEY - skopiuj .env.example do .env i wklej darmowy klucz "
            "z https://fred.stlouisfed.org/docs/api/api_key.html"
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
        # FRED oddaje pierwsze publikacje tylko do 2000 dat wersji. Przekraczaja
        # ten limit serie DZIENNE (DFF ma ich ponad 5000) - czyli dokladnie te,
        # ktore publikowane sa nastepnego dnia i praktycznie nie sa rewidowane.
        # Dla nich stale opoznienie z konfiguracji jest wierne, wiec pobieramy
        # zwykly szereg zamiast rezygnowac z serii.
        vintages_used = False
        payload = get_json(base)

    observations = payload.get("observations", [])
    if not observations:
        raise FetchError(f"fred: pusta odpowiedz dla {series_id}")
    series = parse_fred_observations(
        observations,
        publication_lag_days=publication_lag_days,
        use_vintages=vintages_used,
    )
    series.attrs["vintages_requested"] = bool(use_vintages)
    series.attrs["vintages_available"] = bool(vintages_used)
    return series


def fetch_yahoo_series(ticker: str, start: str = "2009-01-01") -> pd.DataFrame:
    """Dzienne zamkniecia serii rynkowej. Dostepne po zamknieciu, wiec D+1."""
    period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    period2 = int(datetime.now(timezone.utc).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )
    payload = get_json(url)
    result = payload.get("chart", {}).get("result")
    if not result:
        raise FetchError(f"yahoo: brak danych dla {ticker}")
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
    """Wczytuje recznie prowadzona serie.

    Format: observation_date,value[,available_from]. Bez kolumny
    available_from zakladamy stale opoznienie publikacji.
    """
    df = pd.read_csv(path)
    columns = {c.lower().strip(): c for c in df.columns}
    date_col = columns.get("observation_date") or columns.get("date")
    value_col = columns.get("value")
    if not date_col or not value_col:
        raise ValueError(
            f"{path}: wymagane kolumny observation_date (lub date) oraz value"
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
