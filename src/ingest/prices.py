"""Pobieranie dziennych barow OHLCV dla BTC/USD.

Cztery niezalezne zrodla bez klucza API:

* binance  - BTC/USDT od 2017-08. Najglebszy rynek i najwiekszy wolumen,
             wiec od 2017 r. to on jest zrodlem prawdy. Nie siega jednak
             halvingow 2012 i 2016 - sama gielda wystartowala w 2017 r.
* bitstamp - BTC/USD od 2011-08. Jedyne z tych zrodel obejmujace pelna
             historie cykli, plytsze, ale ciagle.
* coinbase - BTC/USD od 2015-07, walidacja krzyzowa.
* yahoo    - agregat rynkowy od 2014-09, trzeci glos przy rozbieznosciach.

Stad domyslny szereg jest ZSZYWANY (`stitch_sources`): Bitstamp do
2017-08, dalej Binance. Szew jest jawny, logowany i sprawdzany na
zakladce - a nie ukryty w srodku danych.

Uwaga o parze: Binance kwotuje BTC/USDT, nie BTC/USD. Historycznie roznica
mieszczi sie w ulamku procenta poza epizodami utraty parytetu USDT
(np. pazdziernik 2018) - dlatego zszycie porownuje zakladke i raportuje
odchylenie zamiast je milczaco akceptowac.

Kazdy provider zwraca ten sam kontrakt: DataFrame z kolumnami
[date, open, high, low, close, volume], posortowany rosnaco, bez duplikatow,
z datami w UTC. Zapis do bazy dokleja `available_from = date + 1 dzien`,
bo bar dnia D jest kompletny dopiero po jego zamknieciu.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd

from ingest.http import FetchError, get_json
from storage import log_ingest, upsert_prices

OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

BINANCE_SYMBOLS = {"BTCUSD": "BTCUSDT"}
BITSTAMP_PAIRS = {"BTCUSD": "btcusd"}
COINBASE_PRODUCTS = {"BTCUSD": "BTC-USD"}
YAHOO_TICKERS = {
    "BTCUSD": "BTC-USD",
    # Grupa kontrolna. Halving jest zdarzeniem wylacznie bitcoinowym, wiec
    # to, co NASDAQ robi wokol tych samych dat, jest testem placebo.
    "NASDAQ": "^IXIC",
    "SP500": "^GSPC",
    "GOLD": "GC=F",
}

# Pierwszy dzien z danymi w kazdym zrodle - uzywane przy zszywaniu i w CLI.
SOURCE_START = {
    "binance": "2017-08-17",
    "bitstamp": "2011-08-18",
    "coinbase": "2015-07-20",
    "yahoo": "2014-09-17",
}


def _to_unix(day: str | pd.Timestamp) -> int:
    ts = pd.Timestamp(day)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def _finalize(rows: list[dict], start: str | None, end: str | None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.dropna(subset=["close"])
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.loc[:, OHLCV_COLUMNS].reset_index(drop=True)


def fetch_binance(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Dzienne swiece z Binance (BTC/USDT), stronicowane po 1000 barow.

    Binance zwraca `openTime` w milisekundach; bar dzienny otwiera sie
    o 00:00 UTC, wiec data bara to dzien jego otwarcia.
    """
    pair = BINANCE_SYMBOLS[symbol]
    day_ms = 86_400_000
    cursor = _to_unix(start) * 1000
    stop = (_to_unix(end) if end else int(datetime.now(timezone.utc).timestamp())) * 1000
    rows: list[dict] = []
    seen: set[int] = set()
    while cursor <= stop:
        url = (
            "https://api.binance.com/api/v3/klines"
            f"?symbol={pair}&interval=1d&startTime={cursor}&endTime={stop}&limit=1000"
        )
        chunk = get_json(url, sleep=0.25)
        if not chunk:
            break
        for kline in chunk:
            open_time = int(kline[0])
            if open_time in seen or open_time > stop:
                continue
            seen.add(open_time)
            rows.append(
                {
                    "date": datetime.fromtimestamp(open_time / 1000, tz=timezone.utc).date(),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                }
            )
        last_open = int(chunk[-1][0])
        if last_open < cursor:
            break
        cursor = last_open + day_ms
        if len(chunk) < 1000:
            break
    return _finalize(rows, start, end)


def fetch_bitstamp(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    pair = BITSTAMP_PAIRS[symbol]
    cursor = _to_unix(start)
    stop = _to_unix(end) if end else int(datetime.now(timezone.utc).timestamp())
    day = 86400
    rows: list[dict] = []
    seen: set[int] = set()
    while cursor <= stop:
        url = (
            f"https://www.bitstamp.net/api/v2/ohlc/{pair}/"
            f"?step={day}&limit=1000&start={cursor}"
        )
        payload = get_json(url, sleep=0.3)
        chunk = payload.get("data", {}).get("ohlc", [])
        if not chunk:
            break
        fresh = 0
        for bar in chunk:
            ts = int(bar["timestamp"])
            if ts in seen or ts > stop:
                continue
            seen.add(ts)
            fresh += 1
            rows.append(
                {
                    "date": datetime.fromtimestamp(ts, tz=timezone.utc).date(),
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": float(bar["volume"]),
                }
            )
        last_ts = max(int(b["timestamp"]) for b in chunk)
        if fresh == 0 or last_ts <= cursor:
            break
        cursor = last_ts + day
    return _finalize(rows, start, end)


def fetch_coinbase(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    product = COINBASE_PRODUCTS[symbol]
    day = timedelta(days=1)
    begin = pd.Timestamp(start).normalize()
    finish = pd.Timestamp(end).normalize() if end else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    rows: list[dict] = []
    window = timedelta(days=280)  # limit API to 300 swiec na zapytanie
    cursor = begin
    while cursor <= finish:
        chunk_end = min(cursor + window, finish)
        url = (
            f"https://api.exchange.coinbase.com/products/{product}/candles"
            f"?granularity=86400&start={cursor.date()}&end={chunk_end.date()}"
        )
        payload = get_json(url, sleep=0.4)
        for candle in payload:
            ts, low, high, open_, close, volume = candle
            rows.append(
                {
                    "date": datetime.fromtimestamp(int(ts), tz=timezone.utc).date(),
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": float(volume),
                }
            )
        cursor = chunk_end + day
    return _finalize(rows, start, end)


def fetch_yahoo(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    ticker = YAHOO_TICKERS[symbol]
    period1 = _to_unix(start)
    period2 = _to_unix(end) if end else int(datetime.now(timezone.utc).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )
    payload = get_json(url)
    result = payload.get("chart", {}).get("result")
    if not result:
        raise FetchError(f"yahoo: brak danych dla {ticker}")
    block = result[0]
    quote = block["indicators"]["quote"][0]
    rows = [
        {
            "date": datetime.fromtimestamp(int(ts), tz=timezone.utc).date(),
            "open": quote["open"][i],
            "high": quote["high"][i],
            "low": quote["low"][i],
            "close": quote["close"][i],
            "volume": quote["volume"][i],
        }
        for i, ts in enumerate(block["timestamp"])
    ]
    return _finalize(rows, start, end)


PROVIDERS = {
    "binance": fetch_binance,
    "bitstamp": fetch_bitstamp,
    "coinbase": fetch_coinbase,
    "yahoo": fetch_yahoo,
}


def fetch_prices(source: str, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    if source not in PROVIDERS:
        raise ValueError(f"nieznane zrodlo cen: {source} (dostepne: {sorted(PROVIDERS)})")
    return PROVIDERS[source](symbol, start, end)


def stitch_sources(
    frames: dict[str, pd.DataFrame], priority: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Skleja szereg z kilku gield wedlug priorytetu, dzien po dniu.

    Dla kazdego dnia wygrywa pierwsze zrodlo z listy `priority`, ktore ten
    dzien ma. Zwraca (szereg, raport): raport zawiera liczbe dni z kazdego
    zrodla, daty szwow i mediane rozbieznosci na zakladkach - bez tego
    zszywanie jest cichym zrodlem bledow.
    """
    available = [s for s in priority if s in frames and not frames[s].empty]
    if not available:
        return pd.DataFrame(columns=OHLCV_COLUMNS), pd.DataFrame()

    indexed = {}
    for source in available:
        frame = frames[source].copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        indexed[source] = frame.drop_duplicates(subset="date", keep="last").set_index("date")

    all_days = sorted(set().union(*(set(f.index) for f in indexed.values())))
    chosen_source: list[str] = []
    rows = []
    for day in all_days:
        for source in available:
            frame = indexed[source]
            if day in frame.index:
                row = frame.loc[day, ["open", "high", "low", "close", "volume"]].to_dict()
                row["date"] = day
                rows.append(row)
                chosen_source.append(source)
                break

    stitched = pd.DataFrame(rows).loc[:, OHLCV_COLUMNS].reset_index(drop=True)
    stitched["source_used"] = chosen_source

    report_rows = []
    for source in available:
        used = stitched[stitched["source_used"] == source]
        report_rows.append(
            {
                "source": source,
                "days_used": len(used),
                "first_day_used": None if used.empty else str(used["date"].min().date()),
                "last_day_used": None if used.empty else str(used["date"].max().date()),
            }
        )
    report = pd.DataFrame(report_rows)

    # Rozbieznosc na zakladce z nastepnym zrodlem w kolejnosci priorytetu.
    overlaps = []
    for higher, lower in zip(available, available[1:]):
        common = indexed[higher].index.intersection(indexed[lower].index)
        if len(common) == 0:
            continue
        a = indexed[higher].loc[common, "close"]
        b = indexed[lower].loc[common, "close"]
        rel = ((a - b).abs() / b.abs())
        overlaps.append(
            {
                "pair": f"{higher} vs {lower}",
                "overlap_days": int(len(common)),
                "median_rel_diff": float(rel.median()),
                "max_rel_diff": float(rel.max()),
                "worst_day": str(rel.idxmax().date()),
            }
        )
    overlap_report = pd.DataFrame(overlaps)
    report.attrs["overlap"] = overlap_report
    return stitched, report


def load_stitched(
    conn: sqlite3.Connection, symbol: str, priority: list[str]
) -> pd.DataFrame:
    """Zszyty szereg zbudowany z tego, co juz jest w bazie."""
    from storage import read_prices

    frames = {}
    for source in priority:
        frame = read_prices(conn, symbol, source=source)
        if not frame.empty:
            frames[source] = frame
    stitched, _ = stitch_sources(frames, priority)
    return stitched


def store_prices(
    conn: sqlite3.Connection, df: pd.DataFrame, symbol: str, source: str
) -> int:
    """Zapisuje bary, doklejajac metadane i data dostepnosci (D+1)."""
    if df.empty:
        log_ingest(conn, "prices", f"{symbol}:{source}", 0, status="empty")
        return 0
    out = df.copy()
    out["symbol"] = symbol
    out["source"] = source
    out["available_from"] = out["date"] + pd.Timedelta(days=1)
    rows = upsert_prices(conn, out)
    log_ingest(
        conn,
        "prices",
        f"{symbol}:{source}",
        rows,
        first_date=str(out["date"].min().date()),
        last_date=str(out["date"].max().date()),
    )
    return rows
