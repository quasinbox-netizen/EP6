"""Fetching daily OHLCV bars for BTC/USD.

Four independent sources, none requiring an API key:

* binance  - BTC/USDT from 2017-08. The deepest market and the largest volume,
             so from 2017 onwards it is the source of truth. It does not reach
             the 2012 and 2016 halvings - the exchange itself launched in 2017.
* bitstamp - BTC/USD from 2011-08. The only one of these covering the full
             cycle history; shallower, but continuous.
* coinbase - BTC/USD from 2015-07, cross-validation.
* yahoo    - a market aggregate from 2014-09, a third voice on divergences.

Hence the default series is STITCHED (`stitch_sources`): Bitstamp up to
2017-08, Binance afterwards. The seam is explicit, logged and checked on the
overlap - not buried inside the data.

A note on the pair: Binance quotes BTC/USDT, not BTC/USD. Historically the
difference is a fraction of a percent outside USDT de-pegging episodes (October
2018, for instance) - which is why stitching compares the overlap and reports
the deviation instead of silently accepting it.

Every provider returns the same contract: a DataFrame with columns
[date, open, high, low, close, volume], sorted ascending, without duplicates,
dates in UTC. Storing appends `available_from = date + 1 day`, because the bar
for day D is only complete once the day has closed.
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
    # Control group. A halving is a Bitcoin-only event, so whatever the NASDAQ
    # does around the same dates is a placebo test.
    "NASDAQ": "^IXIC",
    "SP500": "^GSPC",
    "GOLD": "GC=F",
}

# First day with data in each source - used when stitching and in the CLI.
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
    """Daily candles from Binance (BTC/USDT), paged 1000 bars at a time.

    Binance returns `openTime` in milliseconds; a daily bar opens at 00:00 UTC,
    so the bar's date is the day it opened.
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
    window = timedelta(days=280)  # the API caps a request at 300 candles
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
        raise FetchError(f"yahoo: no data for {ticker}")
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
        raise ValueError(f"unknown price source: {source} (available: {sorted(PROVIDERS)})")
    return PROVIDERS[source](symbol, start, end)


def stitch_sources(
    frames: dict[str, pd.DataFrame], priority: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stitch a series from several exchanges by priority, day by day.

    For each day the first source in `priority` that has it wins. Returns
    (series, report): the report carries the number of days taken from each
    source, the seam dates and the median divergence on the overlaps - without
    those, stitching is a silent source of errors.
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

    # Divergence on the overlap with the next source in priority order.
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
    """The stitched series, built from whatever is already in the database."""
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
    """Store bars, appending metadata and the availability date (D+1)."""
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
