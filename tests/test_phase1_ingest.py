"""Phase 1 - data. Tests of the database contract and the quality checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ingest.events import load_events_csv
from ingest.macro import load_manual_csv
from ingest.prices import store_prices
from ingest.quality import check_macro, check_prices, compare_sources
from storage import connect, read_macro, read_prices, upsert_macro


def test_clean_series_passes(prices):
    report = check_prices(prices, name="synthetic")
    assert report.is_clean
    assert report.problems["missing_days"] == 0
    assert report.problems["duplicate_dates"] == 0
    assert report.rows == len(prices)


def test_detects_missing_days(prices):
    gapped = prices.drop(index=[100, 101, 102]).reset_index(drop=True)
    report = check_prices(gapped)
    assert report.problems["missing_days"] == 3
    assert pd.Timestamp(prices.loc[101, "date"]) in report.missing_days


def test_detects_duplicates(prices):
    duplicated = pd.concat([prices, prices.iloc[[500]]], ignore_index=True)
    report = check_prices(duplicated)
    assert report.problems["duplicate_dates"] == 1
    assert not report.is_clean


def test_detects_outliers_and_bad_ohlc(prices):
    broken = prices.copy()
    broken.loc[800, ["open", "high", "low", "close"]] = [100.0, 90.0, 95.0, 99.0]
    broken.loc[1200, "close"] = broken.loc[1200, "close"] * 20
    report = check_prices(broken)
    assert report.problems["ohlc_violations"] >= 1
    assert any(ts == pd.Timestamp(broken.loc[1200, "date"]) for ts, _ in report.return_outliers)
    assert not report.is_clean


def test_detects_non_positive_and_stale(prices):
    broken = prices.copy()
    broken.loc[10, "close"] = 0.0
    broken.loc[600:610, "close"] = broken.loc[600, "close"]
    report = check_prices(broken)
    assert report.problems["non_positive"] == 1
    assert report.problems["stale_runs"] >= 1


def test_outlier_threshold_is_robust_to_a_single_crash(prices):
    """A single crash must not raise the threshold enough to hide itself."""
    crashed = prices.copy()
    crashed.loc[1500, "close"] = crashed.loc[1500, "close"] * 0.35
    report = check_prices(crashed)
    flagged = {ts for ts, _ in report.return_outliers}
    assert pd.Timestamp(crashed.loc[1500, "date"]) in flagged


def test_store_prices_is_idempotent(db_path, prices):
    with connect(db_path) as conn:
        first = store_prices(conn, prices, "BTCUSD", "synthetic")
        second = store_prices(conn, prices, "BTCUSD", "synthetic")
        stored = read_prices(conn, "BTCUSD")
    assert first == second == len(prices)
    assert len(stored) == len(prices), "re-importing must not duplicate rows"


def test_stored_prices_carry_next_day_availability(db_path, prices):
    with connect(db_path) as conn:
        store_prices(conn, prices, "BTCUSD", "synthetic")
        stored = read_prices(conn, "BTCUSD")
    lag = (stored["available_from"] - stored["date"]).dt.days
    assert (lag == 1).all(), "the bar for day D is known only after it closes"


def test_read_macro_respects_publication_date(db_path):
    """Macro series must not be visible before their publication date."""
    observations = pd.DataFrame(
        {
            "series": "m2",
            "source": "test",
            "date": pd.to_datetime(["2020-02-29", "2020-03-31"]),
            "value": [15.0, 16.0],
            "available_from": pd.to_datetime(["2020-03-25", "2020-04-28"]),
            "ingested_at": "now",
        }
    )
    with connect(db_path) as conn:
        upsert_macro(conn, observations)
        early = read_macro(conn, "m2", as_of="2020-04-01")
        later = read_macro(conn, "m2", as_of="2020-05-01")
    assert len(early) == 1
    assert early.iloc[0]["date"] == pd.Timestamp("2020-02-29")
    assert len(later) == 2


def test_compare_sources_flags_divergence(prices):
    other = prices.copy()
    other.loc[300, "close"] = other.loc[300, "close"] * 1.5
    flagged = compare_sources(prices, other, tolerance=0.05)
    assert len(flagged) == 1
    assert flagged.iloc[0]["date"] == pd.Timestamp(prices.loc[300, "date"])


def test_compare_sources_quiet_when_equal(prices):
    assert compare_sources(prices, prices).empty


def test_check_macro_flags_negative_lag():
    bad = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"]),
            "value": [1.0],
            "available_from": pd.to_datetime(["2020-01-15"]),
        }
    )
    report = check_macro(bad)
    assert report["negative_lag"] == 1
    assert not report["is_clean"]


def test_seed_events_csv_is_valid():
    from config import load_config

    events = load_events_csv(load_config().root / "data" / "raw" / "events.csv")
    assert len(events) >= 15
    assert (events["available_from"] >= events["date"]).all()
    assert events["date"].is_monotonic_increasing
    assert events[events["category"] == "halving"].shape[0] == 4


def test_manual_csv_defaults_to_configured_lag(tmp_path):
    path = tmp_path / "pmi.csv"
    path.write_text("observation_date,value\n2020-01-31,47.2\n2020-02-29,50.1\n", encoding="utf-8")
    series = load_manual_csv(path, default_lag_days=3)
    assert list(series.columns) == ["date", "value", "available_from"]
    assert (series["available_from"] - series["date"]).dt.days.tolist() == [3, 3]


def test_empty_frame_reports_cleanly():
    report = check_prices(pd.DataFrame(columns=["date", "close"]), name="empty")
    assert report.rows == 0
    assert report.first_date is None
    assert np.isfinite(0)


@pytest.mark.network
def test_live_bitstamp_matches_contract():
    from ingest.prices import fetch_prices

    df = fetch_prices("bitstamp", "BTCUSD", "2024-01-01", "2024-01-31")
    assert len(df) == 31
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert check_prices(df).is_clean
