"""Phase 2 - features. The point: no feature may use data from the future."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.build import (
    FeatureInputs,
    add_forward_returns,
    build_features,
    feature_columns,
    price_features,
)
from features.checks import (
    assert_no_lookahead,
    pointwise_lookahead_report,
    targets_are_forward_only,
)
from features.events import days_since_event, event_flags
from features.halving import CONFIRMED_HALVINGS, halving_features, halving_windows
from features.macro_phase import asof_series, macro_phase
from ingest.prices import stitch_sources
from validation.synthetic import random_walk_prices


@pytest.fixture
def inputs() -> FeatureInputs:
    prices = random_walk_prices(3000, start="2013-01-01", seed=11)
    prices["available_from"] = prices["date"] + pd.Timedelta(days=1)

    # A monthly macro series published with a 30-day lag.
    observation_dates = pd.date_range("2012-01-31", "2021-12-31", freq="ME")
    macro = pd.DataFrame(
        {
            "series": "m2",
            "date": observation_dates,
            "value": np.linspace(10.0, 21.0, len(observation_dates)),
            "available_from": observation_dates + pd.Timedelta(days=30),
        }
    )
    daily_dates = pd.date_range("2012-01-01", "2021-12-31", freq="D")
    dxy = pd.DataFrame(
        {
            "series": "dxy",
            "date": daily_dates,
            "value": 90 + 5 * np.sin(np.arange(len(daily_dates)) / 90.0),
            "available_from": daily_dates + pd.Timedelta(days=1),
        }
    )
    fed_funds = pd.DataFrame(
        {
            "series": "fed_funds",
            "date": daily_dates,
            "value": np.clip(2.5 - np.arange(len(daily_dates)) / 800.0, 0.05, None),
            "available_from": daily_dates + pd.Timedelta(days=1),
        }
    )
    macro = pd.concat([macro, dxy, fed_funds], ignore_index=True)

    events = pd.DataFrame(
        {
            "name": ["halving_2016", "shock_a", "shock_b"],
            "date": pd.to_datetime(["2016-07-09", "2017-09-04", "2020-03-12"]),
            "category": ["halving", "regulation", "macro"],
            "description": [None, None, None],
            "available_from": pd.to_datetime(["2016-07-09", "2017-09-04", "2020-03-12"]),
            "source": [None, None, None],
        }
    )
    return FeatureInputs(prices=prices, macro=macro, events=events)


# --- halving -------------------------------------------------------------------


def test_halving_distance_counts_from_last_halving():
    dates = pd.DatetimeIndex(["2016-07-09", "2016-07-10", "2020-05-10", "2020-05-11"])
    frame = halving_features(dates)
    assert frame.loc[pd.Timestamp("2016-07-09"), "days_since_halving"] == 0
    assert frame.loc[pd.Timestamp("2016-07-10"), "days_since_halving"] == 1
    assert frame.loc[pd.Timestamp("2020-05-11"), "days_since_halving"] == 0
    assert frame.loc[pd.Timestamp("2020-05-10"), "days_since_halving"] == 1401


def test_cycle_index_increments_on_each_halving():
    dates = pd.DatetimeIndex(
        ["2011-01-01", "2013-01-01", "2017-01-01", "2021-01-01", "2025-01-01"]
    )
    assert halving_features(dates)["cycle_index"].tolist() == [0, 1, 2, 3, 4]


def test_strict_mode_drops_forward_looking_columns():
    dates = pd.date_range("2015-01-01", periods=10, freq="D")
    strict = halving_features(dates, strict=True)
    loose = halving_features(dates, strict=False)
    assert "days_to_next_halving" not in strict.columns
    assert "cycle_progress" not in strict.columns
    assert "days_to_next_halving" in loose.columns


def test_halving_window_flag_turns_on_only_after_the_event():
    dates = pd.date_range("2020-05-08", periods=10, freq="D")
    flags = halving_windows(dates, [30])["halving_after_30d"]
    assert flags.loc[pd.Timestamp("2020-05-10")] == 0
    assert flags.loc[pd.Timestamp("2020-05-11")] == 1
    assert flags.loc[pd.Timestamp("2020-05-17")] == 1


def test_halving_window_expires_after_n_days():
    dates = pd.date_range("2020-05-11", periods=60, freq="D")
    flags = halving_windows(dates, [30])["halving_after_30d"]
    assert flags.loc[pd.Timestamp("2020-06-10")] == 1
    assert flags.loc[pd.Timestamp("2020-06-11")] == 0


# --- macro ---------------------------------------------------------------------


def test_asof_series_waits_for_publication_date():
    macro = pd.DataFrame(
        {
            "series": "m2",
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "value": [100.0, 110.0],
            "available_from": pd.to_datetime(["2020-03-01", "2020-03-31"]),
        }
    )
    index = pd.to_datetime(["2020-02-15", "2020-03-01", "2020-03-30", "2020-03-31"])
    values = asof_series(macro, "m2", index)
    assert np.isnan(values.iloc[0]), "before publication even the older observation is unknown"
    assert values.iloc[1] == 100.0
    assert values.iloc[2] == 100.0
    assert values.iloc[3] == 110.0


def test_macro_phase_labels_are_absent_without_enough_history(inputs):
    index = pd.date_range("2013-01-01", "2013-06-30", freq="D")
    phases = macro_phase(inputs.macro, index)
    assert phases["macro_phase"].isna().all()


def test_macro_phase_produces_labels_once_history_exists(inputs):
    index = pd.date_range("2013-01-01", "2020-12-31", freq="D")
    phases = macro_phase(inputs.macro, index)
    labels = set(phases["macro_phase"].dropna().unique())
    assert labels
    assert labels <= {
        "expanding_rising",
        "expanding_falling",
        "contracting_rising",
        "contracting_falling",
    }


# --- events --------------------------------------------------------------------


def test_event_flag_is_zero_before_the_event(inputs):
    index = pd.date_range("2020-03-01", "2020-04-30", freq="D")
    flags = event_flags(inputs.events, index, [30])["event_macro_30d"]
    assert flags.loc[pd.Timestamp("2020-03-11")] == 0
    assert flags.loc[pd.Timestamp("2020-03-12")] == 1
    assert flags.loc[pd.Timestamp("2020-04-11")] == 1
    assert flags.loc[pd.Timestamp("2020-04-13")] == 0


def test_days_since_event_is_nan_before_first_event(inputs):
    index = pd.date_range("2015-01-01", "2018-01-01", freq="D")
    days = days_since_event(inputs.events, index)["days_since_event_regulation"]
    assert np.isnan(days.loc[pd.Timestamp("2017-09-03")])
    assert days.loc[pd.Timestamp("2017-09-04")] == 0
    assert days.loc[pd.Timestamp("2017-09-14")] == 10


# --- price features and targets ------------------------------------------------


def test_price_features_use_only_past_data(inputs):
    frame = price_features(inputs.prices)
    close = frame["close"]
    manual = close.iloc[30] / close.iloc[0] - 1
    assert np.isclose(frame["return_30d"].iloc[30], manual)
    assert frame["return_30d"].iloc[:30].isna().all()
    assert frame["drawdown"].max() <= 0


def test_forward_returns_are_targets_not_features(inputs):
    frame = add_forward_returns(build_features(inputs), [30])
    assert targets_are_forward_only(frame, 30)
    assert "fwd_return_30d" not in feature_columns(frame)
    close = frame["close"]
    expected = close.iloc[30] / close.iloc[0] - 1
    assert np.isclose(frame["fwd_return_30d"].iloc[0], expected)


# --- the main test: no look-ahead ----------------------------------------------


def test_no_lookahead_in_full_feature_frame(inputs):
    """Every checked day must come out the same when only the past is known."""
    test_dates = [
        "2014-06-15", "2015-11-20", "2016-07-20", "2017-12-01",
        "2018-08-08", "2019-04-01", "2020-03-20", "2020-12-31",
    ]
    assert_no_lookahead(lambda as_of: build_features(inputs, as_of=as_of), test_dates)


def test_lookahead_detector_catches_a_planted_leak(inputs):
    """Control on the detector: a feature with an obvious leak must be caught."""

    def leaky_build(as_of):
        frame = build_features(inputs, as_of=as_of)
        if frame.empty:
            return frame
        # The classic leak: normalising by the median of the WHOLE sample.
        frame["leak_zscore"] = frame["close"] / frame["close"].median()
        return frame

    report = pointwise_lookahead_report(leaky_build, ["2016-07-20", "2018-08-08"])
    assert not report.empty
    assert set(report["column"]) == {"leak_zscore"}


def test_point_in_time_build_stops_at_the_cutoff(inputs):
    frame = build_features(inputs, as_of="2017-01-01")
    # The bar for day D is known only on D+1, so the last row is 2016-12-31.
    assert frame.index.max() == pd.Timestamp("2016-12-31")


# --- source stitching -----------------------------------------------------


def test_stitch_prefers_higher_priority_source():
    old = random_walk_prices(400, start="2016-01-01", seed=1)
    new = random_walk_prices(300, start="2016-06-01", seed=2)
    stitched, report = stitch_sources({"bitstamp": old, "binance": new}, ["binance", "bitstamp"])

    overlap_day = pd.Timestamp("2016-07-01")
    chosen = stitched.loc[stitched["date"] == overlap_day, "close"].iloc[0]
    expected = new.loc[new["date"] == overlap_day, "close"].iloc[0]
    assert chosen == expected

    early_day = pd.Timestamp("2016-02-01")
    assert stitched.loc[stitched["date"] == early_day, "source_used"].iloc[0] == "bitstamp"
    assert set(report["source"]) == {"binance", "bitstamp"}
    assert not report.attrs["overlap"].empty


def test_stitch_covers_the_union_of_days():
    old = random_walk_prices(400, start="2016-01-01", seed=1)
    new = random_walk_prices(300, start="2016-06-01", seed=2)
    stitched, _ = stitch_sources({"bitstamp": old, "binance": new}, ["binance", "bitstamp"])
    union = set(old["date"]) | set(new["date"])
    assert len(stitched) == len(union)
    assert stitched["date"].is_monotonic_increasing


def test_stitch_handles_single_source():
    only = random_walk_prices(100, start="2018-01-01", seed=3)
    stitched, report = stitch_sources({"binance": only}, ["binance", "bitstamp"])
    assert len(stitched) == 100
    assert report.attrs["overlap"].empty


@pytest.mark.network
def test_live_binance_matches_contract():
    from ingest.prices import fetch_prices
    from ingest.quality import check_prices

    df = fetch_prices("binance", "BTCUSD", "2024-01-01", "2024-02-15")
    assert len(df) == 46
    assert check_prices(df).is_clean
