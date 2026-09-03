"""Parsing FRED responses - tests without network or key.

It comes down to one thing: `available_from` for M2 must be the date of the
FIRST publication, not the observation date and not today's revision. The rest
of the project rests on that field, so every edge case of a FRED response has
its own test here.
"""
from __future__ import annotations

import pandas as pd
import pytest

from features.macro_phase import asof_series, macro_phase
from ingest.macro import (
    FRED_NO_VINTAGE_SENTINEL,
    MissingCredentials,
    fetch_fred,
    parse_fred_observations,
)
from ingest.quality import check_macro


def observation(date: str, value: str, realtime: str | None = None) -> dict:
    """A row in the shape FRED returns it (values as text)."""
    return {
        "realtime_start": realtime or FRED_NO_VINTAGE_SENTINEL,
        "realtime_end": "9999-12-31",
        "date": date,
        "value": value,
    }


# --- publication dates ---------------------------------------------------------


def test_vintage_date_becomes_publication_date():
    """February 2020 M2 became known in late March - the database must say so."""
    payload = [
        observation("2020-01-31", "15420.0", "2020-02-27"),
        observation("2020-02-29", "15500.5", "2020-03-26"),
        observation("2020-03-31", "16600.0", "2020-04-28"),
    ]
    series = parse_fred_observations(payload, publication_lag_days=30)
    assert series.loc[1, "available_from"] == pd.Timestamp("2020-03-26")
    assert (series["lag_source"] == "vintage").all()
    assert (series["available_from"] >= series["date"]).all()


def test_missing_values_are_dropped():
    payload = [
        observation("2020-01-31", "15420.0", "2020-02-27"),
        observation("2020-02-29", ".", "2020-03-26"),
    ]
    series = parse_fred_observations(payload)
    assert len(series) == 1
    assert series.loc[0, "date"] == pd.Timestamp("2020-01-31")


def test_series_without_vintage_history_falls_back_to_fixed_lag():
    """The 1776-07-04 sentinel means there is no vintage archive."""
    payload = [observation("2020-01-31", "15420.0"), observation("2020-02-29", "15500.5")]
    series = parse_fred_observations(payload, publication_lag_days=30)
    assert (series["lag_source"] == "fallback").all()
    assert (series["available_from"] - series["date"]).dt.days.tolist() == [30, 30]


def test_observations_older_than_the_vintage_archive_use_fixed_lag():
    """We do not pretend 1960 M2 became known in 1997.

    ALFRED keeps vintages from roughly the 1990s. Earlier observations receive
    a `realtime_start` equal to the start of the archive - taken literally,
    every rolling window before 1997 would be empty.
    """
    payload = [
        observation("1960-01-31", "300.0", "1997-01-10"),
        observation("1970-01-31", "600.0", "1997-01-10"),
        observation("2020-01-31", "15420.0", "2020-02-27"),
    ]
    series = parse_fred_observations(payload, publication_lag_days=30)
    old = series[series["date"] < pd.Timestamp("1990-01-01")]
    assert (old["lag_source"] == "fallback").all()
    assert (old["available_from"] - old["date"]).dt.days.tolist() == [30, 30]
    assert series.iloc[-1]["available_from"] == pd.Timestamp("2020-02-27")


def test_implausibly_long_vintage_lag_falls_back():
    """A year-long lag is the result of a methodology revision, not a publication date."""
    payload = [
        observation("2015-01-31", "11800.0", "2015-02-26"),
        observation("2015-02-28", "11850.0", "2016-08-01"),
        observation("2015-03-31", "11900.0", "2015-04-28"),
    ]
    series = parse_fred_observations(payload, publication_lag_days=30, max_extra_lag_days=120)
    assert series.loc[1, "lag_source"] == "fallback"
    assert series.loc[1, "available_from"] == pd.Timestamp("2015-03-30")


def test_publication_never_precedes_observation():
    payload = [observation("2020-03-31", "16600.0", "2020-03-01")]
    series = parse_fred_observations(payload)
    assert series.loc[0, "available_from"] == pd.Timestamp("2020-03-31")


def test_use_vintages_false_ignores_realtime_entirely():
    payload = [observation("2020-01-31", "15420.0", "2020-02-27")]
    series = parse_fred_observations(payload, publication_lag_days=45, use_vintages=False)
    assert series.loc[0, "available_from"] == pd.Timestamp("2020-03-16")
    assert series.loc[0, "lag_source"] == "fallback"


def test_parsed_series_passes_quality_check():
    payload = [
        observation("2020-01-31", "15420.0", "2020-02-27"),
        observation("2020-02-29", "15500.5", "2020-03-26"),
    ]
    report = check_macro(parse_fred_observations(payload), name="m2")
    assert report["is_clean"]
    assert report["negative_lag"] == 0


def test_empty_payload_returns_empty_frame():
    assert parse_fred_observations([]).empty


# --- integration with the macro phase ------------------------------------------


@pytest.fixture
def m2_from_fred() -> pd.DataFrame:
    """A realistic M2: monthly, published ~4 weeks after the month ends."""
    dates = pd.date_range("2011-01-31", "2026-06-30", freq="ME")
    payload = [
        observation(
            str(date.date()),
            f"{9000 + 40 * i + (900 if date >= pd.Timestamp('2020-04-30') else 0):.1f}",
            str((date + pd.Timedelta(days=26)).date()),
        )
        for i, date in enumerate(dates)
    ]
    series = parse_fred_observations(payload, publication_lag_days=30)
    series["series"] = "m2"
    return series


def test_m2_is_invisible_before_its_publication_date(m2_from_fred):
    index = pd.to_datetime(["2020-04-25", "2020-04-26", "2020-05-25", "2020-05-26"])
    values = asof_series(m2_from_fred, "m2", index)
    # The April observation is published on 26 May - on 25 May it does not exist yet.
    assert values.iloc[2] == values.iloc[1]
    assert values.iloc[3] > values.iloc[2]


def test_macro_phase_can_be_forced_onto_m2(m2_from_fred):
    rates = pd.DataFrame(
        {
            "series": "fed_funds",
            "date": pd.date_range("2011-01-01", "2026-06-30", freq="D"),
        }
    )
    rates["value"] = 3.0 - (rates.index / 1500.0)
    rates["available_from"] = rates["date"] + pd.Timedelta(days=1)
    macro = pd.concat([m2_from_fred, rates], ignore_index=True)

    index = pd.date_range("2012-01-01", "2026-06-30", freq="D")
    phases = macro_phase(macro, index, liquidity_source="m2_yoy")
    assert phases.attrs["liquidity_source"] == "m2_yoy"
    assert phases["macro_phase"].notna().any()


def test_forcing_a_missing_source_fails_loudly(m2_from_fred):
    index = pd.date_range("2012-01-01", "2013-01-01", freq="D")
    with pytest.raises(ValueError, match="dxy_chg_3m_inv"):
        macro_phase(m2_from_fred, index, liquidity_source="dxy_chg_3m_inv")


# --- the API key ---------------------------------------------------------------


def test_missing_key_raises_a_helpful_error(monkeypatch):
    """Without a key we want a readable exception, not a 400 from the API.

    We patch `secret` rather than the environment variable: `load_config`
    reads .env on first use, so a deleted variable would come back from file.
    """
    import ingest.macro as macro_module

    monkeypatch.setattr(macro_module, "secret", lambda name: None)
    with pytest.raises(MissingCredentials, match="FRED_API_KEY"):
        fetch_fred("M2SL", api_key=None)


@pytest.mark.network
def test_live_fred_returns_real_m2():
    """Runs only with a real key in .env."""
    from config import secret

    if not secret("FRED_API_KEY"):
        pytest.skip("no FRED_API_KEY - paste a key into .env")

    series = fetch_fred("M2SL", publication_lag_days=45)
    # The M2SL vintage archive starts in 1980, so there are fewer first
    # releases than observations in the plain series (which reaches 1959).
    assert len(series) > 400
    assert series["date"].min() <= pd.Timestamp("1990-01-01")
    assert series["date"].max() > pd.Timestamp("2025-01-01")
    assert series.attrs["vintages_available"] is True
    lag = (series["available_from"] - series["date"]).dt.days
    assert (lag >= 0).all()
    assert 14 <= lag.median() <= 60, "M2 is published about a month after the period ends"
    assert series.attrs["vintage_rows"] > 0, "no publication dates from the vintage archive"


# --- secrets and API limits ----------------------------------------------------


def test_api_key_never_appears_in_error_messages():
    """The key travels in the query string - a raw URL in an exception is a leak."""
    from ingest.http import FetchError, redact

    url = "https://api.stlouisfed.org/fred/series/observations?series_id=DFF&api_key=abc123secret&file_type=json"
    assert "abc123secret" not in redact(url)
    assert "api_key=***" in redact(url)

    error = FetchError(f"{url} -> HTTP 400")
    assert "abc123secret" not in str(error)


def test_redaction_covers_common_secret_parameter_names():
    from ingest.http import redact

    for name in ("api_key", "apikey", "token", "access_key", "secret"):
        assert "secretvalue" not in redact(f"https://x/y?{name}=secretvalue&z=1")


def test_series_with_too_many_vintages_falls_back_to_plain_request(monkeypatch):
    """DFF has over 5000 vintages while FRED serves first releases up to 2000.

    Instead of dropping the series we fetch it without the vintage archive -
    for a daily series published the next day, a fixed lag is faithful.
    """
    import ingest.macro as macro_module
    from ingest.http import FetchError

    calls = []

    def fake_get_json(url, **kwargs):
        calls.append(url)
        if "output_type=4" in url:
            raise FetchError(
                "HTTP 400 {\"error_message\":\"There are 5113 vintage dates ... "
                "This exceeds the maximum number of vintage dates allowed for this file type (2000).\"}"
            )
        return {
            "observations": [
                observation("2020-03-16", "0.65"),
                observation("2020-03-17", "0.25"),
            ]
        }

    monkeypatch.setattr(macro_module, "get_json", fake_get_json)
    series = macro_module.fetch_fred("DFF", publication_lag_days=1, api_key="test")

    assert len(calls) == 2, "first the vintage attempt, then the plain request"
    assert "output_type=4" in calls[0] and "output_type=4" not in calls[1]
    assert series.attrs["vintages_requested"] is True
    assert series.attrs["vintages_available"] is False
    assert (series["available_from"] - series["date"]).dt.days.tolist() == [1, 1]


def test_other_http_errors_are_not_silently_retried(monkeypatch):
    """Only the vintage limit justifies a fallback - other errors must hurt."""
    import ingest.macro as macro_module
    from ingest.http import FetchError

    def fake_get_json(url, **kwargs):
        raise FetchError("HTTP 400 {\"error_message\":\"Bad Request. Variable series_id is invalid.\"}")

    monkeypatch.setattr(macro_module, "get_json", fake_get_json)
    with pytest.raises(FetchError, match="series_id is invalid"):
        macro_module.fetch_fred("NIE_ISTNIEJE", api_key="test")
