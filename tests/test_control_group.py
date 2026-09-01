"""Grupa kontrolna - test placebo i roznica w roznicach.

Dwie rzeczy musza dzialac, zeby ten modul cokolwiek znaczyl:

1. Kalendarz. NASDAQ ma ~252 sesje w roku, BTC 365 dni. Bez wyrownania
   "365 dni po halvingu" znaczyloby dla obu aktywow co innego.
2. Parowanie. Halvingi wypadaja w konkretnych warunkach makro, ktore ruszaja
   oba aktywa naraz. Roznica musi byc liczona per zdarzenie, nie jako
   roznica dwoch niezaleznych srednich.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.control import (
    compare_with_control,
    per_event_car,
    placebo_event_study,
    to_calendar,
)
from validation.synthetic import inject_drift, random_walk_prices


def trading_days_only(prices: pd.DataFrame) -> pd.DataFrame:
    """Zostawia same dni robocze - imitacja gieldy wobec ciagle handlowanego BTC."""
    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out[out["date"].dt.dayofweek < 5].reset_index(drop=True)


@pytest.fixture
def anchors() -> list[str]:
    """Cztery kotwice mieszczace sie z zapasem w probie 3000 dni od 2013-01-01.

    Ostatnia musi zostawic miejsce na pelne okno 365 dni, inaczej wypadnie
    z proby i test bedzie mierzyl co innego, niz sadzi.
    """
    return ["2014-01-15", "2016-07-09", "2018-05-20", "2020-02-10"]


# --- kalendarz ------------------------------------------------------------


def test_to_calendar_fills_weekends_with_last_close():
    prices = trading_days_only(random_walk_prices(400, start="2020-01-01", seed=4))
    calendar = to_calendar(prices).set_index("date")

    friday = pd.Timestamp("2020-01-03")
    saturday = pd.Timestamp("2020-01-04")
    sunday = pd.Timestamp("2020-01-05")
    assert calendar.loc[saturday, "close"] == calendar.loc[friday, "close"]
    assert calendar.loc[sunday, "close"] == calendar.loc[friday, "close"]


def test_to_calendar_has_no_gaps_and_no_lookahead():
    prices = trading_days_only(random_walk_prices(500, start="2020-01-01", seed=5))
    calendar = to_calendar(prices).set_index("date")

    expected = pd.date_range(calendar.index.min(), calendar.index.max(), freq="D")
    assert calendar.index.equals(expected), "kalendarz musi byc ciagly"

    # Kazda wartosc kalendarzowa pochodzi z sesji NIE POZNIEJSZEJ niz ten dzien.
    sessions = prices.set_index(pd.to_datetime(prices["date"]))["close"]
    for day in calendar.index[::37]:
        past = sessions[sessions.index <= day]
        assert calendar.loc[day, "close"] == past.iloc[-1]


def test_without_calendar_alignment_most_events_are_lost(anchors):
    """Brak wyrownania kosztuje nie precyzje, tylko zdarzenia.

    Na szeregu samych sesji "365 dni po" znaczy 365 WIERSZY, czyli okolo 511
    dni kalendarzowych. Do tego zdarzenie wypadajace w weekend nie istnieje
    w indeksie. Efekt: z czterech kotwic zostaje jedna, a badanie po cichu
    zamienia sie w opis jednego przypadku.
    """
    prices = trading_days_only(random_walk_prices(3000, start="2013-01-01", seed=6))
    raw = per_event_car(prices, anchors, post=365)
    aligned = per_event_car(to_calendar(prices), anchors, post=365)

    assert len(aligned) == 4
    assert len(raw) == 1, "trzy kotwice gina: dwie w weekend, jedna przez przekroczenie proby"
    survivor = raw.index[0]
    assert not np.isclose(raw.loc[survivor], aligned.loc[survivor]), (
        "nawet ocalale zdarzenie ma inne okno"
    )


def test_events_land_on_calendar_days_even_when_market_was_closed():
    """Halving w sobote nie moze wyrzucic zdarzenia z proby kontrolnej."""
    prices = trading_days_only(random_walk_prices(1500, start="2015-01-01", seed=7))
    saturday = "2016-07-09"
    assert pd.Timestamp(saturday).dayofweek == 5
    assert per_event_car(prices, [saturday], post=90).empty
    assert len(per_event_car(to_calendar(prices), [saturday], post=90)) == 1


# --- roznica w roznicach --------------------------------------------------


def test_shared_shock_leaves_no_difference(anchors):
    """Gdy oba aktywa dostaja ten sam impuls, roznica ma byc nieistotna."""
    rng = np.random.default_rng(11)
    treatment = random_walk_prices(3000, start="2013-01-01", seed=21, sigma=0.02)
    control = random_walk_prices(3000, start="2013-01-01", seed=22, sigma=0.012)

    # Wspolny impuls: ten sam dryf w tych samych oknach dla obu aktywow.
    treatment = inject_drift(treatment, anchors, window=180, daily_drift=0.004)
    control = inject_drift(control, anchors, window=180, daily_drift=0.004)
    _ = rng

    comparison = compare_with_control(
        treatment, trading_days_only(control), anchors, post=180, horizons=[180]
    )
    assert comparison.n_events == 4
    assert comparison.summary_at["difference_p_value"] > 0.05
    assert "nie da sie odroznic" in comparison.verdict()


def test_effect_only_in_treatment_is_detected(anchors):
    """Gdy dryf dostaje TYLKO badane aktywo, roznica musi zostac wykryta."""
    treatment = random_walk_prices(3000, start="2013-01-01", seed=21, sigma=0.012)
    control = random_walk_prices(3000, start="2013-01-01", seed=22, sigma=0.012)
    treatment = inject_drift(treatment, anchors, window=180, daily_drift=0.006)

    comparison = compare_with_control(
        treatment, trading_days_only(control), anchors, post=180, horizons=[180]
    )
    assert comparison.summary_at["difference"] > 0.5
    assert comparison.summary_at["difference_p_value"] < 0.05
    assert "NIE jest wspolny" in comparison.verdict()


def test_pairing_removes_a_common_time_effect():
    """Sparowanie usuwa wspolny ruch rynku; porownanie srednich by go zostawilo.

    Oba aktywa dostaja ten sam duzy impuls, ale rozny dla roznych zdarzen.
    Sparowana roznica ma byc bliska zeru mimo ogromnego rozrzutu CAR.
    """
    anchors = ["2014-01-15", "2016-07-09", "2018-05-20", "2020-02-10"]
    treatment = random_walk_prices(3000, start="2013-01-01", seed=31, sigma=0.01)
    control = random_walk_prices(3000, start="2013-01-01", seed=32, sigma=0.01)
    for anchor, drift in zip(anchors, (0.010, -0.006, 0.008, -0.004)):
        treatment = inject_drift(treatment, [anchor], window=180, daily_drift=drift)
        control = inject_drift(control, [anchor], window=180, daily_drift=drift)

    comparison = compare_with_control(
        treatment, trading_days_only(control), anchors, post=180, horizons=[180]
    )
    per_event = comparison.per_event
    # Rozrzut samych CAR jest duzy, ale rozrzut ROZNIC juz nie.
    assert per_event["BTC"].std() > 3 * per_event["difference"].std()
    assert comparison.summary_at["difference_p_value"] > 0.05


def test_table_reports_every_requested_horizon(anchors):
    treatment = random_walk_prices(3000, start="2013-01-01", seed=41)
    control = trading_days_only(random_walk_prices(3000, start="2013-01-01", seed=42))
    comparison = compare_with_control(
        treatment, control, anchors, post=365, horizons=[30, 90, 365]
    )
    assert list(comparison.table.index) == [30, 90, 365]
    assert (comparison.table["n_events"] == 4).all()


def test_horizons_longer_than_the_window_are_dropped(anchors):
    treatment = random_walk_prices(3000, start="2013-01-01", seed=41)
    control = trading_days_only(random_walk_prices(3000, start="2013-01-01", seed=42))
    comparison = compare_with_control(
        treatment, control, anchors, post=90, horizons=[30, 90, 365]
    )
    assert list(comparison.table.index) == [30, 90]


def test_single_event_reports_no_p_value():
    """Z jednym zdarzeniem nie ma rozrzutu - i nie wolno udawac, ze jest."""
    treatment = random_walk_prices(2000, start="2013-01-01", seed=51)
    control = trading_days_only(random_walk_prices(2000, start="2013-01-01", seed=52))
    comparison = compare_with_control(
        treatment, control, ["2016-01-15"], post=180, horizons=[180]
    )
    assert comparison.n_events == 1
    assert np.isnan(comparison.summary_at["difference_p_value"])
    assert "za malo zdarzen" in comparison.verdict()


def test_no_shared_events_gives_empty_table():
    treatment = random_walk_prices(500, start="2013-01-01", seed=61)
    control = trading_days_only(random_walk_prices(500, start="2013-01-01", seed=62))
    comparison = compare_with_control(
        treatment, control, ["2024-01-01"], post=90, horizons=[90]
    )
    assert comparison.table.empty
    assert comparison.n_events == 0


# --- placebo --------------------------------------------------------------


def test_placebo_on_control_finds_nothing_on_noise(anchors):
    """Metoda nie moze "znajdowac" efektu halvingu w aktywie bez halvingu."""
    control = trading_days_only(random_walk_prices(3000, start="2013-01-01", seed=71))
    result = placebo_event_study(control, anchors, post=180, n_boot=1000)
    assert result.n_events == 4
    assert result.car_summary["p_value"] > 0.05


def test_paired_difference_test_is_calibrated():
    """Na dwoch niezaleznych szumach odrzucen ma byc ~5%, nie 30%.

    Bez tego testu nie wiadomo, czy "roznica nieistotna" na prawdziwych danych
    znaczy cokolwiek: metoda mogla by po prostu nigdy nic nie wykrywac albo
    wykrywac wszystko. Kontrola jest odfiltrowana do dni roboczych, wiec
    sprawdzamy przy okazji, ze samo wyrownanie kalendarza nie wprowadza
    systematycznego przesuniecia.
    """
    anchors = ["2014-01-15", "2016-07-09", "2018-05-20", "2020-02-10"]
    rejections = 0
    differences = []
    trials = 40
    for seed in range(trials):
        treatment = random_walk_prices(3000, start="2013-01-01", seed=7000 + seed)
        control = random_walk_prices(3000, start="2013-01-01", seed=9000 + seed)
        comparison = compare_with_control(
            treatment, trading_days_only(control), anchors, post=90, horizons=[90]
        )
        differences.append(comparison.summary_at["difference"])
        if comparison.summary_at["difference_p_value"] < 0.05:
            rejections += 1

    assert rejections <= 6, f"za duzo falszywych odkryc: {rejections}/{trials}"
    mean_difference = float(np.mean(differences))
    standard_error = float(np.std(differences) / np.sqrt(trials))
    assert abs(mean_difference) < 4 * standard_error, (
        f"wyrownanie kalendarza przesuwa roznice o {mean_difference:+.3f}"
    )
