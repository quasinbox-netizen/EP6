"""Faza 6 - spiecie calosci i dashboard.

Dashboard testujemy dwojako: sprawdzamy, ze wszystkie funkcje, ktorych uzywa,
dzialaja na kompletnej (syntetycznej) bazie, oraz ze sam plik dashboardu nie
zawiera logiki badawczej - inaczej wykres i terminal zaczna sie rozjezdzac.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import REPO_ROOT, load_config
from features.halving import CONFIRMED_HALVINGS
from ingest.events import load_events_csv, store_events
from ingest.prices import store_prices
from pipeline import (
    category_event_studies,
    halving_event_study,
    hypothesis_columns,
    load_lab_data,
    out_of_sample_check,
    run_strategies,
    scan_hypotheses,
)
from storage import connect, upsert_macro
from validation.synthetic import random_walk_prices

DASHBOARD = REPO_ROOT / "dashboard" / "app.py"


@pytest.fixture
def lab_config(tmp_path, monkeypatch):
    """Kopia konfiguracji wskazujaca na tymczasowa baze wypelniona szumem."""
    config = load_config()
    db_path = tmp_path / "lab.sqlite"

    prices = random_walk_prices(4000, start="2012-01-01", seed=99, initial=5.0)
    observation_dates = pd.date_range("2011-01-31", "2026-01-31", freq="ME")
    macro = pd.DataFrame(
        {
            "series": "m2",
            "source": "test",
            "date": observation_dates,
            "value": np.linspace(9.0, 21.0, len(observation_dates)),
            "available_from": observation_dates + pd.Timedelta(days=30),
            "ingested_at": "test",
        }
    )
    daily_dates = pd.date_range("2011-01-01", "2026-01-31", freq="D")
    fed_funds = pd.DataFrame(
        {
            "series": "fed_funds",
            "source": "test",
            "date": daily_dates,
            "value": np.clip(3.0 - np.arange(len(daily_dates)) / 1500.0, 0.05, None),
            "available_from": daily_dates + pd.Timedelta(days=1),
            "ingested_at": "test",
        }
    )
    events = load_events_csv(config.root / "data" / "raw" / "events.csv")

    with connect(db_path) as conn:
        store_prices(conn, prices, config["price"]["symbol"], "bitstamp")
        upsert_macro(conn, macro)
        upsert_macro(conn, fed_funds)
        store_events(conn, events)

    monkeypatch.setattr(type(config), "db_path", property(lambda self: db_path))
    return config


# --- pelny przebieg -------------------------------------------------------


def test_pipeline_builds_a_complete_frame(lab_config):
    data = load_lab_data(lab_config)
    assert not data.is_empty
    assert len(data.features) == 4000
    assert "close" in data.features.columns
    assert "days_since_halving" in data.features.columns
    assert "fwd_return_90d" in data.features.columns
    assert data.features["macro_phase"].notna().any()


def test_hypothesis_columns_cover_halving_and_event_windows(lab_config):
    columns = hypothesis_columns(load_lab_data(lab_config).features)
    assert any(c.startswith("halving_after_") for c in columns)
    assert any(c.startswith("event_") for c in columns)


def test_halving_study_uses_only_halvings_covered_by_data(lab_config):
    """Proba konczy sie w 2022 r., wiec halving 2024 musi wypasc - i byc zgloszony."""
    result = halving_event_study(load_lab_data(lab_config), post=180, config=lab_config)
    assert result.n_events == 3
    assert pd.Timestamp("2024-04-20") in result.skipped_events
    assert len(result.used_events) + len(result.skipped_events) == len(CONFIRMED_HALVINGS)
    assert set(["car", "car_ci_low", "car_ci_high", "car_p_value"]) <= set(result.table.columns)


def test_category_studies_cover_registered_categories(lab_config):
    studies = category_event_studies(load_lab_data(lab_config), post=90, config=lab_config)
    assert "halving" in studies
    assert all(result.n_events > 0 for result in studies.values())


def test_scan_is_corrected_for_multiple_testing(lab_config):
    scan = scan_hypotheses(load_lab_data(lab_config), config=lab_config)
    assert not scan.empty
    assert "p_adjusted" in scan.columns
    assert (scan["p_adjusted"].dropna() >= scan["p_value"].dropna() - 1e-12).all()
    assert scan.attrs["n_tests"] >= 10


def test_scan_on_noise_finds_nothing_after_correction(lab_config):
    """Baza testowa to czysty szum - po korekcie nie moze zostac nic."""
    scan = scan_hypotheses(load_lab_data(lab_config), config=lab_config)
    assert not scan["significant_adjusted"].any()


def test_out_of_sample_check_runs_on_both_splits(lab_config):
    report = out_of_sample_check(load_lab_data(lab_config), config=lab_config)
    assert not report.empty
    assert {"train_effect", "test_effect", "replicated"} <= set(report.columns)
    assert not report["replicated"].any(), "na szumie nic nie ma prawa sie powtorzyc"


def test_backtests_include_the_baseline(lab_config):
    table, results = run_strategies(load_lab_data(lab_config), config=lab_config)
    assert "kup i trzymaj" in table.index
    assert len(results) >= 3
    assert (table["total_cost"] >= 0).all()
    assert (table["max_drawdown"] <= 0).all()


# --- dashboard bez logiki -------------------------------------------------


def test_dashboard_exists_and_parses():
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"))
    assert isinstance(tree, ast.Module)


def test_dashboard_does_not_import_analysis_libraries():
    """Dashboard nie moze liczyc statystyk na wlasna reke."""
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"numpy", "scipy", "statsmodels", "sklearn"}
    assert not (imported & forbidden), f"logika w dashboardzie: {imported & forbidden}"


def test_dashboard_calls_pipeline_for_every_result():
    """Kazda liczba na ekranie ma pochodzic z pipeline/src, nie z app.py."""
    source = DASHBOARD.read_text(encoding="utf-8")
    for function in (
        "load_lab_data", "halving_event_study", "category_event_studies",
        "scan_hypotheses", "out_of_sample_check", "run_strategies",
        "control_comparison",
    ):
        assert function in source, f"dashboard nie uzywa {function}"


def test_dashboard_has_no_hardcoded_statistics():
    """Zadnych recznych progow istotnosci ani wzorow w warstwie prezentacji."""
    source = DASHBOARD.read_text(encoding="utf-8")
    for pattern in ("p_value <", "np.mean", ".std(", "ddof=", "1.96"):
        assert pattern not in source, f"logika statystyczna w dashboardzie: {pattern}"


# --- grupa kontrolna w pipeline -------------------------------------------


def test_control_comparison_reports_missing_control_clearly(lab_config):
    """Brak grupy kontrolnej to nie blad - to instrukcja, co pobrac."""
    from pipeline import control_comparison

    report = control_comparison(load_lab_data(lab_config), config=lab_config)
    assert "error" in report
    assert "ingest --what control" in report["error"]


def test_control_comparison_runs_when_control_exists(lab_config, tmp_path):
    """Z aktywem kontrolnym w bazie porownanie ma zwrocic tabele i placebo."""
    from pipeline import control_comparison
    from storage import connect

    control = random_walk_prices(4000, start="2012-01-01", seed=123, initial=1000.0)
    control = control[pd.DatetimeIndex(control["date"]).dayofweek < 5].reset_index(drop=True)
    with connect(lab_config.db_path) as conn:
        store_prices(conn, control, "NASDAQ", "yahoo")

    data = load_lab_data(lab_config)
    assert data.has_controls

    report = control_comparison(data, post=180, config=lab_config)
    comparison = report["comparisons"]["NASDAQ"]
    assert not comparison.table.empty
    assert comparison.n_events >= 3
    assert set(comparison.per_event.columns) == {"BTCUSD", "NASDAQ", "difference"}
    # Roznica musi byc policzona parami, nie jako roznica srednich.
    expected = comparison.per_event["BTCUSD"] - comparison.per_event["NASDAQ"]
    assert np.allclose(comparison.per_event["difference"], expected)
    assert 0.0 <= comparison.summary_at["difference_p_value"] <= 1.0
    # Kalibracje testu sprawdza test_control_group; tutaj pojedyncze losowanie
    # nic by nie dowiodlo - przy alpha=0.05 co dwudzieste wyszloby istotne.
    assert not comparison.per_event.empty


# --- walidacja kroczaca w pipeline ----------------------------------------


def test_walk_forward_produces_many_disjoint_folds(lab_config):
    """Sens walk-forward: kilkanascie okien zamiast jednego podzialu."""
    from pipeline import walk_forward_check

    table = walk_forward_check(load_lab_data(lab_config), config=lab_config)
    assert not table.empty
    assert table.attrs["n_folds_total"] >= 5
    assert table.attrs["test_windows_disjoint"] is True
    assert {"sign_p_value", "sign_p_adjusted", "mean_test_effect"} <= set(table.columns)


def test_walk_forward_finds_nothing_on_noise(lab_config):
    """Baza testowa to czysty szum - nic nie ma prawa przezyc korekty."""
    from pipeline import walk_forward_check

    table = walk_forward_check(load_lab_data(lab_config), config=lab_config)
    assert not table["sign_significant_adjusted"].any()
    assert not table["test_effect_significant_adjusted"].any()


def test_overlapping_test_windows_suppress_the_t_test(lab_config):
    """Przy nakladajacych sie oknach test t jest niewazny i NIE moze byc podany.

    Zaklada on niezaleznosc obserwacji; gdy sasiednie okna testowe dziela
    polowe dni, p-value byloby zanizone. Pipeline wykrywa to na faktycznych
    indeksach, a nie na parametrach.
    """
    from pipeline import walk_forward_check

    settings = lab_config["validation"]["walk_forward"]
    original = settings["step_days"]
    settings["step_days"] = max(1, int(settings["test_days"]) // 2)
    try:
        table = walk_forward_check(load_lab_data(lab_config), config=lab_config)
        assert table.attrs["test_windows_disjoint"] is False
        assert table["test_effect_p_value"].isna().all()
        assert table["test_effect_p_adjusted"].isna().all()
        # Test znaku pozostaje wazny - nie zalezy od niezaleznosci okien.
        assert table["sign_p_value"].notna().any()
    finally:
        settings["step_days"] = original


def test_walk_forward_corrects_both_statistics(lab_config):
    """Obie statystyki sa hipotezami, wiec obie musza przejsc korekte."""
    from pipeline import walk_forward_check

    table = walk_forward_check(load_lab_data(lab_config), config=lab_config)
    sign = table[["sign_p_value", "sign_p_adjusted"]].dropna()
    assert (sign["sign_p_adjusted"] >= sign["sign_p_value"] - 1e-12).all()
    effect = table[["test_effect_p_value", "test_effect_p_adjusted"]].dropna()
    assert (effect["test_effect_p_adjusted"] >= effect["test_effect_p_value"] - 1e-12).all()
