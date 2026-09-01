"""Warstwa spinajaca moduly w kompletne przebiegi.

Cel: CLI i dashboard maja byc cienkie. Cala logika badawcza mieszka tutaj,
zeby wynik z terminala i wynik z przegladarki nie mogly sie rozjechac.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analysis.control import compare_with_control, placebo_event_study
from analysis.correlation import regime_returns
from analysis.event_study import circular_shift_test, event_study, window_scan
from backtest.engine import BacktestConfig, compare, run_backtest
from backtest.strategies import buy_and_hold, halving_window, macro_regime, trend_following
from config import load_config
from features.build import FeatureInputs, add_forward_returns, build_features
from features.macro_phase import macro_phase
from features.halving import CONFIRMED_HALVINGS
from ingest.prices import load_stitched
from storage import connect, read_events, read_macro, read_prices
from validation.multiple_testing import correct, summarize
from validation.splits import cycle_split, replicate_finding, split_frame

DEFAULT_HORIZON = 90


@dataclass
class LabData:
    prices: pd.DataFrame
    macro: pd.DataFrame
    events: pd.DataFrame
    features: pd.DataFrame
    controls: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.prices.empty

    @property
    def has_controls(self) -> bool:
        return any(not frame.empty for frame in self.controls.values())


def load_lab_data(config=None, *, horizons: list[int] | None = None) -> LabData:
    """Wczytuje wszystko z bazy i buduje ramke cech z celami."""
    config = config or load_config()
    symbol = config["price"]["symbol"]
    priority = config["price"].get("stitch_priority", config["price"]["sources"])

    control_config = config.get("control", {})
    control_source = control_config.get("source", "yahoo")

    with connect(config.db_path) as conn:
        prices = load_stitched(conn, symbol, priority)
        macro = read_macro(conn)
        events = read_events(conn)
        controls = {
            name: read_prices(conn, name, source=control_source)
            for name in control_config.get("symbols", {})
        }
    controls = {name: frame for name, frame in controls.items() if not frame.empty}

    if prices.empty:
        return LabData(prices, macro, events, pd.DataFrame(), controls)

    prices = prices.copy()
    prices["available_from"] = pd.to_datetime(prices["date"]) + pd.Timedelta(days=1)
    inputs = FeatureInputs(prices=prices, macro=macro, events=events)
    frame = build_features(
        inputs,
        event_windows=config["features"]["event_windows"],
        halving_window_list=config["features"]["halving_windows"],
    )
    frame = add_forward_returns(frame, horizons or [7, 30, 90, 180])
    return LabData(prices, macro, events, frame, controls)


def halving_event_study(data: LabData, *, pre: int = 30, post: int = 365, config=None):
    config = config or load_config()
    return event_study(
        data.prices,
        CONFIRMED_HALVINGS,
        pre=pre,
        post=post,
        n_boot=int(config["validation"]["bootstrap_iterations"]),
        seed=int(config["validation"]["random_seed"]),
    )


def category_event_studies(data: LabData, *, pre: int = 30, post: int = 90, config=None) -> dict:
    """Event study osobno dla kazdej kategorii zdarzen z rejestru."""
    config = config or load_config()
    results = {}
    if data.events.empty:
        return results
    for category, group in data.events.groupby("category"):
        result = event_study(
            data.prices,
            group["available_from"],
            pre=pre,
            post=post,
            n_boot=int(config["validation"]["bootstrap_iterations"]),
            seed=int(config["validation"]["random_seed"]),
        )
        if result.n_events:
            results[category] = result
    return results


def with_phase_dummies(frame: pd.DataFrame) -> pd.DataFrame:
    """Dokleja kolumny 0/1 dla kazdej fazy makro.

    Faza jest hipoteza dokladnie tak samo jak okno halvingowe, wiec musi
    przejsc przez te sama korekte na wielokrotne testowanie. Trzymanie jej
    poza skanem bylo by cichym uprzywilejowaniem: cztery dodatkowe testy,
    ktorych nikt nie liczy.
    """
    if "macro_phase" not in frame.columns:
        return frame
    labels = sorted(frame["macro_phase"].dropna().unique())
    if not labels:
        return frame
    out = frame.copy()
    for label in labels:
        out[f"phase_{label}"] = (out["macro_phase"] == label).astype(int)
    return out


def hypothesis_columns(frame: pd.DataFrame) -> list[str]:
    """Wszystkie okna i fazy, ktore traktujemy jako osobne hipotezy."""
    return [
        c for c in frame.columns
        if c.startswith("halving_after_")
        or c.startswith("event_")
        or c.startswith("phase_")
    ]


def scan_hypotheses(
    data: LabData, *, target: str = "log_return", config=None
) -> pd.DataFrame:
    """Skan wszystkich okien i faz + korekta na wielokrotne testowanie."""
    config = config or load_config()
    frame = with_phase_dummies(data.features)
    if frame.empty:
        return pd.DataFrame()
    columns = hypothesis_columns(frame)
    scan = window_scan(
        frame,
        columns,
        target,
        n_permutations=int(config["validation"]["bootstrap_iterations"]),
        seed=int(config["validation"]["random_seed"]),
    )
    return correct(
        scan,
        method=config["validation"]["fdr_method"],
        alpha=float(config["validation"]["alpha"]),
    )


def out_of_sample_check(
    data: LabData, *, target: str = "log_return", config=None
) -> pd.DataFrame:
    """Powtarza kazda hipoteze osobno na treningu i na tescie.

    Efekt, ktory istnieje tylko w probie treningowej, jest dopasowaniem do
    szumu - i tu to widac wprost, bez zadnej agregacji.
    """
    config = config or load_config()
    frame = with_phase_dummies(data.features)
    if frame.empty:
        return pd.DataFrame()

    split = cycle_split(
        frame.index,
        list(config["validation"]["train_cycles"]),
        list(config["validation"]["test_cycles"]),
        embargo_days=DEFAULT_HORIZON,
    )
    train, test = split_frame(frame, split)
    if train.empty or test.empty:
        return pd.DataFrame()

    rows = []
    permutations = int(config["validation"]["bootstrap_iterations"])
    for column in hypothesis_columns(frame):
        if train[column].sum() == 0 or test[column].sum() == 0:
            continue
        train_result = circular_shift_test(
            train[target], train[column] > 0, n_permutations=permutations
        )
        test_result = circular_shift_test(
            test[target], test[column] > 0, n_permutations=permutations
        )
        row = {"hypothesis": column, "split": split.name}
        row.update(replicate_finding(train_result, test_result,
                                     alpha=float(config["validation"]["alpha"])))
        row["train_p_value"] = train_result["p_value"]
        row["test_p_value"] = test_result["p_value"]
        rows.append(row)
    return pd.DataFrame(rows)


def macro_phase_comparison(data: LabData, config=None) -> dict:
    """Porownuje faze makro liczona z prawdziwego M2 i z proxy dolarowego.

    Sens tego porownania: dopoki nie ma klucza FRED, os plynnosci jest
    zastepowana odwroconym indeksem dolara. To rozsadne przyblizenie
    warunkow finansowych, ale NIE jest to podaz pieniadza. Ta funkcja mowi,
    jak bardzo obie wersje sie roznia - i czy wnioski od tego zaleza.
    """
    config = config or load_config()
    frame = data.features
    if frame.empty or data.macro.empty:
        return {"error": "brak danych makro"}

    available = set(data.macro["series"].unique())
    index = frame.index
    variants: dict[str, pd.Series] = {}
    if "m2" in available:
        variants["m2"] = macro_phase(
            data.macro, index, liquidity_source="m2_yoy"
        )["macro_phase"]
    if "dxy" in available:
        variants["proxy_dxy"] = macro_phase(
            data.macro, index, liquidity_source="dxy_chg_3m_inv"
        )["macro_phase"]

    if not variants:
        return {"error": "brak serii plynnosciowej (ani m2, ani dxy)"}

    returns = frame["log_return"]
    summary = {}
    for name, phases in variants.items():
        table = regime_returns(returns, phases)
        summary[name] = {
            "coverage_days": int(phases.notna().sum()),
            "first_labelled_day": (
                str(phases.dropna().index.min().date()) if phases.notna().any() else None
            ),
            "regimes": table,
        }

    result = {"variants": variants, "summary": summary}
    if len(variants) == 2:
        left, right = variants["m2"], variants["proxy_dxy"]
        both = left.notna() & right.notna()
        result["agreement"] = {
            "compared_days": int(both.sum()),
            "identical_label": float((left[both] == right[both]).mean()) if both.any() else float("nan"),
            "liquidity_axis_agrees": float(
                (
                    left[both].str.split("_").str[0] == right[both].str.split("_").str[0]
                ).mean()
            ) if both.any() else float("nan"),
        }
    return result


def control_comparison(
    data: LabData,
    *,
    event_dates=None,
    label: str = "halvingi",
    post: int = 365,
    config=None,
) -> dict:
    """Porownuje reakcje BTC z reakcja aktywow kontrolnych na te same daty.

    Zwraca slownik: nazwa kontroli -> ControlComparison, plus event study
    placebo dla kazdej kontroli. Brak kontroli w bazie nie jest bledem -
    zwracamy pusty wynik z informacja, co pobrac.
    """
    config = config or load_config()
    if data.is_empty:
        return {"error": "baza jest pusta"}
    if not data.has_controls:
        return {
            "error": "brak grupy kontrolnej w bazie - uruchom "
                     "`python -m cli ingest --what control`"
        }

    dates = CONFIRMED_HALVINGS if event_dates is None else pd.DatetimeIndex(
        pd.to_datetime(event_dates)
    )
    comparisons = {}
    placebos = {}
    for name, frame in data.controls.items():
        comparisons[name] = compare_with_control(
            data.prices,
            frame,
            dates,
            treatment_name=config["price"]["symbol"],
            control_name=name,
            post=post,
        )
        placebos[name] = placebo_event_study(
            frame,
            dates,
            post=post,
            n_boot=int(config["validation"]["bootstrap_iterations"]),
            seed=int(config["validation"]["random_seed"]),
        )
    return {"label": label, "comparisons": comparisons, "placebos": placebos}


def run_strategies(data: LabData, config=None) -> tuple[pd.DataFrame, list]:
    """Backtest kilku prostych strategii wzgledem kup-i-trzymaj."""
    config = config or load_config()
    frame = data.features
    if frame.empty:
        return pd.DataFrame(), []

    close = frame["close"]
    settings = BacktestConfig.from_config(config)
    results = [
        run_backtest(close, buy_and_hold(close.index), settings, name="kup i trzymaj"),
        run_backtest(close, trend_following(close), settings, name="trend 50/200"),
    ]
    for days in config["features"]["halving_windows"]:
        results.append(
            run_backtest(
                close,
                halving_window(close.index, days_after=days),
                settings,
                name=f"halving +{days}d",
            )
        )
    if "macro_phase" in frame.columns and frame["macro_phase"].notna().any():
        results.append(
            run_backtest(
                close,
                macro_regime(frame["macro_phase"]),
                settings,
                name="makro: plynnosc rosnie, stopy spadaja",
            )
        )
    return compare(results), results


def full_report(config=None) -> dict:
    """Jeden przebieg: dane -> event study -> walidacja -> backtest."""
    config = config or load_config()
    data = load_lab_data(config)
    if data.is_empty:
        return {"error": "baza jest pusta - uruchom najpierw `ingest`"}

    scan = scan_hypotheses(data, config=config)
    table, results = run_strategies(data, config=config)
    return {
        "data": data,
        "halving_study": halving_event_study(data, config=config),
        "category_studies": category_event_studies(data, config=config),
        "scan": scan,
        "scan_summary": summarize(scan) if not scan.empty else "brak hipotez",
        "out_of_sample": out_of_sample_check(data, config=config),
        "backtest_table": table,
        "backtest_results": results,
    }
