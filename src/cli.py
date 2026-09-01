r"""Interfejs wiersza polecen.

Na Windowsie uruchamiaj przez launcher, ktory sam przygotuje srodowisko:

    btc ingest --what all
    btc quality
    btc study --post 365
    btc control
    btc validate
    btc backtest
    btc all

Bez launchera dziala tez bezposrednie wywolanie - modul sam dokleja `src`
do sciezki, wiec nie trzeba ustawiac PYTHONPATH:

    .venv\Scripts\python.exe src\cli.py study --post 365

Kazda komenda zapisuje wynik do data/processed/ i wypisuje skrot na ekran.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config, secret  # noqa: E402
from ingest.events import load_events_csv, store_events  # noqa: E402
from ingest.macro import (  # noqa: E402
    MissingCredentials,
    fetch_fred,
    fetch_yahoo_series,
    load_manual_csv,
    store_macro,
)
from ingest.prices import (  # noqa: E402
    SOURCE_START,
    fetch_prices,
    load_stitched,
    stitch_sources,
    store_prices,
)
from ingest.quality import check_macro, check_prices, compare_sources  # noqa: E402
from pipeline import (  # noqa: E402
    category_event_studies,
    control_comparison,
    halving_event_study,
    load_lab_data,
    macro_phase_comparison,
    out_of_sample_check,
    run_strategies,
    scan_hypotheses,
)
from storage import connect, read_macro, read_prices, table_summary  # noqa: E402
from validation.multiple_testing import summarize  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)


def _processed_dir(config) -> Path:
    path = config.path("processed")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(frame: pd.DataFrame, config, name: str) -> Path:
    path = _processed_dir(config) / name
    frame.to_csv(path, index=frame.index.name is not None)
    return path


# --- komendy --------------------------------------------------------------


def cmd_ingest(args) -> int:
    config = load_config()
    symbol = config["price"]["symbol"]
    what = args.what

    with connect(config.db_path) as conn:
        if what in ("all", "prices"):
            for source in config["price"]["sources"]:
                start = args.start or max(config["price"]["start"], SOURCE_START.get(source, "2010-01-01"))
                print(f"[ceny] {source}: pobieram od {start} ...")
                try:
                    frame = fetch_prices(source, symbol, start, args.end)
                    rows = store_prices(conn, frame, symbol, source)
                    print(f"[ceny] {source}: {rows} barow")
                except Exception as exc:  # zrodlo moze byc chwilowo niedostepne
                    print(f"[ceny] {source}: BLAD {exc}")

        if what in ("all", "macro"):
            for name, ticker in config["macro"]["yahoo"].items():
                try:
                    frame = fetch_yahoo_series(ticker)
                    rows = store_macro(conn, frame, name, "yahoo")
                    print(f"[makro] {name} ({ticker}): {rows} obserwacji")
                except Exception as exc:
                    print(f"[makro] {name}: BLAD {exc}")

            for name, spec in config["macro"]["fred"].items():
                try:
                    frame = fetch_fred(
                        spec["series_id"],
                        publication_lag_days=int(spec["publication_lag_days"]),
                    )
                    rows = store_macro(conn, frame, name, "fred")
                    print(f"[makro] {name} ({spec['series_id']}): {rows} obserwacji")
                except MissingCredentials as exc:
                    print(f"[makro] pomijam FRED: {exc}")
                    break
                except Exception as exc:
                    print(f"[makro] {name}: BLAD {exc}")

            manual_dir = config.root / config["macro"]["manual_dir"]
            for path in sorted(manual_dir.glob("*.csv")):
                try:
                    frame = load_manual_csv(path)
                    rows = store_macro(conn, frame, path.stem, "manual")
                    print(f"[makro] {path.stem} (reczne): {rows} obserwacji")
                except Exception as exc:
                    print(f"[makro] {path.name}: BLAD {exc}")

        if what in ("all", "control"):
            control_config = config.get("control", {})
            source = control_config.get("source", "yahoo")
            start = args.start or control_config.get("start", "2011-01-01")
            for name in control_config.get("symbols", {}):
                try:
                    frame = fetch_prices(source, name, start, args.end)
                    rows = store_prices(conn, frame, name, source)
                    print(f"[kontrola] {name}: {rows} sesji")
                except Exception as exc:
                    print(f"[kontrola] {name}: BLAD {exc}")

        if what in ("all", "events"):
            events = load_events_csv(config.root / "data" / "raw" / "events.csv")
            rows = store_events(conn, events)
            print(f"[zdarzenia] {rows} pozycji")

        print("\n--- zawartosc bazy ---")
        print(table_summary(conn).to_string(index=False))
    return 0


def cmd_quality(args) -> int:
    config = load_config()
    symbol = config["price"]["symbol"]
    priority = config["price"].get("stitch_priority", config["price"]["sources"])

    with connect(config.db_path) as conn:
        frames = {}
        for source in priority:
            frame = read_prices(conn, symbol, source=source)
            if frame.empty:
                print(f"[jakosc] {source}: brak danych")
                continue
            frames[source] = frame
            print("[jakosc]", check_prices(frame, name=source).summary())

        if len(frames) >= 2:
            stitched, report = stitch_sources(frames, priority)
            print("\n--- zszycie zrodel ---")
            print(report.to_string(index=False))
            overlap = report.attrs["overlap"]
            if not overlap.empty:
                print(overlap.to_string(index=False))
                tolerance = float(config["price"].get("overlap_tolerance", 0.01))
                for _, row in overlap.iterrows():
                    if row["median_rel_diff"] > tolerance:
                        print(
                            f"UWAGA: {row['pair']} rozjezdza sie o "
                            f"{row['median_rel_diff']:.2%} (prog {tolerance:.2%})"
                        )
            print("\n[jakosc]", check_prices(stitched, name="zszyty szereg").summary())

            first, second = priority[0], priority[1]
            if first in frames and second in frames:
                divergence = compare_sources(frames[first], frames[second], tolerance=0.05)
                print(f"[jakosc] dni z rozbieznoscia >5% ({first} vs {second}): {len(divergence)}")

        macro = read_macro(conn)
        if not macro.empty:
            print("\n--- makro ---")
            rows = [check_macro(group, name=name) for name, group in macro.groupby("series")]
            print(pd.DataFrame(rows).to_string(index=False))
    return 0


def cmd_features(args) -> int:
    config = load_config()
    data = load_lab_data(config)
    if data.is_empty:
        print("baza jest pusta - uruchom najpierw `ingest`")
        return 1
    path = _save(data.features, config, "features.csv")
    print(f"cechy: {data.features.shape[0]} dni x {data.features.shape[1]} kolumn -> {path}")
    print(data.features.tail(3).to_string())
    return 0


def cmd_study(args) -> int:
    config = load_config()
    data = load_lab_data(config)
    if data.is_empty:
        print("baza jest pusta - uruchom najpierw `ingest`")
        return 1

    result = halving_event_study(data, pre=args.pre, post=args.post, config=config)
    print("--- halvingi ---")
    print(result.summary())
    if len(result.skipped_events):
        print(f"pominiete (brak pelnego okna): {[str(d.date()) for d in result.skipped_events]}")
    if not result.table.empty:
        path = _save(result.table, config, "event_study_halving.csv")
        milestones = [d for d in (0, 30, 90, 180, 365) if d in result.table.index]
        print(
            result.table.loc[
                milestones, ["car", "car_ci_low", "car_ci_high", "car_p_value", "n_events"]
            ].to_string()
        )
        print(f"-> {path}")

    print("\n--- kategorie zdarzen ---")
    rows = []
    for category, study in category_event_studies(data, post=args.post, config=config).items():
        rows.append({"kategoria": category, "n": study.n_events, **study.car_summary})
    if rows:
        table = pd.DataFrame(rows)
        print(table.to_string(index=False))
        _save(table, config, "event_study_categories.csv")
        print(
            "\nUWAGA: powyzsze p-value NIE sa skorygowane o liczbe testow. "
            "Do wnioskowania uzyj `python -m cli validate`."
        )
    return 0


def cmd_validate(args) -> int:
    config = load_config()
    data = load_lab_data(config)
    if data.is_empty:
        print("baza jest pusta - uruchom najpierw `ingest`")
        return 1

    scan = scan_hypotheses(data, config=config)
    if scan.empty:
        print("brak hipotez do sprawdzenia")
        return 1
    print("--- skan hipotez (cel: dzienny zwrot logarytmiczny) ---")
    print(
        scan.loc[
            :, ["hypothesis", "n_in", "mean_in", "mean_out", "difference",
                "p_value", "p_adjusted", "significant_adjusted"]
        ].to_string(index=False)
    )
    print("\n" + summarize(scan))
    _save(scan, config, "hypothesis_scan.csv")

    out_of_sample = out_of_sample_check(data, config=config)
    if not out_of_sample.empty:
        print("\n--- replikacja poza proba (podzial po cyklach) ---")
        print(
            out_of_sample.loc[
                :, ["hypothesis", "train_effect", "test_effect", "same_sign",
                    "significant_out_of_sample", "effect_retained", "replicated"]
            ].to_string(index=False)
        )
        survivors = out_of_sample[out_of_sample["replicated"]]["hypothesis"].tolist()
        print(f"\nprzetrwalo out-of-sample: {survivors or 'nic'}")
        _save(out_of_sample, config, "out_of_sample.csv")
    return 0


def cmd_macro(args) -> int:
    """Stan osi plynnosci: czym jest liczona i czy wybor zrodla zmienia wnioski."""
    config = load_config()

    if args.check_key:
        key = secret("FRED_API_KEY")
        if not key:
            print("FRED_API_KEY: brak. Wklej klucz do .env (wzor w .env.example).")
            return 1
        print(f"FRED_API_KEY: obecny ({len(key)} znakow)")
        try:
            probe = fetch_fred("M2SL", publication_lag_days=30)
        except Exception as exc:
            print(f"FRED: klucz odrzucony albo API niedostepne -> {exc}")
            return 1
        lag = (probe["available_from"] - probe["date"]).dt.days
        print(
            f"FRED: OK, {len(probe)} obserwacji M2SL "
            f"({probe['date'].min().date()} -> {probe['date'].max().date()})"
        )
        print(
            f"  daty publikacji z archiwum wersji: {probe.attrs.get('vintage_rows', 0)}, "
            f"ze stalego opoznienia: {probe.attrs.get('fallback_rows', 0)}"
        )
        print(f"  mediana opoznienia publikacji: {lag.median():.0f} dni, maks {lag.max():.0f}")
        return 0

    data = load_lab_data(config)
    if data.is_empty:
        print("baza jest pusta - uruchom najpierw `ingest`")
        return 1

    report = macro_phase_comparison(data, config=config)
    if "error" in report:
        print(report["error"])
        return 1

    for name, summary in report["summary"].items():
        label = "M2 (FRED)" if name == "m2" else "proxy: odwrocony indeks dolara"
        print(f"\n--- os plynnosci: {label} ---")
        print(
            f"dni z etykieta: {summary['coverage_days']} "
            f"(od {summary['first_labelled_day']})"
        )
        print(summary["regimes"].to_string(float_format=lambda v: f"{v:,.4f}"))

    if "agreement" in report:
        agreement = report["agreement"]
        print("\n--- zgodnosc obu wersji ---")
        print(f"porownane dni: {agreement['compared_days']}")
        print(f"identyczna etykieta fazy: {agreement['identical_label']:.1%}")
        print(f"sama os plynnosci zgodna: {agreement['liquidity_axis_agrees']:.1%}")
        if agreement["liquidity_axis_agrees"] < 0.7:
            print(
                "\nProxy dolarowe i M2 opisuja rozne rzeczy - wnioski z fazy makro "
                "policzonej na proxy NIE przenosza sie na wersje z M2."
            )
    else:
        print(
            "\nBrak drugiej wersji do porownania. Zeby policzyc faze na prawdziwym M2, "
            "wklej klucz FRED do .env i uruchom:\n"
            "  python -m cli ingest --what macro"
        )
    return 0


def cmd_control(args) -> int:
    """Test placebo: czy grupa kontrolna reaguje na halvingi tak samo jak BTC."""
    config = load_config()
    data = load_lab_data(config)
    if data.is_empty:
        print("baza jest pusta - uruchom najpierw `ingest`")
        return 1

    dates = None
    label = "halvingi"
    if args.category:
        if data.events.empty:
            print("brak rejestru zdarzen")
            return 1
        subset = data.events[data.events["category"] == args.category]
        if subset.empty:
            available = sorted(data.events["category"].unique())
            print(f"brak zdarzen w kategorii {args.category}; dostepne: {available}")
            return 1
        dates = subset["available_from"]
        label = args.category

    report = control_comparison(
        data, event_dates=dates, label=label, post=args.post, config=config
    )
    if "error" in report:
        print(report["error"])
        return 1

    alpha = float(config["validation"]["alpha"])
    for name, comparison in report["comparisons"].items():
        print(f"\n--- {config['price']['symbol']} vs {name} wokol: {report['label']} ---")
        if comparison.table.empty:
            print("brak wspolnych zdarzen z pelnym oknem")
            continue
        print(
            comparison.table.loc[
                :, [c for c in comparison.table.columns if c != "difference_t_stat"]
            ].to_string(float_format=lambda v: f"{v:,.4f}")
        )
        print("\n" + comparison.verdict(alpha))

        placebo = report["placebos"][name]
        if placebo.car_summary:
            print(
                f"placebo ({name} traktowany jak BTC): CAR({placebo.car_summary['offset']}d) = "
                f"{placebo.car_summary['car']:+.1%}, p={placebo.car_summary['p_value']:.3f}"
            )
        _save(comparison.per_event, config, f"control_{name}_per_event.csv")
        _save(comparison.table, config, f"control_{name}.csv")

    print(
        "\nJak to czytac: jesli roznica nie odstaje od zera, to co widac wokol "
        "halvingu jest wspolne dla rynkow ryzyka, a nie specyficzne dla bitcoina."
    )
    return 0


def cmd_backtest(args) -> int:
    config = load_config()
    data = load_lab_data(config)
    if data.is_empty:
        print("baza jest pusta - uruchom najpierw `ingest`")
        return 1

    table, results = run_strategies(data, config=config)
    print("--- backtest (koszty: {} bps prowizji + {} bps poslizgu) ---".format(
        config["backtest"]["fee_bps"], config["backtest"]["slippage_bps"]
    ))
    for result in results:
        print(result.summary())
    print()
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))
    _save(table, config, "backtest_comparison.csv")

    baseline = table.loc["kup i trzymaj"]
    better = table[table["sharpe"] > baseline["sharpe"]].index.tolist()
    better = [name for name in better if name != "kup i trzymaj"]
    print(f"\nlepszy Sharpe niz kup-i-trzymaj: {better or 'zadna'}")
    return 0


def cmd_all(args) -> int:
    for command in (cmd_quality, cmd_features, cmd_macro, cmd_study, cmd_control,
                    cmd_validate, cmd_backtest):
        print("\n" + "=" * 78)
        code = command(args)
        if code != 0:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btc-cycle-lab", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="pobiera dane do lokalnej bazy")
    ingest.add_argument(
        "--what", choices=["all", "prices", "macro", "events", "control"], default="all"
    )
    ingest.add_argument("--start", default=None, help="data poczatkowa (YYYY-MM-DD)")
    ingest.add_argument("--end", default=None)
    ingest.set_defaults(func=cmd_ingest)

    quality = subparsers.add_parser("quality", help="raport jakosci danych")
    quality.set_defaults(func=cmd_quality)

    features = subparsers.add_parser("features", help="buduje i zapisuje ramke cech")
    features.set_defaults(func=cmd_features)

    study = subparsers.add_parser("study", help="event study wokol halvingow i zdarzen")
    study.add_argument("--pre", type=int, default=30)
    study.add_argument("--post", type=int, default=365)
    study.set_defaults(func=cmd_study)

    validate = subparsers.add_parser("validate", help="skan hipotez + korekta + out-of-sample")
    validate.set_defaults(func=cmd_validate)

    control = subparsers.add_parser(
        "control", help="grupa kontrolna: czy NASDAQ reaguje na halvingi tak samo"
    )
    control.add_argument("--post", type=int, default=365)
    control.add_argument(
        "--category", default=None,
        help="zamiast halvingow uzyj zdarzen z tej kategorii (np. credit_event)",
    )
    control.set_defaults(func=cmd_control)

    macro = subparsers.add_parser("macro", help="os plynnosci: M2 vs proxy, stan klucza FRED")
    macro.add_argument(
        "--check-key", action="store_true",
        help="sprawdza, czy FRED_API_KEY dziala (odpytuje API)",
    )
    macro.set_defaults(func=cmd_macro)

    backtest = subparsers.add_parser("backtest", help="strategie vs kup-i-trzymaj")
    backtest.set_defaults(func=cmd_backtest)

    everything = subparsers.add_parser("all", help="pelny przebieg (bez pobierania danych)")
    everything.add_argument("--pre", type=int, default=30)
    everything.add_argument("--post", type=int, default=365)
    everything.add_argument("--check-key", action="store_true", help=argparse.SUPPRESS)
    everything.add_argument("--category", default=None, help=argparse.SUPPRESS)
    everything.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
