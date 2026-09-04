r"""Command-line interface.

Use the cross-platform launcher, which prepares the environment for you:

    python run.py ingest --what all
    python run.py quality
    python run.py study --post 365
    python run.py control
    python run.py validate
    python run.py walkforward
    python run.py backtest
    python run.py all

On Windows `btc.cmd` and on macOS/Linux `./btc` are thin wrappers around it.

Calling this module directly works too - it appends `src` to the import path
itself, so PYTHONPATH is not needed:

    .venv/bin/python src/cli.py study --post 365

Every command writes its output to data/processed/ and prints a summary.
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
from analysis.specification import (  # noqa: E402
    build_grid,
    curve_statistics,
    permutation_test,
    run_curve,
    verdict,
)
from ingest.quality import check_macro, check_prices, compare_sources  # noqa: E402
from pipeline import (  # noqa: E402
    category_event_studies,
    control_comparison,
    forecast_report,
    halving_event_study,
    load_lab_data,
    macro_phase_comparison,
    out_of_sample_check,
    run_strategies,
    scan_hypotheses,
    walk_forward_check,
)
from storage import connect, read_macro, read_prices, table_summary  # noqa: E402
from validation.multiple_testing import summarize  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)


def _make_output_unicode_safe() -> None:
    """Never let a non-ASCII character kill a command.

    run.py already starts children in UTF-8 mode, but this module is also
    meant to be runnable directly, and then a legacy Windows console code page
    (cp1252 on a Western or Central European install) turns one em dash in an
    event description into a UnicodeEncodeError that ends the run.

    `errors="replace"` is deliberate: a question mark in place of a dash is a
    cosmetic problem, a traceback instead of the analysis is not.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # Already wrapped, or not a real stream (captured in tests).


_make_output_unicode_safe()


def _processed_dir(config) -> Path:
    path = config.path("processed")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(frame: pd.DataFrame, config, name: str) -> Path:
    path = _processed_dir(config) / name
    frame.to_csv(path, index=frame.index.name is not None)
    return path


# --- commands --------------------------------------------------------------


def cmd_ingest(args) -> int:
    config = load_config()
    symbol = config["price"]["symbol"]
    what = args.what

    with connect(config.db_path) as conn:
        if what in ("all", "prices"):
            for source in config["price"]["sources"]:
                start = args.start or max(config["price"]["start"], SOURCE_START.get(source, "2010-01-01"))
                print(f"[prices] {source}: fetching from {start} ...")
                try:
                    frame = fetch_prices(source, symbol, start, args.end)
                    rows = store_prices(conn, frame, symbol, source)
                    print(f"[prices] {source}: {rows} bars")
                except Exception as exc:  # a source may be temporarily unavailable
                    print(f"[prices] {source}: ERROR {exc}")

        if what in ("all", "macro"):
            for name, ticker in config["macro"]["yahoo"].items():
                try:
                    frame = fetch_yahoo_series(ticker)
                    rows = store_macro(conn, frame, name, "yahoo")
                    print(f"[macro] {name} ({ticker}): {rows} observations")
                except Exception as exc:
                    print(f"[macro] {name}: ERROR {exc}")

            for name, spec in config["macro"]["fred"].items():
                try:
                    frame = fetch_fred(
                        spec["series_id"],
                        publication_lag_days=int(spec["publication_lag_days"]),
                    )
                    rows = store_macro(conn, frame, name, "fred")
                    print(f"[macro] {name} ({spec['series_id']}): {rows} observations")
                except MissingCredentials as exc:
                    print(f"[macro] skipping FRED: {exc}")
                    break
                except Exception as exc:
                    print(f"[macro] {name}: ERROR {exc}")

            manual_dir = config.root / config["macro"]["manual_dir"]
            for path in sorted(manual_dir.glob("*.csv")):
                try:
                    frame = load_manual_csv(path)
                    rows = store_macro(conn, frame, path.stem, "manual")
                    print(f"[macro] {path.stem} (manual): {rows} observations")
                except Exception as exc:
                    print(f"[macro] {path.name}: ERROR {exc}")

        if what in ("all", "control"):
            control_config = config.get("control", {})
            source = control_config.get("source", "yahoo")
            start = args.start or control_config.get("start", "2011-01-01")
            for name in control_config.get("symbols", {}):
                try:
                    frame = fetch_prices(source, name, start, args.end)
                    rows = store_prices(conn, frame, name, source)
                    print(f"[control] {name}: {rows} sessions")
                except Exception as exc:
                    print(f"[control] {name}: ERROR {exc}")

        if what in ("all", "events"):
            events = load_events_csv(config.root / "data" / "raw" / "events.csv")
            rows = store_events(conn, events)
            print(f"[events] {rows} entries")

        print("\n--- database contents ---")
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
                print(f"[quality] {source}: no data")
                continue
            frames[source] = frame
            print("[quality]", check_prices(frame, name=source).summary())

        if len(frames) >= 2:
            stitched, report = stitch_sources(frames, priority)
            print("\n--- source stitching ---")
            print(report.to_string(index=False))
            overlap = report.attrs["overlap"]
            if not overlap.empty:
                print(overlap.to_string(index=False))
                tolerance = float(config["price"].get("overlap_tolerance", 0.01))
                for _, row in overlap.iterrows():
                    if row["median_rel_diff"] > tolerance:
                        print(
                            f"WARNING: {row['pair']} diverges by "
                            f"{row['median_rel_diff']:.2%} (threshold {tolerance:.2%})"
                        )
            print("\n[quality]", check_prices(stitched, name="stitched series").summary())

            first, second = priority[0], priority[1]
            if first in frames and second in frames:
                divergence = compare_sources(frames[first], frames[second], tolerance=0.05)
                print(f"[quality] days diverging >5% ({first} vs {second}): {len(divergence)}")

        macro = read_macro(conn)
        if not macro.empty:
            print("\n--- macro ---")
            rows = [check_macro(group, name=name) for name, group in macro.groupby("series")]
            print(pd.DataFrame(rows).to_string(index=False))
    return 0


def cmd_features(args) -> int:
    config = load_config()
    data = load_lab_data(config)
    if data.is_empty:
        print("the database is empty - run `ingest` first")
        return 1
    path = _save(data.features, config, "features.csv")
    print(f"features: {data.features.shape[0]} days x {data.features.shape[1]} columns -> {path}")
    print(data.features.tail(3).to_string())
    return 0


def cmd_study(args) -> int:
    config = load_config()
    data = load_lab_data(config)
    if data.is_empty:
        print("the database is empty - run `ingest` first")
        return 1

    result = halving_event_study(data, pre=args.pre, post=args.post, config=config)
    print("--- halvings ---")
    print(result.summary())
    if len(result.skipped_events):
        print(f"skipped (incomplete window): {[str(d.date()) for d in result.skipped_events]}")
    if not result.table.empty:
        path = _save(result.table, config, "event_study_halving.csv")
        milestones = [d for d in (0, 30, 90, 180, 365) if d in result.table.index]
        print(
            result.table.loc[
                milestones, ["car", "car_ci_low", "car_ci_high", "car_p_value", "n_events"]
            ].to_string()
        )
        print(f"-> {path}")

    print("\n--- event categories ---")
    rows = []
    for category, study in category_event_studies(data, post=args.post, config=config).items():
        rows.append({"category": category, "n": study.n_events, **study.car_summary})
    if rows:
        table = pd.DataFrame(rows)
        print(table.to_string(index=False))
        _save(table, config, "event_study_categories.csv")
        print(
            "\nWARNING: the p-values above are NOT corrected for the number of "
            "tests. For inference use `run.py validate`."
        )
    return 0


def cmd_validate(args) -> int:
    config = load_config()
    data = load_lab_data(config)
    if data.is_empty:
        print("the database is empty - run `ingest` first")
        return 1

    scan = scan_hypotheses(data, config=config)
    if scan.empty:
        print("no hypotheses to check")
        return 1
    print("--- hypothesis scan (target: daily log return) ---")
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
        print("\n--- out-of-sample replication (split by cycle) ---")
        print(
            out_of_sample.loc[
                :, ["hypothesis", "train_effect", "test_effect", "same_sign",
                    "significant_in_train", "significant_out_of_sample",
                    "effect_retained", "replicated"]
            ].to_string(index=False)
        )
        survivors = out_of_sample[out_of_sample["replicated"]]["hypothesis"].tolist()
        print(f"\nsurvived out of sample: {survivors or 'nothing'}")
        _save(out_of_sample, config, "out_of_sample.csv")
    return 0


def cmd_macro(args) -> int:
    """Liquidity axis: what it is computed from, and whether the choice matters."""
    config = load_config()

    if args.check_key:
        key = secret("FRED_API_KEY")
        if not key:
            print("FRED_API_KEY: missing. Paste a key into .env (template in .env.example).")
            return 1
        print(f"FRED_API_KEY: present ({len(key)} characters)")
        try:
            probe = fetch_fred("M2SL", publication_lag_days=30)
        except Exception as exc:
            print(f"FRED: key rejected or API unreachable -> {exc}")
            return 1
        lag = (probe["available_from"] - probe["date"]).dt.days
        print(
            f"FRED: OK, {len(probe)} M2SL observations "
            f"({probe['date'].min().date()} -> {probe['date'].max().date()})"
        )
        print(
            f"  publication dates from the vintage archive: {probe.attrs.get('vintage_rows', 0)}, "
            f"from the fixed lag: {probe.attrs.get('fallback_rows', 0)}"
        )
        print(f"  median publication lag: {lag.median():.0f} days, max {lag.max():.0f}")
        return 0

    data = load_lab_data(config)
    if data.is_empty:
        print("the database is empty - run `ingest` first")
        return 1

    report = macro_phase_comparison(data, config=config)
    if "error" in report:
        print(report["error"])
        return 1

    for name, summary in report["summary"].items():
        label = "M2 (FRED)" if name == "m2" else "proxy: inverted dollar index"
        print(f"\n--- liquidity axis: {label} ---")
        print(
            f"labelled days: {summary['coverage_days']} "
            f"(from {summary['first_labelled_day']})"
        )
        print(summary["regimes"].to_string(float_format=lambda v: f"{v:,.4f}"))

    if "agreement" in report:
        agreement = report["agreement"]
        print("\n--- agreement between the two versions ---")
        print(f"days compared: {agreement['compared_days']}")
        print(f"identical phase label: {agreement['identical_label']:.1%}")
        print(f"liquidity axis alone agrees: {agreement['liquidity_axis_agrees']:.1%}")
        if agreement["liquidity_axis_agrees"] < 0.7:
            print(
                "\nThe dollar proxy and M2 describe different things - conclusions "
                "from a macro phase computed on the proxy do NOT carry over to M2."
            )
    else:
        print(
            "\nNo second version to compare against. To compute the phase on real "
            "M2, paste a FRED key into .env and run:\n"
            "  run.py ingest --what macro"
        )
    return 0


def cmd_walkforward(args) -> int:
    """Walk-forward validation: a dozen or more disjoint test windows."""
    config = load_config()
    data = load_lab_data(config)
    if data.is_empty:
        print("the database is empty - run `ingest` first")
        return 1

    table = walk_forward_check(data, config=config)
    if table.empty:
        print("not enough data for walk-forward validation")
        return 1

    folds = table.attrs["folds"]
    disjoint = table.attrs["test_windows_disjoint"]
    print(f"--- {table.attrs['n_folds_total']} folds, test windows disjoint: {disjoint} ---")
    print(folds.loc[:, ["split", "train_days", "train_span", "test_days", "test_span"]]
          .to_string(index=False))
    if not disjoint:
        print(
            "\nWARNING: the test windows overlap, so the t-test across folds is "
            "invalid and was not computed. Raise step_days in config.yaml."
        )

    print("\n--- sign stability of the effect from training to test ---")
    columns = ["hypothesis", "n_folds", "n_same_sign", "sign_agreement",
               "sign_p_value", "sign_p_adjusted", "mean_test_effect"]
    if disjoint:
        columns += ["test_effect_p_value", "test_effect_p_adjusted"]
    print(table.loc[:, columns].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    sign_hits = table[table["sign_significant_adjusted"]]["hypothesis"].tolist()
    effect_hits = table[table["test_effect_significant_adjusted"]]["hypothesis"].tolist()
    print(f"\nafter {config['validation']['fdr_method']} correction:")
    print(f"  stable sign        : {sign_hits or 'nothing'}")
    print(f"  out-of-sample effect: {effect_hits or 'nothing'}")

    best = table.iloc[0]
    if int(best["n_folds"]) < 6:
        print(
            f"\nPower limit: the best hypothesis ({best['hypothesis']}) has only "
            f"{int(best['n_folds'])} folds. Even perfect sign agreement would give "
            f"p={0.5 ** (int(best['n_folds']) - 1):.4f}, so at this many windows "
            "significance is unreachable."
        )
    _save(table, config, "walk_forward.csv")
    _save(folds, config, "walk_forward_folds.csv")
    return 0


def cmd_speccurve(args) -> int:
    """Vary the analytical choices instead of the hypothesis.

    Answers the one objection a null result always attracts: "you picked the
    wrong window."
    """
    config = load_config()
    data = load_lab_data(config)
    if data.is_empty:
        print("the database is empty - run `ingest` first")
        return 1
    if data.events.empty:
        print("no event registry - run `ingest` first")
        return 1

    halvings = pd.DatetimeIndex(
        pd.to_datetime(data.events[data.events["category"] == "halving"]["date"])
    ).normalize()
    if len(halvings) < 2:
        print(f"only {len(halvings)} halving(s) in the registry - nothing to vary")
        return 1

    symbol = config["price"]["symbol"]
    priority = config["price"].get("stitch_priority", config["price"]["sources"])
    frames = {"stitched": data.prices}
    with connect(config.db_path) as conn:
        for source in priority:
            frame = read_prices(conn, symbol, source=source)
            if not frame.empty:
                frames[source] = frame

    grid = build_grid(sorted(frames))
    print(f"--- specification curve: {len(grid)} specifications ---")
    print(
        "Varying the analytical choices the hypothesis scan holds fixed: "
        "price series, horizon, return type, abnormal or raw, estimation window."
    )
    for name, frame in sorted(frames.items()):
        dates = pd.to_datetime(frame["date"])
        covered = int(((halvings >= dates.min()) & (halvings <= dates.max())).sum())
        print(f"  {name:<10} {len(frame):>5} days from {dates.min().date()} "
              f"| covers {covered}/{len(halvings)} halvings")

    curve = run_curve(frames, halvings, grid=grid)
    stats = curve_statistics(curve)
    _save(curve, config, "specification_curve.csv")

    print(f"\nmedian CAR across specifications : {stats['median_car']:+.1%}")
    print(f"specifications with a positive CAR: {stats['share_positive']:.0%}")
    print(f"significant at 5% (uncorrected)   : {stats['n_significant']}/{stats['n_specs']} "
          f"({stats['share_significant']:.1%})")
    print(f"events actually used              : {stats['n_events_min']}-{stats['n_events_max']} "
          "(the shorter exchange histories cover fewer halvings)")

    print("\n--- most and least favourable specifications ---")
    columns = ["label", "n_events", "car", "ci_low", "ci_high", "p_value"]
    print(curve.tail(3)[columns].to_string(index=False))
    print("   ...")
    print(curve.head(3)[columns].to_string(index=False))

    print(
        f"\nRunning {args.permutations} permutations of the curve. The count "
        "above is NOT a test: these specifications re-analyse the same events\n"
        "and move together, so their share significant is not binomial. "
        "Inference has to be made on the whole curve."
    )
    permutation = permutation_test(
        frames, halvings, stats, n_permutations=args.permutations, grid=grid
    )
    if permutation.get("n_permutations"):
        print("\nunder the null (the same 4 dates, circularly shifted as a block):")
        print(f"  {permutation['share_wrapped']:.0%} of draws wrapped past the end of "
              "the history, so they keep the gaps\n  between events circularly rather "
              "than on the calendar")
        print(f"  |median CAR| exceeded the observed one in "
              f"{permutation['median_p_value']:.1%} of draws")
        print(f"  significant specifications: {permutation['null_significant_mean']:.1f} on "
              f"average, {permutation['null_significant_p95']:.0f} at the 95th percentile "
              f"(observed: {stats['n_significant']})")
        _save(pd.DataFrame([{**stats, **permutation}]), config, "specification_summary.csv")
        # A short run overwrites the saved summary with the same filename as a
        # full one. That is how a `--permutations 5` smoke test silently
        # replaced a committed 200-draw result during development. The count
        # travels in the file and onto the dashboard, and this says so out
        # loud, so a low number cannot be mistaken for the real thing later.
        if args.permutations < 100:
            print(
                f"\nWARNING: {args.permutations} permutations is a smoke test, not "
                "a result, and it has\njust overwritten specification_summary.csv. "
                "Re-run with the default 200 before\nquoting or committing these "
                "numbers."
            )

    print(f"\nVERDICT: {verdict(stats, permutation)}")
    return 0


def cmd_control(args) -> int:
    """Placebo test: does the control group react to halvings the way BTC does?"""
    config = load_config()
    data = load_lab_data(config)
    if data.is_empty:
        print("the database is empty - run `ingest` first")
        return 1

    dates = None
    label = "halvings"
    if args.category:
        if data.events.empty:
            print("no event registry")
            return 1
        subset = data.events[data.events["category"] == args.category]
        if subset.empty:
            available = sorted(data.events["category"].unique())
            print(f"no events in category {args.category}; available: {available}")
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
        print(f"\n--- {config['price']['symbol']} vs {name} around: {report['label']} ---")
        if comparison.table.empty:
            print("no shared events with a complete window")
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
                f"placebo ({name} treated like BTC): CAR({placebo.car_summary['offset']}d) = "
                f"{placebo.car_summary['car']:+.1%}, p={placebo.car_summary['p_value']:.3f}"
            )
        _save(comparison.per_event, config, f"control_{name}_per_event.csv")
        _save(comparison.table, config, f"control_{name}.csv")

    print(
        "\nHow to read this: if the difference is indistinguishable from zero, "
        "what happens around a halving is common to risk assets rather than "
        "specific to Bitcoin."
    )
    return 0


def cmd_forecast(args) -> int:
    """Directional forecast and, more importantly, whether it beats the baselines."""
    config = load_config()
    data = load_lab_data(config)
    if data.is_empty:
        print("the database is empty - run `ingest` first")
        return 1

    report = forecast_report(data, config=config)
    if "error" in report:
        print(report["error"])
        return 1

    run = report["run"]
    horizon = report["horizon"]
    print(f"--- directional forecast, {horizon}-day horizon, {run.n_folds} folds ---")
    print(
        run.folds.loc[:, ["fold", "n_train", "alpha", "n", "brier", "accuracy",
                          "auc", "base_rate"]]
        .to_string(index=False, float_format=lambda v: f"{v:,.3f}")
    )

    print("\n--- pooled, non-overlapping rows only ---")
    print(
        run.pooled.loc[:, ["n", "brier", "log_loss", "accuracy", "auc",
                           "mean_probability", "base_rate"]]
        .to_string(float_format=lambda v: f"{v:,.4f}")
    )
    print(
        f"\n(every row, including overlapping windows: "
        f"n={int(run.pooled_all_rows.loc['model', 'n'])} - the number above is "
        "the honest one, because a 30-day label on consecutive days repeats "
        "almost all of itself)"
    )

    print("\n--- calibration: predicted vs actual ---")
    if run.calibration.empty:
        print("not enough data")
    else:
        print(run.calibration.to_string(float_format=lambda v: f"{v:,.3f}"))

    print("\n" + "=" * 70)
    print("VERDICT: " + run.summary())
    print("=" * 70)

    latest = report["latest"]
    if "error" not in latest:
        print(
            f"\nMost recent prediction ({latest['as_of'].date()}): "
            f"{latest['probability_up']:.1%} chance the next {horizon} days are up."
        )
        print(
            f"Training base rate: {latest['train_base_rate']:.1%}. "
            f"Difference: {latest['edge_over_base_rate']:+.1%} points."
        )
        if "NO EDGE" in run.summary():
            print(
                "\nRead that number as decoration. The walk-forward evaluation "
                "above says this model has no edge out of sample, so the "
                "probability is not evidence about the future - it is what a "
                "model with no demonstrated skill happens to output today."
            )

    print("\n--- strongest coefficients (standardised, last fold) ---")
    if run.weights.empty:
        print("no model")
    else:
        print(run.weights.head(8).to_string(float_format=lambda v: f"{v:+.4f}"))
        print(
            "Direction and rough size only. With collinear predictors and a "
            "penalty individual coefficients are not identified - and none of "
            "it matters if the verdict above is NO EDGE."
        )

    _save(run.folds, config, "forecast_folds.csv")
    _save(run.pooled, config, "forecast_pooled.csv")
    if not run.calibration.empty:
        _save(run.calibration, config, "forecast_calibration.csv")
    return 0


def cmd_backtest(args) -> int:
    config = load_config()
    data = load_lab_data(config)
    if data.is_empty:
        print("the database is empty - run `ingest` first")
        return 1

    table, results = run_strategies(data, config=config)
    print("--- backtest ({} bps fees + {} bps slippage) ---".format(
        config["backtest"]["fee_bps"], config["backtest"]["slippage_bps"]
    ))
    for result in results:
        print(result.summary())
    print()
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))
    _save(table, config, "backtest_comparison.csv")

    baseline = table.loc["buy and hold"]
    better = table[table["sharpe"] > baseline["sharpe"]].index.tolist()
    better = [name for name in better if name != "buy and hold"]
    print(f"\nbetter Sharpe than buy-and-hold: {better or 'none'}")
    return 0


def cmd_all(args) -> int:
    for command in (cmd_quality, cmd_features, cmd_macro, cmd_study, cmd_control,
                    cmd_validate, cmd_walkforward, cmd_backtest, cmd_forecast):
        print("\n" + "=" * 78)
        code = command(args)
        if code != 0:
            return code
    # speccurve is deliberately left out. Its 200 permutations are 32,000 event
    # studies - about ten minutes, four times everything above put together -
    # and running it on every `all` would train people to skip `all`. Running
    # it with fewer permutations instead would be worse: two commands would
    # then report different p-values for the same question.
    print("\n" + "=" * 78)
    print(
        "Not run: `speccurve`, the specification curve. It re-runs the halving "
        "study under\n160 combinations of the analytical choices the scan above "
        "holds fixed, and tests\nthe whole curve against random placement of "
        "the same dates. About ten minutes."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btc-cycle-lab", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="download data into the local database")
    ingest.add_argument(
        "--what", choices=["all", "prices", "macro", "events", "control"], default="all"
    )
    ingest.add_argument("--start", default=None, help="start date (YYYY-MM-DD)")
    ingest.add_argument("--end", default=None)
    ingest.set_defaults(func=cmd_ingest)

    quality = subparsers.add_parser("quality", help="data quality report")
    quality.set_defaults(func=cmd_quality)

    features = subparsers.add_parser("features", help="build and save the feature frame")
    features.set_defaults(func=cmd_features)

    study = subparsers.add_parser("study", help="event study around halvings and events")
    study.add_argument("--pre", type=int, default=30)
    study.add_argument("--post", type=int, default=365)
    study.set_defaults(func=cmd_study)

    validate = subparsers.add_parser("validate", help="hypothesis scan + correction + out-of-sample")
    validate.set_defaults(func=cmd_validate)

    walkforward = subparsers.add_parser(
        "walkforward", help="walk-forward validation across many test windows"
    )
    walkforward.set_defaults(func=cmd_walkforward)

    control = subparsers.add_parser(
        "control", help="control group: does the NASDAQ react to halvings the same way"
    )
    control.add_argument("--post", type=int, default=365)
    control.add_argument(
        "--category", default=None,
        help="use events from this category instead of halvings (e.g. credit_event)",
    )
    control.set_defaults(func=cmd_control)

    forecast = subparsers.add_parser(
        "forecast", help="directional forecast vs baselines - does it beat them?"
    )
    forecast.set_defaults(func=cmd_forecast)

    macro = subparsers.add_parser("macro", help="liquidity axis: M2 vs proxy, FRED key status")
    macro.add_argument(
        "--check-key", action="store_true",
        help="check whether FRED_API_KEY works (queries the API)",
    )
    macro.set_defaults(func=cmd_macro)

    speccurve = subparsers.add_parser(
        "speccurve",
        help="specification curve: vary the analytical choices, not the hypothesis",
    )
    speccurve.add_argument(
        "--permutations",
        type=int,
        default=200,
        help="null draws for the curve-level test (200 takes about 10 minutes)",
    )
    speccurve.set_defaults(func=cmd_speccurve)

    backtest = subparsers.add_parser("backtest", help="strategies vs buy-and-hold")
    backtest.set_defaults(func=cmd_backtest)

    everything = subparsers.add_parser("all", help="full run (without downloading data)")
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
