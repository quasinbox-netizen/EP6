"""The layer that assembles the modules into complete runs.

The point: the CLI and the dashboard stay thin. All research logic lives here,
so that the terminal result and the browser result cannot drift apart.
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
from forecast.walk import latest_prediction, run_walk_forward
from features.halving import CONFIRMED_HALVINGS
from ingest.prices import load_stitched
from storage import connect, read_events, read_macro, read_prices
from validation.multiple_testing import correct, summarize
from validation.splits import (
    assert_no_overlap,
    cycle_split,
    replicate_finding,
    sign_agreement_test,
    split_frame,
    walk_forward_splits,
    window_effect,
)

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
    """Load everything from the database and build the feature frame with targets."""
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
    """A separate event study for each event category in the registry."""
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
    """Attach 0/1 columns for every macro phase.

    A phase is a hypothesis exactly as much as a halving window is, so it has
    to go through the same multiple-testing correction. Keeping it out of the
    scan would be a quiet privilege: four extra tests nobody counts.
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
    """Every window and phase we treat as a separate hypothesis."""
    return [
        c for c in frame.columns
        if c.startswith("halving_after_")
        or c.startswith("event_")
        or c.startswith("phase_")
    ]


def scan_hypotheses(
    data: LabData, *, target: str = "log_return", config=None
) -> pd.DataFrame:
    """Scan every window and phase, then correct for multiple testing."""
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
    """Repeat each hypothesis separately on the training and the test set.

    An effect that exists only in the training sample is a fit to noise - and
    here that is visible directly, without any aggregation.
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


def walk_forward_check(
    data: LabData, *, target: str = "log_return", config=None
) -> pd.DataFrame:
    """Walk-forward validation: a dozen or more disjoint test windows.

    Splitting by cycle gives ONE test set, so "it did not replicate" is a
    single observation there. Here every hypothesis gets many independent
    windows and we can count how often the sign of the effect survives the
    move from training to test.

    The statistic: under the null the sign in the test window is a coin flip,
    so the number of agreeing folds is binomial with p=0.5. Plus a t-test on
    the mean out-of-sample effect - the test windows are disjoint (embargo), so
    treating them as independent observations is allowed.

    Finally a multiple-testing correction, because there are many hypotheses.
    """
    config = config or load_config()
    frame = with_phase_dummies(data.features)
    if frame.empty:
        return pd.DataFrame()

    settings = config["validation"].get("walk_forward", {})
    folds = walk_forward_splits(
        frame.index,
        train_days=int(settings.get("train_days", 730)),
        test_days=int(settings.get("test_days", 365)),
        step_days=int(settings["step_days"]) if settings.get("step_days") else None,
        embargo_days=int(settings.get("embargo_days", 90)),
        expanding=bool(settings.get("expanding", True)),
    )
    if not folds:
        return pd.DataFrame()

    horizon = int(settings.get("embargo_days", 90))
    for fold in folds:
        assert_no_overlap(fold, horizon_days=horizon - 1)

    # The t-test across folds assumes the test windows are independent. With a
    # step shorter than the window they overlap and that assumption fails -
    # the p-value would be understated. We check this on the ACTUAL indices,
    # not on the parameters, and when they overlap we do NOT report the t-test.
    disjoint = True
    for earlier, later in zip(folds, folds[1:]):
        if len(earlier.test.intersection(later.test)) > 0:
            disjoint = False
            break

    rows = []
    for column in hypothesis_columns(frame):
        train_effects, test_effects = [], []
        for fold in folds:
            train, test = split_frame(frame, fold)
            if train.empty or test.empty:
                continue
            train_effects.append(window_effect(train[target], train[column] > 0))
            test_effects.append(window_effect(test[target], test[column] > 0))

        summary = sign_agreement_test(train_effects, test_effects)
        if summary["n_folds"] == 0:
            continue
        summary["hypothesis"] = column
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    if not disjoint:
        table["test_effect_t_stat"] = float("nan")
        table["test_effect_p_value"] = float("nan")
    table = table.loc[
        :, ["hypothesis", "n_folds", "n_same_sign", "sign_agreement", "sign_p_value",
            "mean_test_effect", "test_effect_t_stat", "test_effect_p_value"]
    ]

    method = config["validation"]["fdr_method"]
    alpha = float(config["validation"]["alpha"])

    # Both statistics are hypotheses and both must be corrected. Correcting
    # only one and reporting the other raw would mean picking the more
    # convenient one after seeing the results.
    corrected = correct(table, method=method, alpha=alpha, p_column="sign_p_value")
    corrected = corrected.rename(
        columns={"p_adjusted": "sign_p_adjusted",
                 "significant_raw": "sign_significant_raw",
                 "significant_adjusted": "sign_significant_adjusted"}
    )
    if disjoint:
        effect = correct(corrected, method=method, alpha=alpha,
                         p_column="test_effect_p_value")
        corrected["test_effect_p_adjusted"] = effect["p_adjusted"]
        corrected["test_effect_significant_adjusted"] = effect["significant_adjusted"]
    else:
        corrected["test_effect_p_adjusted"] = float("nan")
        corrected["test_effect_significant_adjusted"] = False

    corrected.attrs["n_folds_total"] = len(folds)
    corrected.attrs["test_windows_disjoint"] = disjoint
    corrected.attrs["folds"] = pd.DataFrame([f.describe() for f in folds])
    return corrected


def forecast_report(data: LabData, config=None) -> dict:
    """Directional forecast, evaluated walk-forward against the baselines.

    Returns the walk-forward run plus the most recent prediction. The two must
    be read together: if the run says there is no edge, the latest probability
    is decoration, and the CLI says so out loud.
    """
    config = config or load_config()
    frame = with_phase_dummies(data.features)
    if frame.empty:
        return {"error": "the database is empty - run `ingest` first"}

    settings = config.get("forecast", {})
    horizon = int(settings.get("horizon_days", 30))
    if f"fwd_return_{horizon}d" not in frame.columns:
        return {
            "error": f"no fwd_return_{horizon}d column - the horizon in config.yaml "
                     "must be one of the horizons load_lab_data builds"
        }

    walk = config["validation"].get("walk_forward", {})
    splits = walk_forward_splits(
        frame.index,
        train_days=int(walk.get("train_days", 730)),
        test_days=int(walk.get("test_days", 365)),
        step_days=int(walk["step_days"]) if walk.get("step_days") else None,
        embargo_days=int(walk.get("embargo_days", 90)),
        expanding=bool(walk.get("expanding", True)),
    )
    if not splits:
        return {"error": "not enough data for walk-forward validation"}

    run = run_walk_forward(frame, splits, horizon=horizon)
    if run.n_folds == 0:
        return {"error": "no fold could be fitted"}

    return {
        "horizon": horizon,
        "run": run,
        "latest": latest_prediction(frame, horizon=horizon),
    }


def macro_phase_comparison(data: LabData, config=None) -> dict:
    """Compare the macro phase computed from real M2 against the dollar proxy.

    Why this matters: without a FRED key the liquidity axis is replaced by the
    inverted dollar index. That is a reasonable proxy for financial
    conditions, but it is NOT the money supply. This function says how far the
    two versions differ - and whether conclusions depend on the choice.
    """
    config = config or load_config()
    frame = data.features
    if frame.empty or data.macro.empty:
        return {"error": "no macro data"}

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
        return {"error": "no liquidity series (neither m2 nor dxy)"}

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
    """Compare BTC's reaction with the control assets' on the same dates.

    Returns a dict: control name -> ControlComparison, plus a placebo event
    study for each control. A missing control group is not an error - we
    return an empty result explaining what to fetch.
    """
    config = config or load_config()
    if data.is_empty:
        return {"error": "the database is empty"}
    if not data.has_controls:
        return {
            "error": "no control group in the database - run "
                     "`run.py ingest --what control`"
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
    """Backtest a handful of simple strategies against buy-and-hold."""
    config = config or load_config()
    frame = data.features
    if frame.empty:
        return pd.DataFrame(), []

    close = frame["close"]
    settings = BacktestConfig.from_config(config)
    results = [
        run_backtest(close, buy_and_hold(close.index), settings, name="buy and hold"),
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
                name="macro: liquidity up, rates down",
            )
        )
    return compare(results), results


def full_report(config=None) -> dict:
    """One pass: data -> event study -> validation -> backtest."""
    config = config or load_config()
    data = load_lab_data(config)
    if data.is_empty:
        return {"error": "the database is empty - run `ingest` first"}

    scan = scan_hypotheses(data, config=config)
    table, results = run_strategies(data, config=config)
    return {
        "data": data,
        "walk_forward": walk_forward_check(data, config=config),
        "halving_study": halving_event_study(data, config=config),
        "category_studies": category_event_studies(data, config=config),
        "scan": scan,
        "scan_summary": summarize(scan) if not scan.empty else "no hypotheses",
        "out_of_sample": out_of_sample_check(data, config=config),
        "backtest_table": table,
        "backtest_results": results,
    }
