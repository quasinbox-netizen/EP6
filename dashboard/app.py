"""Streamlit dashboard - presentation layer only.

The rule: no research logic in this file. Every number comes from a module in
src/ and passes through the same functions as the CLI. If you want to change
how something is computed, do it in src/ - otherwise the chart and the
terminal will start showing different things and you will believe the
prettier one.

Run it with:
    python run.py dashboard
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from features.halving import CONFIRMED_HALVINGS  # noqa: E402
from pipeline import (  # noqa: E402
    category_event_studies,
    control_comparison,
    forecast_report,
    halving_event_study,
    load_lab_data,
    out_of_sample_check,
    run_strategies,
    scan_hypotheses,
)
from validation.multiple_testing import summarize  # noqa: E402

st.set_page_config(page_title="BTC Cycle Lab", layout="wide")

COLORS = {
    "price": "#e8a33d",
    "car": "#4c8bf5",
    "band": "rgba(76, 139, 245, 0.18)",
    "halving": "rgba(232, 163, 61, 0.14)",
    "zero": "#8a8a8a",
}


@st.cache_data(show_spinner="Loading data from the local database...")
def cached_data():
    return load_lab_data()


@st.cache_data(show_spinner="Computing the event study...")
def cached_halving_study(post: int):
    return halving_event_study(cached_data(), post=post)


@st.cache_data(show_spinner="Computing event studies per category...")
def cached_category_studies(post: int):
    return category_event_studies(cached_data(), post=post)


@st.cache_data(show_spinner="Comparing against the control group...")
def cached_control(post: int):
    return control_comparison(cached_data(), post=post)


@st.cache_data(show_spinner="Fitting and scoring the forecast...")
def cached_forecast():
    return forecast_report(cached_data())


@st.cache_data(show_spinner="Scanning hypotheses...")
def cached_scan():
    return scan_hypotheses(cached_data())


@st.cache_data(show_spinner="Checking out-of-sample replication...")
def cached_out_of_sample():
    return out_of_sample_check(cached_data())


@st.cache_data(show_spinner="Running backtests...")
def cached_backtests():
    table, results = run_strategies(cached_data())
    curves = pd.DataFrame({r.name: r.equity for r in results})
    return table, curves


def price_chart(data, halving_window_days: int, show_events: bool) -> go.Figure:
    frame = data.features
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["close"],
            name="BTC/USD",
            line=dict(color=COLORS["price"], width=1.2),
        )
    )
    for halving in CONFIRMED_HALVINGS:
        end = halving + pd.Timedelta(days=halving_window_days)
        figure.add_vrect(x0=halving, x1=end, fillcolor=COLORS["halving"], line_width=0)
        figure.add_vline(x=halving, line=dict(color=COLORS["price"], width=1, dash="dot"))

    if show_events and not data.events.empty:
        non_halving = data.events[data.events["category"] != "halving"]
        prices_by_day = frame["close"]
        for _, event in non_halving.iterrows():
            day = pd.Timestamp(event["available_from"]).normalize()
            if day not in prices_by_day.index:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[day],
                    y=[prices_by_day.loc[day]],
                    mode="markers",
                    marker=dict(size=8, symbol="diamond"),
                    name=event["name"],
                    hovertext=f"{event['category']}: {event['description']}",
                    showlegend=False,
                )
            )

    figure.update_layout(
        yaxis_type="log",
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="price (log scale)",
        hovermode="x unified",
    )
    return figure


def car_chart(result, title: str) -> go.Figure:
    table = result.table.dropna(subset=["car"])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=table.index, y=table["car_ci_high"], line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=table.index, y=table["car_ci_low"], fill="tonexty",
            fillcolor=COLORS["band"], line=dict(width=0),
            name="95% confidence interval",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=table.index, y=table["car"], name="mean CAR",
            line=dict(color=COLORS["car"], width=2),
        )
    )
    figure.add_hline(y=0, line=dict(color=COLORS["zero"], dash="dash", width=1))
    figure.update_layout(
        title=title,
        height=420,
        xaxis_title="days from the event",
        yaxis_title="cumulative abnormal return",
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis_tickformat=".0%",
    )
    return figure


def equity_chart(curves: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for column in curves.columns:
        figure.add_trace(
            go.Scatter(
                x=curves.index, y=curves[column], name=column,
                line=dict(width=2.5 if column == "buy and hold" else 1.4),
            )
        )
    figure.update_layout(
        yaxis_type="log", height=440, yaxis_title="equity (log scale)",
        margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
    )
    return figure


def main() -> None:
    config = load_config()
    st.title("BTC Cycle Lab")
    st.caption(
        "A tool for testing whether the halving cycle and macro events explain "
        "anything in the price of BTC. Every result comes with a confidence "
        "interval and a count of observations - without those it is not a result."
    )

    data = cached_data()
    if data.is_empty:
        st.error(
            "The database is empty. Download the data first:\n\n"
            "`python run.py ingest --what all`"
        )
        st.stop()

    with st.sidebar:
        st.header("Settings")
        halving_window_days = st.slider("Halving window on the chart (days)", 30, 730, 365, 5)
        study_post = st.slider("Event study horizon (days after the event)", 30, 730, 365, 5)
        show_events = st.checkbox("Show events on the chart", value=True)
        st.divider()

        # Everything on this page is cached, so a fresh `ingest` is invisible
        # until the cache is dropped. The toolbar's "Clear cache" is hidden by
        # config.toml along with the Deploy button, so the button lives here
        # instead - where someone looking for it would actually look.
        if st.button("Reload data", icon=":material/refresh:", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.caption("Press this after running `ingest` to pick up new data.")

        st.divider()
        st.caption(
            f"Data: {data.features.index.min().date()} - {data.features.index.max().date()} "
            f"({len(data.features)} days)\n\n"
            f"Stitching: {', '.join(config['price']['stitch_priority'])}\n\n"
            f"Costs: {config['backtest']['fee_bps']} bps fees + "
            f"{config['backtest']['slippage_bps']} bps slippage"
        )

    (
        tab_price, tab_study, tab_control, tab_validation, tab_backtest, tab_forecast
    ) = st.tabs(
        ["Price and cycles", "Event study", "Control group", "Validation",
         "Backtest", "Forecast"]
    )

    with tab_price:
        st.plotly_chart(
            price_chart(data, halving_window_days, show_events), width="stretch"
        )
        columns = st.columns(4)
        columns[0].metric("Days in sample", f"{len(data.features):,}")
        columns[1].metric("Halvings in sample", len(CONFIRMED_HALVINGS))
        columns[2].metric("Events in registry", len(data.events))
        columns[3].metric("Macro series", data.macro["series"].nunique() if not data.macro.empty else 0)
        with st.expander("Event registry"):
            st.dataframe(
                data.events.loc[:, ["date", "category", "name", "description"]],
                width="stretch", hide_index=True,
            )

    with tab_study:
        study = cached_halving_study(study_post)
        st.plotly_chart(
            car_chart(study, f"Halvings (n={study.n_events})"), width="stretch"
        )
        st.info(study.summary())
        if study.n_events < 10:
            st.warning(
                f"The confidence interval is wide because the sample holds "
                f"{study.n_events} events. That is not a flaw in the method - it is "
                "all the information there is."
            )

        st.subheader("Event categories")
        studies = cached_category_studies(min(study_post, 180))
        if studies:
            rows = [
                {"category": name, "n": result.n_events, **result.car_summary}
                for name, result in studies.items()
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            choice = st.selectbox("Chart for category", sorted(studies))
            st.plotly_chart(
                car_chart(studies[choice], f"{choice} (n={studies[choice].n_events})"),
                width="stretch",
            )
        st.caption(
            "The p-values in this table are RAW. Use the Validation tab for inference."
        )

    with tab_control:
        st.caption(
            "A halving concerns Bitcoin alone, so what assets without a halving "
            "did over the same window is a placebo test. The difference is computed "
            "pairwise across events so that shared macro conditions cancel out."
        )
        report = cached_control(study_post)
        if "error" in report:
            st.warning(report["error"])
        else:
            for name, comparison in report["comparisons"].items():
                st.subheader(f"BTC vs {name}")
                if comparison.table.empty:
                    st.info("no shared events with a complete window")
                    continue
                st.dataframe(comparison.table, width="stretch")
                verdict = comparison.verdict()
                if "NOT common" in verdict:
                    st.success(verdict)
                else:
                    st.warning(verdict)
                placebo = report["placebos"][name]
                if placebo.car_summary:
                    st.caption(
                        f"Placebo - {name} treated like BTC: "
                        f"CAR({placebo.car_summary['offset']}d) = "
                        f"{placebo.car_summary['car']:+.1%}, "
                        f"p = {placebo.car_summary['p_value']:.3f}"
                    )
                with st.expander(f"CAR of individual events ({name})"):
                    st.dataframe(comparison.per_event, width="stretch")

    with tab_validation:
        scan = cached_scan()
        if scan.empty:
            st.info("No hypotheses to check.")
        else:
            st.subheader("Window scan with multiple-testing correction")
            st.info(summarize(scan))
            st.dataframe(
                scan.loc[
                    :, ["hypothesis", "n_in", "mean_in", "mean_out", "difference",
                        "p_value", "p_adjusted", "significant_adjusted"]
                ],
                width="stretch", hide_index=True,
            )
        out_of_sample = cached_out_of_sample()
        if not out_of_sample.empty:
            st.subheader("Out-of-sample replication (split by cycle)")
            st.dataframe(
                out_of_sample.loc[
                    :, ["hypothesis", "train_effect", "test_effect", "same_sign",
                        "significant_in_train", "significant_out_of_sample",
                        "effect_retained", "replicated"]
                ],
                width="stretch", hide_index=True,
            )
            survivors = out_of_sample[out_of_sample["replicated"]]["hypothesis"].tolist()
            if survivors:
                st.success(f"Survived out of sample: {', '.join(survivors)}")
            else:
                st.warning(
                    "No hypothesis replicated outside the training sample. That is "
                    "the most common and most instructive result in this project."
                )

    with tab_backtest:
        table, curves = cached_backtests()
        if table.empty:
            st.info("No backtest results.")
        else:
            st.plotly_chart(equity_chart(curves), width="stretch")
            st.dataframe(
                table.loc[
                    :, ["total_return", "cagr", "sharpe", "sortino", "max_drawdown",
                        "calmar", "win_rate", "time_in_market", "turnover_annual", "total_cost"]
                ],
                width="stretch",
            )
            baseline = table.loc["buy and hold"]
            better = [
                name for name in table.index
                if name != "buy and hold" and table.loc[name, "sharpe"] > baseline["sharpe"]
            ]
            st.caption(
                "Always compared against buy-and-hold. Better Sharpe: "
                + (", ".join(better) if better else "no strategy")
            )

    with tab_forecast:
        st.caption(
            "Probability that the forward return is positive - not a price "
            "forecast. The number that matters is whether it beats the "
            "baselines, especially `always_up`: Bitcoin rose in most historical "
            "windows, so the reference point is that base rate, not 50%."
        )
        # Streamlit renders every tab body on every run, so an unguarded call
        # here would make the whole page wait for 13 model fits before showing
        # anything - even for someone who never opens this tab. Hence the
        # explicit gate; after the first run the result is cached.
        if not st.session_state.get("forecast_requested"):
            st.info(
                "Fitting the model across 13 walk-forward folds takes about a "
                "minute. It is not run until you ask for it, so the rest of the "
                "dashboard stays fast."
            )
            if st.button("Run the forecast", icon=":material/play_arrow:"):
                st.session_state["forecast_requested"] = True
                st.rerun()
            report = None
        else:
            report = cached_forecast()

        if report is None:
            pass
        elif "error" in report:
            st.warning(report["error"])
        else:
            run = report["run"]
            verdict_text = run.summary()
            if "NO EDGE" in verdict_text:
                st.error(verdict_text)
            else:
                st.success(verdict_text)

            st.subheader("Pooled, non-overlapping rows")
            st.dataframe(
                run.pooled.loc[
                    :, ["n", "brier", "log_loss", "accuracy", "auc",
                        "mean_probability", "base_rate"]
                ],
                width="stretch",
            )
            st.caption(
                f"Scored on {int(run.pooled.loc['model', 'n'])} non-overlapping "
                f"rows out of {int(run.pooled_all_rows.loc['model', 'n'])} daily "
                f"predictions. A {report['horizon']}-day label on consecutive "
                "days repeats almost all of itself, so the smaller number is "
                "the honest one."
            )

            latest = report["latest"]
            if "error" not in latest:
                left, right = st.columns(2)
                left.metric(
                    f"Next {report['horizon']} days up",
                    f"{latest['probability_up']:.1%}",
                    delta=f"{latest['edge_over_base_rate']:+.1%} vs base rate",
                )
                right.metric("Training base rate", f"{latest['train_base_rate']:.1%}")
                if "NO EDGE" in verdict_text:
                    st.warning(
                        "Read that probability as decoration. The evaluation "
                        "above says the model has no edge out of sample, so it "
                        "is not evidence about the future."
                    )

            with st.expander("Per fold"):
                st.dataframe(
                    run.folds.loc[
                        :, ["fold", "n_train", "alpha", "n", "brier", "accuracy",
                            "auc", "base_rate"]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            with st.expander("Calibration: predicted vs actual"):
                if run.calibration.empty:
                    st.info("Not enough data.")
                else:
                    st.dataframe(run.calibration, width="stretch")
                    st.caption(
                        "A well calibrated model has `mean_predicted` close to "
                        "`share_positive` in every row."
                    )


if __name__ == "__main__":
    main()
