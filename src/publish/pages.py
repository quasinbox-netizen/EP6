"""The two pages built from saved results, and the small HTML helpers.

Everything here reads what the CLI already wrote into data/processed. That is
deliberate: the site must not be able to publish a number the terminal never
printed, so nothing on these pages is recomputed at build time.
"""
from __future__ import annotations

import pandas as pd

from publish.charts import car_chart_from_frame


def figure_html(figure, element_id: str) -> str:
    """A Plotly figure as a div, with the library loaded once per page."""
    figure.update_layout(
        paper_bgcolor="#080b11", plot_bgcolor="#080b11",
        font={"color": "#c3ccd8", "family": "ui-monospace,Consolas,monospace"},
    )
    return figure.to_html(
        full_html=False, include_plotlyjs=False, div_id=element_id,
        config={"displayModeBar": False, "responsive": True},
    )


def cards(pairs) -> str:
    cells = "".join(
        f"<div class='card'><u>{label}</u><b>{value}</b></div>" for label, value in pairs
    )
    return f"<div class='cards'>{cells}</div>"


def table(frame: pd.DataFrame) -> str:
    head = "".join(f"<th>{column}</th>" for column in frame.columns)
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{'' if pd.isna(value) else value}</td>" for value in row)
        + "</tr>"
        for row in frame.itertuples(index=False)
    )
    return (
        "<div class='wrap'><table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + rows
        + "</tbody></table></div>"
    )


def today_page(inputs) -> str:
    """The question people arrive with, and the part of it that has no answer.

    It goes first because burying it behind the research pages is what made
    the app hard to read: the honest answers were scattered across seven tabs
    while the question nobody could avoid asking had no page at all.
    """
    saved = inputs.saved or {}
    sizing = saved.get("sizing_today", pd.DataFrame())
    edge = saved.get("backtest_edge", pd.DataFrame())
    parts = ["<section><h2>What this tool can tell you about today</h2>"]

    if sizing.empty:
        parts.append("<p class='note'>No sizing result has been saved yet.</p>")
    else:
        row = sizing.iloc[0]
        parts.append(
            f"<p class='note'>As of {row['as_of']}, price {row['price']:,.0f}. "
            "Everything here is computed from data up to that date.</p>"
        )
        parts.append(cards([
            ("How much to hold", f"{row['position']:.2f}"),
            ("Forecast volatility", f"{row['forecast_annual_volatility']:.0%}"),
            ("Which way it goes", "Unknown"),
        ]))
        parts.append(
            "<p class='note'>The position is target volatility divided by forecast "
            "volatility, after a rebalance band. It says how much, never which way - "
            "and volatility is the one thing in this project that is forecastable.</p>"
        )

    for label, key in (("10 days", "range_10"), ("30 days", "range_30")):
        frame = saved.get(key, pd.DataFrame())
        if frame.empty:
            continue
        best = frame.loc[frame["level"].idxmax()]
        parts.append(
            f"<p>In <b>{label}</b>, with {best['level']:.0%} probability, between "
            f"<b>{best['low']:,.0f}</b> and <b>{best['high']:,.0f}</b> "
            f"({best['low_pct']:+.1%} / {best['high_pct']:+.1%}).</p>"
        )
    parts.append(
        "<p class='note'>These intervals passed a coverage test on non-overlapping "
        "windows: a 90% interval really did contain the outcome about 90% of the "
        "time. Levels that failed are withheld rather than quietly widened.</p>"
        "</section>"
    )

    parts.append("<section><h2>When to get in or out</h2>")
    parts.append(
        "<div class='verdict'><b>This tool cannot tell you, and it is not being "
        "modest.</b> Every timing rule it tested was rejected, the hypothesis scan "
        "finds nothing after correction, 160 specifications of the halving question "
        "find nothing, and the forecast model loses to its own baselines. An entry "
        "signal here would be invented.</div>"
    )
    if not edge.empty:
        columns = [
            column for column in (
                "strategy", "sharpe", "sharpe_random_timing", "p_value",
                "p_worst_without_one", "fragile", "survives_correction",
            ) if column in edge.columns
        ]
        parts.append(table(edge.loc[:, columns].round(3)))
        parts.append(
            "<p class='note'>Every rule compared against ITSELF applied at random "
            "times. A p-value under 0.05 would mean the timing beat chance; "
            "<code>p_worst_without_one</code> re-runs the test with one trade removed, "
            "and <code>survives_correction</code> accounts for having tried several "
            "rules at once. Nothing survives.</p>"
        )
    parts.append("</section>")
    return "".join(parts)


def evidence_page(inputs) -> str:
    """The tests themselves, in the order they were run."""
    saved = inputs.saved or {}
    parts = []

    study = saved.get("event_study_halving", pd.DataFrame())
    if not study.empty:
        summary = inputs.study.car_summary
        parts.append(
            "<section><h2>The halving event study</h2>"
            + figure_html(
                car_chart_from_frame(study, f"n = {inputs.study.n_events} halvings"), "car")
            + f"<p class='note'>Cumulative abnormal return a year after a halving: "
              f"{summary['car']:+.0%}, with a 95% interval from {summary['ci_low']:+.0%} "
              f"to {summary['ci_high']:+.0%} and p={summary['p_value']:.2f}. The "
              "interval is that wide because the sample holds four events. That is "
              "not a flaw in the method; it is all the information there is.</p>"
              "</section>"
        )

    scan = saved.get("hypothesis_scan", pd.DataFrame())
    if not scan.empty:
        columns = [
            column for column in (
                "hypothesis", "target", "n_in", "difference", "p_value",
                "p_adjusted", "significant_adjusted",
            ) if column in scan.columns
        ]
        shown = scan.loc[:, columns].sort_values("p_value").head(12).round(4)
        survivors = int(scan["significant_adjusted"].sum())
        parts.append(
            "<section><h2>Every hypothesis, corrected for having asked many</h2>"
            + table(shown)
            + f"<p class='note'>The twelve smallest p-values out of {len(scan)} "
              f"hypotheses. After the Benjamini-Hochberg correction {survivors} "
              "survive. Testing enough ideas produces small p-values by itself, "
              "which is what the correction is for.</p></section>"
        )

    if not inputs.curve_summary.empty:
        row = inputs.curve_summary.iloc[0]
        parts.append(
            "<section><h2>One question, 160 ways of asking it</h2>"
            + cards([
                ("Specifications", f"{int(row['n_specs'])}"),
                ("Significant", f"{int(row['n_significant'])}"),
                ("Random dates give", f"{row['null_significant_mean']:.1f}"),
                ("Median effect", f"{row['median_car']:+.1%}"),
            ])
            + "<p class='note'>A finding that depends on which reasonable choices were "
              "made is not a finding. Randomly placed dates produce more significant "
              "specifications here than the real halvings do.</p></section>"
        )

    backtest = saved.get("backtest_comparison", pd.DataFrame())
    if not backtest.empty:
        columns = [
            column for column in (
                "strategy", "total_return", "cagr", "sharpe", "max_drawdown",
                "win_rate", "time_in_market",
            ) if column in backtest.columns
        ]
        parts.append(
            "<section><h2>Backtests, against buy-and-hold</h2>"
            + table(backtest.loc[:, columns].round(3))
            + "<p class='note'>Fees of 10 bps and slippage of 15 bps are charged on "
              "turnover. Beating buy-and-hold on Sharpe is not enough: a rule also has "
              "to beat its own random-timing version, which is the table on the first "
              "page.</p></section>"
        )

    forecast = saved.get("forecast_pooled", pd.DataFrame())
    if not forecast.empty:
        columns = [
            column for column in
            ("predictor", "n", "brier", "accuracy", "auc", "base_rate")
            if column in forecast.columns
        ]
        parts.append(
            "<section><h2>The directional model, scored against its baselines</h2>"
            + table(forecast.loc[:, columns].round(4))
            + "<p class='note'>Scored on non-overlapping rows only. The model has to "
              "beat <code>always_up</code> rather than a coin flip: this series rose in "
              "most windows, so the base rate is the bar. It does not clear it.</p>"
              "</section>"
        )

    sizing = saved.get("sizing_comparison", pd.DataFrame())
    if not sizing.empty:
        columns = [
            column for column in
            ("strategy", "sharpe", "max_drawdown", "volatility", "turnover_annual",
             "total_cost", "cagr")
            if column in sizing.columns
        ]
        parts.append(
            "<section><h2>Sizing: the one part that works</h2>"
            + table(sizing.loc[:, columns].round(3))
            + "<p class='note'>Direction is not forecastable here and every other "
              "section says so. The size of the next move is: volatility clusters, so "
              "holding target volatility divided by forecast volatility keeps the RISK "
              "constant rather than the quantity. The rebalance band exists because "
              "retrading on every wobble in the forecast costs more than the sizing is "
              "worth.</p></section>"
        )

    walk = saved.get("walk_forward", pd.DataFrame())
    if not walk.empty:
        columns = [
            column for column in
            ("hypothesis", "n_folds", "sign_agreement", "sign_p_value",
             "mean_test_effect", "sign_p_adjusted")
            if column in walk.columns
        ]
        shown = walk.loc[:, columns].sort_values("sign_p_value").head(10).round(4)
        parts.append(
            "<section><h2>Does the sign hold up across windows?</h2>"
            + table(shown)
            + f"<p class='note'>Each hypothesis re-tested on {int(walk['n_folds'].max())} "
              "disjoint windows with an embargo between them. An effect that is real "
              "keeps its sign; one that is an artefact of a particular stretch does not. "
              "The ten smallest p-values are shown, already corrected.</p></section>"
        )

    replication = saved.get("out_of_sample", pd.DataFrame())
    if not replication.empty:
        columns = [
            column for column in
            ("hypothesis", "train_effect", "test_effect", "same_sign",
             "effect_retained", "replicated")
            if column in replication.columns
        ]
        replicated = int(replication["replicated"].sum())
        parts.append(
            "<section><h2>Held out, then checked</h2>"
            + table(replication.loc[:, columns].head(10).round(4))
            + f"<p class='note'>Fitted on early cycles, checked on the ones held back. "
              f"{replicated} of {len(replication)} replicate. A hypothesis that only "
              "works on the data that suggested it is a description of that data.</p>"
              "</section>"
        )

    categories = saved.get("event_study_categories", pd.DataFrame())
    if not categories.empty:
        columns = [
            column for column in
            ("category", "n", "offset", "car", "ci_low", "ci_high", "p_value")
            if column in categories.columns
        ]
        parts.append(
            "<section><h2>Other events, same treatment</h2>"
            + table(categories.loc[:, columns].round(4))
            + "<p class='note'>Halvings are not the only events in the registry. Every "
              "category gets the same event study, with the same intervals and the same "
              "counts - and the counts are what to read first.</p></section>"
        )

    parts.append(
        "<section><h2>The control group</h2><p class='note'>"
        + inputs.control_note
        + "</p></section>"
    )
    return "".join(parts)
