"""The two pages built from saved results, and the small HTML helpers.

Everything here reads what the CLI already wrote into data/processed. That is
deliberate: the site must not be able to publish a number the terminal never
printed, so nothing on these pages is recomputed at build time.
"""
from __future__ import annotations

from html import escape

import pandas as pd

from publish.charts import car_chart_from_frame, edge_chart, tape_chart

# Where the front page sends a reader who wants the workings rather than the
# answer. The front page is now one page and a short one; these are the rest.
WORKINGS = (
    ("now.html", "The cycle",
     "Four halvings on one dial, and what usually happened from days like today."),
    ("trader.html", "The agent",
     "Every trade it made, and where the two constants that size it came from."),
    ("evidence.html", "The evidence",
     "All ten tests, in the order they were run, with their counts."),
    ("signals.html", "The desk",
     "What each rule says right now, against a live quote."),
    ("receipts.html", "Receipts",
     "Claims written down before their window closed, scored when it closes."),
)

# What each column is called on the page, and the sentence that explains it
# where the name alone cannot. The tables used to print the DataFrame's own
# column name straight into the <th>, so a reader met `p_worst_without_one`
# and `survives_correction` with nothing to decode them by. A header that has
# to be decoded is a column nobody reads, and the numbers under it are wasted
# - which is most of what made this site hard to read rather than long.
COLUMNS: dict[str, tuple[str, str]] = {
    "strategy": ("rule", ""),
    "predictor": ("model", ""),
    "hypothesis": ("hypothesis", ""),
    "category": ("event", ""),
    "sharpe": ("Sharpe", "Return per unit of volatility, annualised. Higher is "
                         "better, and on its own it proves nothing."),
    "sharpe_random_timing": (
        "Sharpe, random dates",
        "The same rule entered on randomly chosen dates. This, not zero, is "
        "the bar the real timing has to clear.",
    ),
    "p_value": ("p", "How often chance alone produces a result at least this "
                     "good. Below 0.05 is the usual bar."),
    "p_adjusted": ("p, corrected", "The p-value after accounting for how many "
                                   "hypotheses were tested at once."),
    "p_worst_without_one": (
        "p, best trade removed",
        "The same test re-run with the single best trade dropped. A result "
        "that rests on one trade shows itself here.",
    ),
    "fragile": ("rests on one trade",
                "Whether removing a single trade destroys the result."),
    "survives_correction": ("survives correction",
                            "Whether it still holds once every rule that was "
                            "tried at all is counted."),
    "significant_adjusted": ("significant",
                             "Significant after the Benjamini-Hochberg correction."),
    "target": ("measures", ""),
    "n_in": ("days inside", "How many days fall inside the condition being tested."),
    "difference": ("difference", "The gap between days inside the condition and "
                                 "every other day."),
    "total_return": ("total return", ""),
    "cagr": ("a year", "Compound annual growth rate over the whole test."),
    "max_drawdown": ("worst fall", "The deepest peak-to-trough loss over the test."),
    "win_rate": ("winning days", ""),
    "time_in_market": ("time invested", "The share of the test spent holding anything."),
    "n": ("observations", "How many independent observations the row rests on."),
    "brier": ("Brier", "Mean squared error of the probabilities. Lower is better."),
    "accuracy": ("accuracy", ""),
    "auc": ("AUC", "Ranking quality. 0.5 is a coin flip."),
    "base_rate": ("base rate", "How often the thing being predicted happened anyway."),
    "volatility": ("volatility", ""),
    "turnover_annual": ("turnover a year",
                        "How much of the position is traded over a year."),
    "total_cost": ("cost paid", "Fees and slippage charged on that turnover."),
    "n_folds": ("windows", "Disjoint stretches of history the test was repeated on."),
    "sign_agreement": ("sign held",
                       "The share of windows in which the effect kept its direction."),
    "sign_p_value": ("p, sign", ""),
    "sign_p_adjusted": ("p, sign, corrected", ""),
    "mean_test_effect": ("mean effect", ""),
    "train_effect": ("effect, fitted",
                     "Measured on the early cycles the hypothesis was fitted on."),
    "test_effect": ("effect, held out", "Measured on the cycles held back."),
    "same_sign": ("same direction", ""),
    "effect_retained": ("size kept",
                        "How much of the fitted effect survived on held-out data."),
    "replicated": ("replicated", ""),
    "offset": ("days after", ""),
    "car": ("effect", "Cumulative abnormal return: the move beyond what the rest "
                      "of the window would predict."),
    "ci_low": ("interval, low", ""),
    "ci_high": ("interval, high", ""),
    # the cycle board
    "median_matched": ("median %, days like today",
                       "The middle outcome across days that looked like today."),
    "positive_matched": ("rose %, days like today", ""),
    "independent_windows": ("independent windows",
                            "Non-overlapping forward windows. This, not the number "
                            "of matched days, is what the row rests on."),
    "median_all": ("median %, every day",
                   "The same statistic over every day in the sample, as a baseline."),
    "positive_all": ("rose %, every day", ""),
    "quotable": ("quotable", "Whether there are enough independent windows to "
                             "quote the row at all."),
    "position": ("position", "Whether the rule is holding BTC right now."),
    "since": ("since", ""),
    "what flips it": ("what flips it", ""),
    "as_of": ("recorded", ""),
    "settles": ("settles", ""),
    "horizon": ("horizon", ""),
    "claim": ("claim", ""),
    "level": ("confidence", ""),
    "low": ("low", ""),
    "high": ("high", ""),
    "p_up": ("p, up", ""),
    "matured": ("settled", ""),
    "realised": ("outcome", ""),
}


def _label(column: str) -> str:
    """The reader's name for a column, without the tooltip."""
    return COLUMNS.get(str(column), (str(column).replace("_", " "), ""))[0]


def _header(column: str) -> str:
    label, hint = COLUMNS.get(str(column), (str(column).replace("_", " "), ""))
    if not hint:
        return escape(label)
    return f'<abbr title="{escape(hint, quote=True)}">{escape(label)}</abbr>'


def _kind(frame: pd.DataFrame, column, first) -> str:
    """How a column should be set: as the row's name, a flag, a number, prose."""
    if column == first:
        return "key"
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return "flag"
    filled = values.dropna()
    if len(filled) and all(isinstance(v, str) and v in ("True", "False") for v in filled):
        return "flag"
    if pd.api.types.is_numeric_dtype(values):
        return "num"
    if any(isinstance(v, str) and len(v) > 24 for v in filled):
        return "txt"
    return ""


def _decimals(values: pd.Series) -> int:
    """How many decimals the column actually carries, so a column lines up.

    Right-aligned tabular figures still read badly when one row shows 0.93 and
    the next 0.711: the decimal points sit in different places. Padding to the
    column's own precision is presentation, not a restatement - 0.93 rounded to
    three places is 0.930.
    """
    most = 0
    for value in values.dropna():
        try:
            text = f"{float(value):.10f}".rstrip("0")
        except (TypeError, ValueError):
            continue
        most = max(most, len(text.partition(".")[2]))
    return min(most, 4)


def _cell(value, kind: str, decimals: int) -> str:
    try:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if kind == "flag":
        yes = value is True or str(value) == "True"
        return '<i class="yes">yes</i>' if yes else '<i class="no">no</i>'
    if kind == "num":
        number = float(value)
        return f"{number:,.0f}" if decimals == 0 else f"{number:,.{decimals}f}"
    return escape(str(value))


def figure_html(figure, element_id: str) -> str:
    """A Plotly figure as a div, with the library loaded once per page."""
    figure.update_layout(
        paper_bgcolor="#101219", plot_bgcolor="#101219",
        font={"color": "#c3cad6", "size": 13,
              "family": '-apple-system,BlinkMacSystemFont,"SF Pro Text",'
                        '"Segoe UI Variable Text","Segoe UI",Roboto,'
                        '"Helvetica Neue",Arial,sans-serif'},
    )
    # Plotly's default grid is white on a light plot. On these cards it drew a
    # cage over every chart and read louder than the line inside it; one place
    # to set it means a figure written before this cannot miss the change.
    figure.update_xaxes(gridcolor="rgba(255,255,255,.055)", zerolinecolor="#39414f",
                        linecolor="#39414f")
    figure.update_yaxes(gridcolor="rgba(255,255,255,.055)", zerolinecolor="#39414f",
                        linecolor="#39414f")
    return figure.to_html(
        full_html=False, include_plotlyjs=False, div_id=element_id,
        config={"displayModeBar": False, "responsive": True},
    )


def cards(pairs) -> str:
    cells = "".join(
        f"<div class='card'><u>{label}</u><b>{value}</b></div>" for label, value in pairs
    )
    return f"<div class='cards'>{cells}</div>"


def table(frame: pd.DataFrame, *, highlight: str | None = None) -> str:
    """A frame as a table a person can read rather than a frame printed as HTML.

    Three things happen here that did not before. Columns are named the way a
    reader would say them, with the explanation on the header itself rather
    than in a paragraph underneath. Booleans stop being `True`/`False`.
    And every cell carries the column's name, so the stylesheet can turn each
    row into a card on a phone - where all seven of these tables used to run
    off the side of the screen with nothing to say they had.
    """
    columns = list(frame.columns)
    if not columns:
        return ""
    first = columns[0]
    kinds = {column: _kind(frame, column, first) for column in columns}
    decimals = {
        column: _decimals(frame[column])
        for column in columns if kinds[column] == "num"
    }

    def classes(column) -> str:
        marked = " mark" if highlight is not None and column == highlight else ""
        return (kinds[column] + marked).strip()

    head = "".join(
        f'<th class="{classes(column)}">{_header(column)}</th>' for column in columns
    )
    rows = []
    for row in frame.itertuples(index=False):
        cells = "".join(
            f'<td class="{classes(column)}" data-label="{escape(_label(column), quote=True)}">'
            f"{_cell(value, kinds[column], decimals.get(column, 0))}</td>"
            for column, value in zip(columns, row)
        )
        rows.append("<tr>" + cells + "</tr>")
    return (
        "<div class='wrap'><table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def lede(text: str) -> str:
    """The one line of a section that is its answer, set larger than the rest."""
    return f"<p class='lede'>{text}</p>"


def method(summary: str, body: str) -> str:
    """A caveat, folded away.

    Every section here ends in a paragraph about how the number was arrived at,
    and each one is worth reading once. Printing all of them at the weight of
    the finding is what made the page unreadable: with nothing to separate the
    result from the footnote, the whole page reads as footnote.
    """
    return (
        f"<details class='method'><summary>{summary}</summary>"
        f"<div class='note'>{body}</div></details>"
    )


def agent_in_plain_words(inputs) -> str:
    """What the robot did, what it will do next, in sentences a child can read.

    The rest of the site is written for someone who already knows what a
    p-value is, and a reader who does not can leave without ever finding out
    that there is a machine here trading its own pretend money in public. That
    is the one thing on the site with a plain answer: it is holding or it is
    not, it bought at a price, and there is a price at which it sells.

    What this must never turn into is an entry signal. The site's own tests
    reject every timing rule it has, this one included, so the block carries
    the refusal in the same plain language as the rest - a reader who can
    follow "it will sell below $46,355" can follow "this rule could not beat
    picking dates at random", and is owed both.
    """
    paper = inputs.paper or {}
    state = (paper.get("payload") or {}).get("state") or {}
    if not state.get("started"):
        return ""

    trades = (paper.get("payload") or {}).get("trades") or []
    weight = float(state.get("weight") or 0.0)
    holding = weight > 0.0
    flip = state.get("flipLevel")
    capital = float(state.get("capital") or 0.0)
    equity = float(state.get("equity") or 0.0)
    hold = float(state.get("hold") or 0.0)

    parts = [
        "<section><h2>What the robot is doing with its own money</h2>",
        lede("A machine on this site runs a pretend portfolio in the open. "
             "Here is what it holds, and what would make it act next."),
        cards([
            ("Right now", "Holding bitcoin" if holding else "In cash"),
            ("Of the money, in bitcoin", f"{weight:.0%}"),
            ("Sells if tomorrow closes below" if holding
             else "Buys if tomorrow closes above",
             "-" if flip is None else f"${float(flip):,.0f}"),
            ("Since it started",
             "-" if not capital else f"{equity / capital - 1:+.1%}"),
        ]),
    ]

    steps = []
    if holding:
        steps.append(
            f"<b>Right now it owns bitcoin.</b> {weight:.0%} of its money is in "
            "bitcoin, and the rest is sitting in cash."
        )
    else:
        steps.append("<b>Right now it owns no bitcoin at all.</b> All of its "
                     "money is sitting in cash.")

    if trades:
        # "It bought" is not the same sentence as "it bought more", and a
        # reader who is being told this in four lines cannot be left to work
        # out which one happened. Most of these trades are the volatility
        # forecast changing the size of a position that was already open.
        last = trades[-1]
        came = float(last.get("weight_from") or 0.0)
        went = float(last.get("weight_to") or 0.0)
        if went > came:
            what = ("put its cash into bitcoin" if came <= 0
                    else "bought a bit more bitcoin")
        else:
            what = ("sold all of its bitcoin and went back to cash" if went <= 0
                    else "sold a bit of its bitcoin")
        price = last.get("price")
        priced = "" if price is None else f", at ${float(price):,.0f}"
        steps.append(
            f"<b>Last time it acted</b> was {last.get('date')}: it {what}"
            f"{priced}, going from {came:.0%} in bitcoin to {went:.0%}."
        )

    if flip is not None:
        level = f"${float(flip):,.0f}"
        # Not "it sells below X" - the level is worked out from the two moving
        # averages and is exact for tomorrow's close only. Said as a standing
        # floor it becomes a promise the number cannot keep, and it moves far
        # enough between days to make a fool of anyone who wrote it down.
        steps.append(
            f"<b>Next it sells</b>, and it sells if <b>tomorrow</b> closes "
            f"below {level}. That price is for tomorrow only - it is worked "
            "out fresh each day, and it moves. Until it is hit, it does nothing."
            if holding else
            f"<b>Next it buys</b>, and it buys if <b>tomorrow</b> closes above "
            f"{level}. That price is for tomorrow only - it is worked out fresh "
            "each day, and it moves. Until it is hit, it does nothing."
        )

    if capital:
        steps.append(
            f"<b>The score so far.</b> It began with ${capital:,.0f} on "
            f"{state['started']} and has ${equity:,.0f} today "
            f"({equity / capital - 1:+.1%}). Someone who bought once on that "
            f"same day and never touched it again would have ${hold:,.0f} "
            f"({hold / capital - 1:+.1%})."
        )

    parts.append("<ul class='steps'><li>" + "</li><li>".join(steps) + "</li></ul>")
    parts.append(
        "<div class='verdict'><b>Please do not copy it.</b> It follows one "
        "simple rule - buy when the 50-day average price crosses above the "
        "200-day one, sell when it crosses back. This project tested that rule "
        "and <b>could not show it beats buying on dates picked out of a hat</b>. "
        "The robot is here so you can watch a rule fail in public, with the "
        "dates written down in advance. It is not here to be followed, and "
        "nothing on this site is advice.</div>"
    )
    parts.append(method(
        "The small print behind those four sentences",
        "The buy and sell prices above are the direction rule flipping. How "
        "much it holds is a separate decision made by the volatility forecast, "
        "so the size of the position can change on a day when the rule itself "
        "says nothing. Fees and slippage are charged on every change. "
        "<a href='trader.html'>The agent page</a> shows every trade and the "
        "curve behind them.",
    ))
    parts.append("</section>")
    return "".join(parts)


def _trend_rule(inputs):
    """The 50/200 rule's report, which is the one the portfolio follows."""
    for item in (inputs.signals or {}).get("strategies", []):
        if item["report"].name == "trend 50/200":
            return item["report"]
    return None


def hero(inputs) -> str:
    """The front page in one screen: where it stands, and where it traded.

    This replaced three sections of prose. Where a rule bought is a fact with
    a date and a price; a chart says it in one glance and lets a reader check
    it against the trade list. The one thing the picture must not imply is a
    direction - nothing on it continues past today except an interval, and the
    interval is labelled as one.
    """
    signals = inputs.signals or {}
    close = signals.get("price")
    if close is None or not len(close):
        return ""
    report = _trend_rule(inputs)
    trigger = None if report is None else report.trigger.level
    state = ((inputs.paper or {}).get("payload") or {}).get("state") or {}
    trades = ((inputs.paper or {}).get("payload") or {}).get("trades") or []
    price = float(close.iloc[-1])

    holding = float(state.get("weight") or 0.0) > 0.0
    parts = [
        "<section class='hero'>",
        cards([
            ("Trend, measured today",
             "-" if report is None else ("Up" if report.side == "long" else "Down")),
            ("Price", f"${price:,.0f}"),
            ("The robot", ("Holding" if holding else "In cash") if state else "-"),
            # "It sells below" was a lie of tense. The level is exact for one
            # close and moves the day after, so the label has to carry the day.
            ("Sells if tomorrow closes below",
             "-" if trigger is None else f"${float(trigger):,.0f}"),
        ]),
        figure_html(
            tape_chart(close, trades, trigger,
                       inputs.range_forecast, inputs.range_days),
            "tape",
        ),
        lede(
            "<b>Green is where it bought. Red is where it sold.</b> The short "
            "red dash at the right is tomorrow's sell level - good for one "
            "close and no further, because it moves as the averages move. "
            "Nothing here runs past today except the blue bar, and that bar is "
            "how far the price can travel in 30 days, not which way."
        ),
        "</section>",
    ]
    return "".join(parts)


def which_way(inputs) -> str:
    """The question everyone arrives with, answered in a picture.

    The seven-column table of p-values that used to sit here is now on the
    evidence page. What it was trying to say is one chart: the grey bar is the
    same rule entered on dates out of a hat, and no rule clears its own grey
    bar by enough to matter.
    """
    edge = (inputs.saved or {}).get("backtest_edge", pd.DataFrame())
    parts = [
        "<section><h2>Which way next?</h2>",
        "<div class='verdict'><b>Nobody here can tell you, and this page is the "
        "reason.</b> Every timing rule on this site was tested and rejected. An "
        "arrow pointing up or down would be invented.</div>",
    ]
    if not edge.empty and {"strategy", "sharpe", "sharpe_random_timing"} <= set(edge.columns):
        parts.append(figure_html(edge_chart(edge), "edge"))
        parts.append(lede(
            "Gold is each rule as written. Grey is <b>the same rule entered on "
            "dates picked out of a hat</b>. A rule worth having would leave its "
            "own grey bar behind."
        ))
        # One bar always looks like it won, and a picture cannot say why it did
        # not. Naming it here is the difference between a chart that tells the
        # truth and a chart that needs a footnote nobody opens.
        widest = (edge["sharpe"] - edge["sharpe_random_timing"]).idxmax()
        top = edge.loc[widest]
        if "survives_correction" in edge.columns and not bool(top["survives_correction"]):
            worst = top.get("p_worst_without_one")
            tail = ""
            if worst is not None and not pd.isna(worst):
                tail = (f" Drop its single best trade and its p-value goes from "
                        f"{float(top['p_value']):.3f} to {float(worst):.2f}.")
            parts.append(lede(
                f"<b>One bar does leave its grey behind: {top['strategy']}.</b> "
                f"It is also the one the tests throw out.{tail} That is what the "
                "rest of this site is for."
            ))
        parts.append(
            "<p class='note'>The p-values behind the picture, and the test that "
            "drops the best trade to see what is left, are on "
            "<a href='evidence.html'>the evidence page</a>.</p>"
        )
    parts.append("</section>")
    return "".join(parts)


def how_far(inputs) -> str:
    """The one forward-looking number the project stands behind, drawn."""
    saved = inputs.saved or {}
    close = (inputs.signals or {}).get("price")
    if close is None or not len(close):
        return ""
    price = float(close.iloc[-1])

    bars = []
    for label, key in (("In 10 days", "range_10"), ("In 30 days", "range_30")):
        frame = saved.get(key, pd.DataFrame())
        if frame.empty:
            continue
        best = frame.loc[frame["level"].idxmax()]
        low, high = float(best["low"]), float(best["high"])
        span = high - low
        at = 0.5 if span <= 0 else min(max((price - low) / span, 0.0), 1.0)
        bars.append(
            "<div class='range'>"
            f"<div class='range-top'><b>{label}</b>"
            f"<span>{float(best['level']):.0%} of the time</span></div>"
            f"<div class='range-bar'><i style='left:{at:.1%}'></i></div>"
            "<div class='range-ends'>"
            f"<span><u>{best['low_pct']:+.0%}</u>${low:,.0f}</span>"
            f"<span><u>{best['high_pct']:+.0%}</u>${high:,.0f}</span>"
            "</div></div>"
        )
    if not bars:
        return ""
    return (
        "<section><h2>How far, though - that it can say</h2>"
        + lede("Volatility is the one thing on this site that is forecastable. "
               "The marker is today's price; the bar is where it can be.")
        + "<div class='ranges'>" + "".join(bars) + "</div>"
        + method(
            "How the intervals were checked",
            "Each level passed a coverage test on non-overlapping windows: a 90% "
            "interval really did contain the outcome about 90% of the time. "
            "Levels that failed are withheld rather than quietly widened.",
        )
        + "</section>"
    )


def workings() -> str:
    """The rest of the site, as a row of doors rather than a row of tabs."""
    cells = "".join(
        f"<a class='door' href='{href}'><b>{title}</b><span>{blurb}</span></a>"
        for href, title, blurb in WORKINGS
    )
    return (
        "<section><h2>The full workings</h2>"
        + lede("Everything above is a summary. Nothing below it is hidden.")
        + f"<div class='doors'>{cells}</div></section>"
    )


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
            + lede(
                f"A year after a halving, the abnormal return is "
                f"<b>{summary['car']:+.0%}</b> - somewhere between "
                f"{summary['ci_low']:+.0%} and {summary['ci_high']:+.0%}, "
                f"p={summary['p_value']:.2f}."
            )
            + method(
                "Why the interval is that wide",
                "The sample holds four events. That is not a flaw in the method; "
                "it is all the information there is.",
            )
            + "</section>"
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
            + lede(
                f"<b>{survivors} of {len(scan)}</b> hypotheses survive the "
                "correction for how many were asked."
            )
            + table(shown, highlight="significant_adjusted")
            + method(
                "What is in the table",
                f"The twelve smallest p-values out of {len(scan)} hypotheses. "
                "Testing enough ideas produces small p-values by itself, which is "
                "what the Benjamini-Hochberg correction is for.",
            )
            + "</section>"
        )

    if not inputs.curve_summary.empty:
        row = inputs.curve_summary.iloc[0]
        parts.append(
            "<section><h2>One question, 160 ways of asking it</h2>"
            + lede(
                f"<b>{int(row['n_significant'])} of {int(row['n_specs'])}</b> "
                "specifications come out significant - fewer than randomly placed "
                f"dates manage ({row['null_significant_mean']:.1f})."
            )
            + cards([
                ("Specifications", f"{int(row['n_specs'])}"),
                ("Significant", f"{int(row['n_significant'])}"),
                ("Random dates give", f"{row['null_significant_mean']:.1f}"),
                ("Median effect", f"{row['median_car']:+.1%}"),
            ])
            + method(
                "Why ask the same question 160 ways",
                "A finding that depends on which reasonable choices were made is "
                "not a finding.",
            )
            + "</section>"
        )

    edge = saved.get("backtest_edge", pd.DataFrame())
    if not edge.empty:
        columns = [
            column for column in (
                "strategy", "sharpe", "sharpe_random_timing", "p_value",
                "p_worst_without_one", "fragile", "survives_correction",
            ) if column in edge.columns
        ]
        parts.append(
            "<section><h2>Every rule, against itself entered at random</h2>"
            + lede("<b>Nothing survives.</b> The front page draws this table; "
                   "these are the numbers under the picture.")
            + table(edge.loc[:, columns].round(3), highlight="survives_correction")
            + method(
                "How each rule was tested",
                "Every rule is compared against ITSELF applied at random times, so "
                "the bar is its own random-timing Sharpe rather than zero. A p "
                "under 0.05 would mean the timing beat chance. <b>p, best trade "
                "removed</b> re-runs the test with one trade dropped, and "
                "<b>survives correction</b> accounts for having tried several rules "
                "at once. Hover a column heading for what it measures.",
            )
            + "</section>"
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
            + lede(
                "Beating buy-and-hold on Sharpe is <b>not enough</b>: a rule also "
                "has to beat its own random-timing version, which is the table on "
                "the first page."
            )
            + table(backtest.loc[:, columns].round(3), highlight="sharpe")
            + method(
                "What the returns are charged",
                "Fees of 10 bps and slippage of 15 bps are charged on turnover.",
            )
            + "</section>"
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
            + lede(
                "The model has to beat <code>always_up</code> rather than a coin "
                "flip. <b>It does not clear it.</b>"
            )
            + table(forecast.loc[:, columns].round(4), highlight="brier")
            + method(
                "How it was scored",
                "Scored on non-overlapping rows only. This series rose in most "
                "windows, so the base rate - not 50% - is the bar.",
            )
            + "</section>"
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
            + lede(
                "Direction is not forecastable here. <b>The size of the next move "
                "is</b> - volatility clusters, so holding target volatility divided "
                "by forecast volatility keeps the risk constant rather than the "
                "quantity."
            )
            + table(sizing.loc[:, columns].round(3), highlight="sharpe")
            + method(
                "Why there is a rebalance band at all",
                "Retrading on every wobble in the forecast costs more than the "
                "sizing is worth.",
            )
            + "</section>"
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
            + lede(
                "An effect that is real <b>keeps its sign</b>; one that is an "
                "artefact of a particular stretch of history does not."
            )
            + table(shown, highlight="sign_p_adjusted")
            + method(
                "How the windows were cut",
                f"Each hypothesis re-tested on {int(walk['n_folds'].max())} disjoint "
                "windows with an embargo between them. The ten smallest p-values "
                "are shown, already corrected.",
            )
            + "</section>"
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
            + lede(
                f"<b>{replicated} of {len(replication)}</b> replicate on the cycles "
                "held back."
            )
            + table(replication.loc[:, columns].head(10).round(4),
                    highlight="replicated")
            + method(
                "Why hold anything back",
                "Fitted on early cycles, checked on the ones held back. A "
                "hypothesis that only works on the data that suggested it is a "
                "description of that data.",
            )
            + "</section>"
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
            + lede(
                "Every category gets the same event study, the same intervals and "
                "the same counts - and <b>the counts are what to read first</b>."
            )
            + table(categories.loc[:, columns].round(4), highlight="n")
            + method(
                "What is in the registry",
                "Halvings are not the only events recorded here; the rest are "
                "treated no differently.",
            )
            + "</section>"
        )

    parts.append(
        "<section><h2>The control group</h2><p class='note'>"
        + inputs.control_note
        + "</p></section>"
    )
    return "".join(parts)
