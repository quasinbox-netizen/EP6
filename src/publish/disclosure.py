"""The two things a published number owes the reader: where it came from, and
when it changed.

This module exists because the project failed both tests on itself.

A restatement was shipped silently. The paper portfolio traded on a two-day-old
signal for as long as it had been published, the lag was fixed, and the headline
went from -14.5% to -7.7% overnight. The commit message explained it. The page
did not, and readers do not read git log. A number that improves by seven points
with no notice attached is indistinguishable from a number that was quietly
tuned, and the difference is the only thing this project sells.

And two constants were chosen on this dataset without saying so. The volatility
target is BTC's own sample median. The rebalance band is whichever value scored
the highest Sharpe among those compared - selected on the same history the edge
test then scores. Both are defensible; neither is an outside convention, and the
site described them as though they were.

So: CORRECTIONS is append-only and rendered on the page whose numbers moved, and
the provenance of every constant that sizes the portfolio is printed next to the
portfolio. Neither is decoration. A restatement that is not in this list is a
restatement the reader never sees.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Correction:
    """One published number that changed meaning, and why.

    `was` and `now` are the figures as they appeared, verbatim, because a
    correction that does not carry the old number cannot be checked against
    what the reader remembers.
    """

    date: str
    page: str
    headline: str
    was: str
    now: str
    why: str


# Append-only. Entries are never edited or removed: an old correction stops
# being interesting long before it stops being the evidence that the last one
# was disclosed.
CORRECTIONS: tuple[Correction, ...] = (
    Correction(
        date="2026-09-07",
        page="trader.html",
        headline="The portfolio was trading on a two-day-old signal",
        was="-14.5% since the last halving",
        now="-7.7% since the last halving",
        why=(
            "The paper account applies the one-day execution lag itself, and it was "
            "being handed a series the backtest engine had already lagged. Every "
            "entry and exit therefore landed a day later than the rule says, and "
            "the account paid for a day of moves it was not in. The published "
            "trades moved with it: the 2024-08-11 sell is now 2024-08-10, and so "
            "on down the ledger. Nothing about the rule changed - it was never "
            "as bad as the page said, and it is still losing to doing nothing."
        ),
    ),
    Correction(
        date="2026-09-08",
        page="trader.html",
        headline="The sweeps were computed on a price history that has been re-stitched",
        was="the band ranked 34th of 61, the target 14th of 39",
        now="the band ranks last of 61, the target 36th of 39",
        why=(
            "The volatility forecast is built from the stitched price series and "
            "was cached. Since it was last built, the stitch began preferring "
            "Binance from 2017-08-17 - the date Binance's history starts - in "
            "place of Bitstamp, which changed 3,305 of the 4,035 daily forecasts "
            "behind every sizing figure on this page. The estimator itself is "
            "unchanged and deterministic: the same prices give the same numbers, "
            "and four more days of data on their own change nothing before the "
            "last day. What moved was the input. The forecast is now rebuilt "
            "every morning so it cannot drift from the prices again. Read the "
            "size of this move as what it is: a parameter whose rank falls from "
            "the middle of its grid to the bottom when the price source changes "
            "underneath it was never ranked on anything real, which is what the "
            "sweep below already said in a weaker way."
        ),
    ),
)


def corrections_for(page: str) -> tuple[Correction, ...]:
    return tuple(item for item in CORRECTIONS if item.page == page)


def corrections_html(page: str) -> str:
    """The correction notice, newest first, or nothing if the page has none."""
    items = corrections_for(page)
    if not items:
        return ""
    rows = []
    for item in sorted(items, key=lambda c: c.date, reverse=True):
        rows.append(
            f"<p style='margin:0 0 10px'><b>{item.date} - {item.headline}.</b> "
            f"This page said <b>{item.was}</b>; it now says <b>{item.now}</b>. "
            f"{item.why}</p>"
        )
    return (
        "<section><h2>Corrections</h2>"
        "<div class='verdict' style='border-left-color:#f7931a'>"
        + "".join(rows)
        + "<p class='note' style='margin:0'>Every restatement of a published "
        "figure is listed here, permanently, with the old number kept next to "
        "the new one.</p></div></section>"
    )


def provenance_html(sizing: pd.DataFrame, comparison: pd.DataFrame) -> str:
    """Where the two constants that size the portfolio actually came from.

    Both are read out of the saved results rather than written here, so the
    page cannot claim a target the run did not use. The comparison table is
    printed in full - including any row that beats the adopted one, which is
    the whole reason for printing it.
    """
    if sizing.empty:
        return ""
    row = sizing.iloc[0]
    band = float(row["band"])
    median = float(row["median_annual_volatility"])
    target = row.get("target_annual_volatility")
    named = "" if target is None or pd.isna(target) else f" ({float(target):.0%})"

    target_line = (
        f"<li><b>The volatility target{named}</b> is not an outside convention. "
        "It is BTC's own median forecast volatility over this history, which "
        f"the same run measured at <b>{median:.2%}</b>. A conventional target - "
        "the 15% a multi-asset book would use - was rejected because on this "
        "asset it holds a few percent almost always, and the result would then "
        "describe the cap rather than the method. That is a reasonable choice, "
        "and it was made with this dataset in view.</li>"
    )

    beaten = ""
    if not comparison.empty and "sharpe" in comparison.columns:
        adopted = f"vol target, band {band:.0%}"
        table = comparison
        if "strategy" in table.columns:
            table = table.set_index("strategy")
        if adopted in table.index:
            mine = float(table.loc[adopted, "sharpe"])
            better = [
                (name, float(value)) for name, value in table["sharpe"].items()
                if value > mine and name != "buy and hold"
            ]
            if better:
                names = ", ".join(
                    f"<b>{name}</b> at {value:.4f}" for name, value in better)
                beaten = (
                    f"<p class='note'>The adopted row scores {mine:.4f}. Among the "
                    f"variants compared, {names} scores higher. It was not adopted, "
                    "and this sentence is here so that choice is visible rather than "
                    "buried in a table nobody sorts.</p>"
                )

    return (
        "<section><h2>Where these two numbers came from</h2>"
        "<div class='verdict'>"
        "<b>Both constants that size this portfolio were chosen with this "
        "dataset in front of us.</b> The project spends most of its pages "
        "rejecting rules for exactly that, so it says so about itself here."
        "</div>"
        "<ul>"
        + target_line
        + f"<li><b>The rebalance band ({band:.0%})</b> was picked as the "
        "highest-Sharpe variant among those compared - three of them - and the "
        "comparison ran on the same history the edge test then scores. That is "
        "selection in sample, and the sweep below shows what it was worth. "
        "The band's job is to cut turnover, and the sweep prices that job; on "
        "it the choice is cheap to defend. As a return claim it is not, "
        "because a value chosen for having the best score on a sample has no "
        "score left to report on that sample.</li>"
        "</ul>"
        + beaten
        + "<p class='note'>Neither number is being defended as optimal. They are "
        "being disclosed as fitted, so that anything built on top of them "
        "inherits the caveat instead of shedding it.</p></section>"
    )


def sweep_html(sweep: pd.DataFrame, adopted: float, chart: str = "") -> str:
    """What every value of the band earned, and why that disqualifies ranking them.

    Every figure is read out of the saved sweep. The conclusion this prints is
    the one the numbers support and not a softer one: the band in use is not
    the best value, the best value is not stable, and the highest scores belong
    to bands so wide the rule barely trades.
    """
    if sweep.empty or "sharpe" not in sweep.columns:
        return ""
    table = sweep.set_index("band") if "band" in sweep.columns else sweep
    sharpe = table["sharpe"]
    # min() over the labels themselves. Subtracting from the index and taking
    # idxmin returns the DIFFERENCE, not the band, which quietly reported the
    # sweep's first row as the band in use.
    nearest = min((float(band) for band in table.index),
                  key=lambda band: abs(band - adopted))
    mine = float(sharpe.loc[nearest])
    best_band, best = float(sharpe.idxmax()), float(sharpe.max())
    rank = int((sharpe > mine).sum()) + 1
    step = float(sharpe.diff().abs().mean())

    # Frozen: the weight never changes again after the entry. See
    # backtest.sweep.FROZEN_CHANGES for why this counts rather than thresholds.
    frozen = (
        table.index[table["n_position_changes"] <= 1]
        if "n_position_changes" in table.columns else []
    )
    # Read, not written: the daily refresh moves these, and a hardcoded "8.9x"
    # was already wrong the first morning the job ran on its own.
    turnover = ""
    if "turnover_annual" in table.columns:
        busiest = float(table["turnover_annual"].max())
        here = float(table["turnover_annual"].loc[nearest])
        turnover = (
            f"it takes annual turnover from {busiest:.1f}x, with no band at all, "
            f"to {here:.1f}x"
        )

    frozen_line = ""
    if len(frozen):
        frozen_line = (
            f"<li>From <b>{float(min(frozen)):.0%}</b> the band is wider than the "
            "position ever moves. The rule stops trading, holds whatever it last "
            "held, and inherits buy-and-hold's Sharpe by construction - and the "
            "highest-scoring bands on the whole grid sit directly against that "
            "edge. Scoring well by nearly switching yourself off is not a "
            "finding, and it is most of what the right-hand side of this chart "
            "is showing.</li>"
        )

    return (
        "<section><h2>Every value of the band, not the three that were tried</h2>"
        "<div class='verdict'>"
        f"<b>The band in use is not the best value, and the best value is not "
        f"worth having.</b> Across {len(table)} settings it ranks "
        f"<b>{rank}</b> - below the middle of its own sweep."
        "</div>"
        + chart
        + "<ul>"
        f"<li>The band in use ({nearest:.0%}) scores <b>{mine:.4f}</b>. The best "
        f"({best_band:.0%}) scores <b>{best:.4f}</b>. It was adopted for winning "
        "a three-point comparison, and a three-point grid cannot tell a real "
        "maximum from a bump because it has no shape to show.</li>"
        f"<li>Moving the band by one percentage point changes the Sharpe by "
        f"<b>{step:.4f}</b> on average, in a grid that spans only "
        f"{sharpe.max() - sharpe.min():.4f} end to end. A surface that jerks "
        "like that under a change nobody would defend on its merits is not "
        "ranking strategies; it is ranking noise.</li>"
        + frozen_line
        + "</ul>"
        "<p class='note'>Nothing here was changed in response. Moving the band "
        f"to {best_band:.0%} because it won this sweep is precisely the mistake "
        "the sweep exposes, and it would need its own out-of-sample test before "
        "it meant anything. The band stays where it is, defended on the one "
        f"ground that does not move: {turnover or 'it cuts turnover sharply'}, "
        "and turnover is a cost, not a score.</p></section>"
    )


def target_sweep_html(sweep: pd.DataFrame, adopted: float, chart: str = "") -> str:
    """Every volatility target, and why its Sharpe column cannot choose one.

    This section makes a different argument from the band's. The band's problem
    was that its ranking is noise. The target's problem is that ranking it on a
    Sharpe was never the right question: it sets how much risk to carry, and a
    Sharpe ratio is built not to see how much risk is being carried. The figure
    that does see it - the share of days the position is jammed against the
    no-borrowing cap - had never been published at all.
    """
    if sweep.empty or "sharpe" not in sweep.columns:
        return ""
    table = sweep.set_index("target") if "target" in sweep.columns else sweep
    sharpe = table["sharpe"]
    nearest = min((float(value) for value in table.index),
                  key=lambda value: abs(value - adopted))
    mine = float(sharpe.loc[nearest])
    best_value, best = float(sharpe.idxmax()), float(sharpe.max())
    rank = int((sharpe > mine).sum()) + 1

    capped = ""
    if "at_the_cap" in table.columns:
        share = float(table["at_the_cap"].loc[nearest])
        pinned = table.index[table["at_the_cap"] >= 0.99]
        tail = ""
        if len(pinned):
            tail = (
                f" From a target of <b>{float(min(pinned)):.0%}</b> the position "
                "is at the cap on virtually every day, and the rule is simply "
                "buy-and-hold wearing a sizing rule's name - which is what the "
                "drawdown line flattening onto buy-and-hold's own drawdown, at "
                "the right of the chart, is showing."
            )
        capped = (
            f"<li><b>On {share:.1%} of days the position is pinned at the "
            "no-borrowing cap.</b> On those days the rule is not sizing "
            "anything: it wants more of the asset than it is allowed to hold, "
            "so it holds all of it and stops responding to the forecast. The "
            "target was set to this asset's own median volatility, and that is "
            f"what putting it there does.{tail}</li>"
        )

    drawdown = ""
    if "max_drawdown" in table.columns:
        worst = float(table["max_drawdown"].min())
        gentlest = float(table["max_drawdown"].max())
        here = float(table["max_drawdown"].loc[nearest])
        drawdown = (
            f"<li><b>Across this grid the worst drawdown runs from {gentlest:.0%} "
            f"to {worst:.0%}</b>, and at the target in use it is {here:.0%}. That "
            "is the column the target actually chooses. The Sharpe column moves "
            f"by {sharpe.max() - sharpe.min():.4f} end to end and is close to "
            "useless here, because a Sharpe ratio is built to be indifferent to "
            "how large a position is - which is the one thing this parameter "
            "sets.</li>"
        )

    return (
        "<section><h2>Every volatility target, and why its score cannot pick one</h2>"
        "<div class='verdict'>"
        "<b>This one is not a performance setting. It is a dial on how much "
        "risk to carry</b>, and it was chosen by reading the statistic designed "
        f"not to see risk. On that statistic it ranks <b>{rank}</b> of "
        f"{len(table)}."
        "</div>"
        + chart
        + "<ul>"
        f"<li>The target in use ({nearest:.0%}) scores <b>{mine:.4f}</b>; the "
        f"best ({best_value:.0%}) scores <b>{best:.4f}</b>. As with the band, "
        "that gap is not worth acting on - and unlike the band, it would not "
        "even be the right reason to act.</li>"
        + capped
        + drawdown
        + "</ul>"
        "<p class='note'>The target did not move either. What it should be "
        "defended on is the risk level a reader is willing to carry, which is a "
        "choice rather than a measurement - and the honest version of that "
        "statement includes the fact that at this setting the rule spends half "
        "its days holding everything it is allowed to hold.</p></section>"
    )


def rank_history_html(history: pd.DataFrame, chart: str = "",
                      minimum: int = 3) -> str:
    """The sweeps' argument, made by measurement instead of by statistics.

    The sweeps say a ranking computed on this sample means nothing, and support
    it with numbers taken from inside a single run. This says the same thing by
    keeping the ranking and watching it move, which is the version that does
    not require the reader to trust a standard error.

    Below `minimum` distinct days there is no movement to show, and the section
    says so instead of drawing a line through two points. A chart that claimed
    wandering from two observations would be doing what the sweeps warn against.
    """
    if history.empty or "parameter" not in history.columns:
        return ""
    days = int(history["recorded"].nunique())
    parts = ["<section><h2>Where these constants have placed, run after run</h2>"]

    if days < minimum:
        runs = "run" if days == 1 else "runs"
        return "".join(parts + [
            "<p class='note'>The record starts here and holds "
            f"<b>{days} {runs}</b> so far, which is not enough to draw a line "
            "through. It is written every time the sizing is rebuilt, which is "
            "now every morning, so this becomes a chart on its own without "
            "anybody deciding to make one.</p>"
            "<p class='note'>What is already known sits in the corrections "
            "below: the band placed in the middle of its grid on one price "
            "series and last on the restated one, without a line of code "
            "changing between the two. That is a same-day restatement rather "
            "than a series over time, which is why it is a correction and not "
            "a point on this chart.</p></section>",
        ])

    parts.append(chart)
    lines = []
    for name in sorted(history["parameter"].unique()):
        rows = history.loc[history["parameter"] == name].sort_values("recorded")
        best, worst = int(rows["rank"].min()), int(rows["rank"].max())
        moved = int(rows["best"].nunique())
        lines.append(
            f"<li>The <b>{name}</b> in use has placed between <b>{best}</b> and "
            f"<b>{worst}</b> of {int(rows['of'].iloc[-1])} across "
            f"{int(rows['recorded'].nunique())} runs, and the value that scored "
            f"highest has been {moved} different one{'s' if moved != 1 else ''} "
            "over the same period.</li>"
        )
    parts.append("<ul>" + "".join(lines) + "</ul>")
    parts.append(
        "<p class='note'>Nothing about the rule changes between these points. "
        "The sample grows by a day, and the answer to \"which value is best\" "
        "moves. That is the sweeps' argument again, made by measurement rather "
        "than by a standard error the reader has to take on faith - and it is "
        "the reason neither constant has been moved to whichever value happens "
        "to be winning today.</p></section>"
    )
    return "".join(parts)
