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
        "highest-Sharpe variant among those compared, and the comparison ran "
        "on the same history the edge test then scores. That is selection in "
        "sample. The band's job is to cut turnover - it takes annual turnover "
        "from roughly 8.9x to 1.1x - and on that job the choice is cheap to "
        "defend; as a return claim it is not, because a value chosen for "
        "having the best score on a sample has no score left to report on "
        "that sample.</li>"
        "</ul>"
        + beaten
        + "<p class='note'>Neither number is being defended as optimal. They are "
        "being disclosed as fitted, so that anything built on top of them "
        "inherits the caveat instead of shedding it.</p></section>"
    )
