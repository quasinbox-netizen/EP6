"""The individual questions asked of a track record.

Each function here answers one question and returns one `Check`. They are
deliberately independent: a check that cannot run on the data provided
returns `N/A` with the reason, and never silently drops out of the report.
An absent test is information, so it is printed.

The ordering below is the order they appear in the report, and it is roughly
"cheapest to argue with" first: descriptive numbers, then how much of them
could be chance, then how much rests on a handful of days, then whether the
result would have survived contact with costs and with the second half of its
own history.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from validation.multiple_testing import benjamini_hochberg, bonferroni
from validation.synthetic import block_bootstrap

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
NA = "N/A"

# Blocks of about a month: long enough to carry volatility clustering and the
# autocorrelation of a position through a resample, short enough that 2000
# draws still explore the sample rather than reproducing it.
BOOTSTRAP_BLOCK = 21
BOOTSTRAP_DRAWS = 2000

# Below this many round trips, a p-value is arithmetic about very little.
MIN_EPISODES_FOR_CONFIDENCE = 8


@dataclass
class Check:
    key: str
    title: str
    question: str
    verdict: str
    headline: str
    detail: str = ""
    numbers: dict = field(default_factory=dict)
    # Blocking checks decide the overall verdict. Descriptive ones (the metric
    # table) inform it without voting.
    blocking: bool = True

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "question": self.question,
            "verdict": self.verdict,
            "headline": self.headline,
            "detail": self.detail,
            "numbers": {k: _plain(v) for k, v in self.numbers.items()},
            "blocking": self.blocking,
        }


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """365 for a series that trades weekends, 252 for one that does not.

    Getting this wrong rescales every Sharpe in the report by 1.2x, which is
    the difference between a mediocre strategy and a good one on paper. It is
    inferred rather than asked because a user who has to answer it will guess.
    """
    if len(index) < 10:
        return 252
    weekend_share = float(pd.DatetimeIndex(index).dayofweek.isin((5, 6)).mean())
    # A calendar-daily series is ~2/7 weekend. Anything under a tenth is a
    # business-day series with the odd stray row.
    return 365 if weekend_share > 0.10 else 252


def sharpe(returns: np.ndarray | pd.Series, periods_per_year: int) -> float:
    values = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if values.size < 2:
        return float("nan")
    deviation = values.std(ddof=1)
    if deviation == 0:
        return float("nan")
    return float(values.mean() / deviation * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def longest_underwater(equity: pd.Series) -> int:
    """Days between a high-water mark and the day it is next exceeded."""
    peaks = equity.cummax()
    under = equity < peaks
    if not under.any():
        return 0
    longest = current = 0
    for flag in under.to_numpy():
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


# --------------------------------------------------------------------------
# 1. What the record says about itself
# --------------------------------------------------------------------------
def describe(data, periods_per_year: int) -> Check:
    net = data.net_returns.dropna()
    equity = data.equity
    years = len(net) / periods_per_year
    # equity starts at exactly 1.0, so the ratio and the last value agree.
    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan")
    drawdown = max_drawdown(equity)
    numbers = {
        "first_day": data.span[0],
        "last_day": data.span[1],
        "observations": int(len(net)),
        "years": round(years, 2),
        "total_return": total,
        "cagr": cagr,
        "volatility": float(net.std(ddof=1) * np.sqrt(periods_per_year)),
        "sharpe": sharpe(net, periods_per_year),
        "max_drawdown": drawdown,
        "longest_underwater_days": longest_underwater(equity),
        "win_rate": float((net > 0).mean()),
        "periods_per_year": periods_per_year,
    }
    if data.positions is not None:
        active = data.positions.reindex(net.index).fillna(0.0)
        numbers["time_in_market"] = float((active.abs() > 0).mean())
        numbers["turnover_total"] = float(active.diff().abs().sum())
    return Check(
        key="record",
        title="The record as given",
        question="What does this track record claim, before anyone argues with it?",
        verdict=NA,
        headline=(
            f"{total:+.1%} over {years:.1f} years ({cagr:+.1%} a year), "
            f"Sharpe {numbers['sharpe']:.2f}, worst drawdown {drawdown:.1%}."
        ),
        detail=(
            "These are the numbers the file contains. Not one of them is evidence "
            "of anything yet - every check below exists because a number in this "
            "row can be produced by luck, by costs left out, or by a handful of days."
        ),
        numbers=numbers,
        blocking=False,
    )


# --------------------------------------------------------------------------
# 2. Would random timing of the same rule have done as well?
# --------------------------------------------------------------------------
def retiming(data, *, permutations: int, cost_rate: float, periods_per_year: int) -> Check:
    question = "Would this rule, applied at random times, have done as well?"
    if not data.can_retime:
        return Check(
            key="retiming", title="Timing vs chance", question=question, verdict=NA,
            headline="Not available: the file has no position column next to a price.",
            detail=(
                "This is the strongest test in the report and the only one that can "
                "distinguish a rule that picks moments from a rule that is simply "
                "invested a lot in a rising market. It needs to know WHEN the rule "
                "held a position. Re-export with a `close` column and a `position` "
                "column (1.0 invested, 0 flat) to unlock it."
            ),
        )

    from backtest.edge import edge_test  # local import: heavy module, optional path

    try:
        result = edge_test(
            data.positions, data.asset_returns, cost_rate=cost_rate,
            n_permutations=permutations, statistic="sharpe",
            periods_per_year=periods_per_year,
        )
    except ValueError as exc:
        return Check(
            key="retiming", title="Timing vs chance", question=question, verdict=NA,
            headline=f"Not available: {exc}",
        )

    numbers = {
        "observed_sharpe": result.observed,
        "random_timing_sharpe": result.null_mean,
        "random_timing_p95": result.null_p95,
        "p_value": result.p_value,
        "permutations": result.n_permutations,
        "exposure": result.exposure,
        "round_trips": result.n_trades,
    }
    if not np.isfinite(result.p_value):
        return Check(
            key="retiming", title="Timing vs chance", question=question, verdict=NA,
            headline="The permutation test could not produce a p-value on this series.",
            numbers=numbers,
        )

    if result.p_value < 0.05:
        return Check(
            key="retiming", title="Timing vs chance", question=question, verdict=PASS,
            headline=(
                f"Timing beats chance on this history (p={result.p_value:.4f}): the same "
                f"rule re-timed at random scores {result.null_mean:.2f} against "
                f"{result.observed:.2f} observed."
            ),
            detail=(
                "The null here keeps the price path exactly as it happened and moves "
                "the rule instead, so exposure, turnover and costs are unchanged and "
                "only the alignment with the market is destroyed. Passing means the "
                "alignment was worth something ON THIS SAMPLE. It does not survive "
                "being one of many rules tried - see the multiplicity check - and it "
                "is only as strong as the number of round trips behind it."
            ),
            numbers=numbers,
        )
    return Check(
        key="retiming", title="Timing vs chance", question=question, verdict=FAIL,
        headline=(
            f"No timing edge (p={result.p_value:.4f}). Applied at random times the same "
            f"rule averages {result.null_mean:.2f} against {result.observed:.2f} observed."
        ),
        detail=(
            "Whatever this record earned, the evidence does not support the rule's "
            "choice of moments as the reason. A rule invested "
            f"{result.exposure:.0%} of the time in a market that rose will show a "
            "return whether or not its logic works."
        ),
        numbers=numbers,
    )


# --------------------------------------------------------------------------
# 3. How much of it rests on one trade?
# --------------------------------------------------------------------------
def fragility_check(data, *, permutations: int, cost_rate: float, periods_per_year: int) -> Check:
    question = "Does the result survive removing its single best episode?"
    if not data.can_retime:
        return Check(
            key="fragility", title="Fragility", question=question, verdict=NA,
            headline="Not available without a position column: episodes cannot be identified.",
        )

    from backtest.edge import fragility

    try:
        result = fragility(
            data.positions, data.asset_returns, cost_rate=cost_rate,
            n_permutations=max(200, permutations // 4),
            periods_per_year=periods_per_year,
        )
    except ValueError as exc:
        return Check(key="fragility", title="Fragility", question=question,
                     verdict=NA, headline=f"Not available: {exc}")

    numbers = {
        "episodes": result.n_episodes,
        "p_value_full": result.full_p_value,
        "p_value_worst_without_one": result.worst_p_value,
        "worst_episode": result.worst_episode,
    }
    if result.n_episodes < MIN_EPISODES_FOR_CONFIDENCE:
        base = (
            f"Only {result.n_episodes} separate trades. A p-value computed from this "
            "is as precise as the number of permutation draws and as strong as "
            f"{result.n_episodes} observations, and those are not the same number."
        )
        return Check(key="fragility", title="Fragility", question=question,
                     verdict=WARN, headline=base,
                     detail=(
                         "Permutation draws multiply the arrangements of the evidence, "
                         "not the evidence. Treat any significance above as provisional "
                         "until the rule has traded more."
                     ),
                     numbers=numbers)

    if result.fragile:
        return Check(
            key="fragility", title="Fragility", question=question, verdict=FAIL,
            headline=(
                f"Fragile: removing one trade of {result.n_episodes} takes the p-value "
                f"from {result.full_p_value:.4f} to {result.worst_p_value:.4f}."
            ),
            detail=(
                "The finding rests on a single episode. That is one observation, "
                "however many decimal places the p-value carries, and one observation "
                "is not a strategy."
            ),
            numbers=numbers,
        )
    return Check(
        key="fragility", title="Fragility", question=question, verdict=PASS,
        headline=(
            f"Robust to leaving one out: across {result.n_episodes} trades the worst "
            f"single removal moves the p-value only to {result.worst_p_value:.4f}."
        ),
        numbers=numbers,
    )


# --------------------------------------------------------------------------
# 4. How many variants were tried before this one?
# --------------------------------------------------------------------------
def multiplicity(primary_p: float, *, variants_tried: int | None, alpha: float = 0.05) -> Check:
    question = "How many versions were tried before this one was kept?"
    if not np.isfinite(primary_p):
        return Check(
            key="multiplicity", title="Multiple testing", question=question, verdict=NA,
            headline="No p-value available to correct.",
        )
    if variants_tried is None:
        return Check(
            key="multiplicity", title="Multiple testing", question=question, verdict=WARN,
            headline=(
                "Not declared. If this is the only rule and the only parameter set ever "
                f"tested, p={primary_p:.4f} stands as printed. If it is not, it is wrong."
            ),
            detail=(
                "Run 100 rules on pure noise and about five come back significant at "
                "0.05 - and it is those five that get written up. Re-run with "
                "--variants-tried N, counting every parameter set, every lookback and "
                "every asset you tried and discarded. The count is the honest part; "
                "the arithmetic is trivial."
            ),
            numbers={"p_value_raw": primary_p},
        )

    n = max(1, int(variants_tried))
    adjusted_bonf = float(bonferroni(np.array([primary_p]), n_tests=n)[0])
    adjusted_bh = float(benjamini_hochberg(np.array([primary_p]), n_tests=n)[0])
    numbers = {
        "variants_tried": n,
        "p_value_raw": primary_p,
        "p_value_bonferroni": adjusted_bonf,
        "p_value_benjamini_hochberg": adjusted_bh,
        "expected_false_positives": n * alpha,
    }
    if n == 1:
        return Check(
            key="multiplicity", title="Multiple testing", question=question, verdict=PASS,
            headline=f"One variant declared, so p={primary_p:.4f} needs no correction.",
            detail=(
                "This is a strong claim and an uncommon one. It holds only if no "
                "parameter was tuned, no lookback compared and no asset swapped in "
                "before this record was produced."
            ),
            numbers=numbers,
        )
    if adjusted_bonf < alpha:
        return Check(
            key="multiplicity", title="Multiple testing", question=question, verdict=PASS,
            headline=(
                f"Survives {n} variants: p={primary_p:.4f} becomes {adjusted_bonf:.4f} "
                "after Bonferroni, still under 0.05."
            ),
            detail=(
                f"Chance alone would hand you about {n * alpha:.1f} significant results "
                f"from {n} tries. This one is stronger than that."
            ),
            numbers=numbers,
        )
    return Check(
        key="multiplicity", title="Multiple testing", question=question, verdict=FAIL,
        headline=(
            f"Does not survive {n} variants: p={primary_p:.4f} becomes {adjusted_bonf:.4f} "
            f"after Bonferroni ({adjusted_bh:.4f} after Benjamini-Hochberg)."
        ),
        detail=(
            f"Testing {n} variants gives chance about {n * alpha:.1f} chances to produce "
            "a result this good. Keeping the best of them and reporting its raw p-value "
            "is the most common way a backtest lies, and it is rarely deliberate."
        ),
        numbers=numbers,
    )


# --------------------------------------------------------------------------
# 5. How wide is the Sharpe, really?
# --------------------------------------------------------------------------
def bootstrap_interval(data, *, periods_per_year: int, seed: int = 20260917) -> Check:
    question = "If this history had unfolded slightly differently, what Sharpe would it show?"
    values = data.net_returns.dropna().to_numpy(dtype=float)
    if values.size < 60:
        return Check(key="bootstrap", title="Sharpe interval", question=question,
                     verdict=NA, headline="Too few observations to resample.")

    rng = np.random.default_rng(seed)
    observed = sharpe(values, periods_per_year)
    draws = np.array([
        sharpe(block_bootstrap(values, block_length=BOOTSTRAP_BLOCK, rng=rng), periods_per_year)
        for _ in range(BOOTSTRAP_DRAWS)
    ])
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return Check(key="bootstrap", title="Sharpe interval", question=question,
                     verdict=NA, headline="The resample produced no usable Sharpe values.")

    low, high = (float(x) for x in np.percentile(draws, [5, 95]))
    share_negative = float((draws <= 0).mean())
    numbers = {
        "sharpe": observed,
        "sharpe_p05": low,
        "sharpe_p95": high,
        "share_of_resamples_negative": share_negative,
        "block_length_days": BOOTSTRAP_BLOCK,
        "draws": int(draws.size),
    }
    detail = (
        "Resampled in blocks of about a month, which keeps volatility clustering and "
        "the persistence of a position intact - an i.i.d. bootstrap would treat days "
        "as independent, which prices are not, and would report an interval far too "
        "narrow. This interval is about how much the ESTIMATE moves, not about "
        "whether the rule works."
    )
    if low > 0:
        return Check(
            key="bootstrap", title="Sharpe interval", question=question, verdict=PASS,
            headline=(
                f"Sharpe {observed:.2f}, 90% interval [{low:.2f}, {high:.2f}] - "
                "above zero throughout."
            ),
            detail=detail, numbers=numbers,
        )
    return Check(
        key="bootstrap", title="Sharpe interval", question=question, verdict=FAIL,
        headline=(
            f"Sharpe {observed:.2f}, 90% interval [{low:.2f}, {high:.2f}] - the interval "
            f"includes zero, and {share_negative:.0%} of resamples come out negative."
        ),
        detail=detail + " A headline Sharpe whose interval crosses zero is not a result.",
        numbers=numbers,
    )


# --------------------------------------------------------------------------
# 6. Does the second half look like the first?
# --------------------------------------------------------------------------
def out_of_sample(data, *, periods_per_year: int, embargo_days: int = 21) -> Check:
    question = "Does the second half of the record resemble the first?"
    net = data.net_returns.dropna()
    if len(net) < 2 * 60 + embargo_days:
        return Check(key="out_of_sample", title="Out of sample", question=question,
                     verdict=NA,
                     headline=(
                         f"Too short to split: {len(net)} observations, and each half "
                         "needs at least 60 with an embargo between them."
                     ))

    midpoint = len(net) // 2
    first = net.iloc[:midpoint]
    second = net.iloc[midpoint + embargo_days:]
    first_sharpe = sharpe(first, periods_per_year)
    second_sharpe = sharpe(second, periods_per_year)
    first_total = float((1 + first).prod() - 1)
    second_total = float((1 + second).prod() - 1)
    numbers = {
        "split_date": net.index[midpoint],
        "embargo_days": embargo_days,
        "first_half_sharpe": first_sharpe,
        "second_half_sharpe": second_sharpe,
        "first_half_return": first_total,
        "second_half_return": second_total,
        "first_half_days": int(len(first)),
        "second_half_days": int(len(second)),
    }
    detail = (
        f"The halves are separated by a {embargo_days}-day embargo so a position held "
        "across the boundary cannot leak the first half's information into the second. "
        "This is the weakest form of out-of-sample testing - the second half was "
        "available while the rule was being built - so passing it is necessary, not "
        "sufficient."
    )
    if not np.isfinite(first_sharpe) or not np.isfinite(second_sharpe):
        return Check(key="out_of_sample", title="Out of sample", question=question,
                     verdict=NA, headline="One half produced no usable Sharpe.",
                     numbers=numbers, detail=detail)

    if second_sharpe <= 0 < first_sharpe:
        return Check(
            key="out_of_sample", title="Out of sample", question=question, verdict=FAIL,
            headline=(
                f"The result is in the first half only: Sharpe {first_sharpe:.2f} then "
                f"{second_sharpe:.2f} ({first_total:+.1%} then {second_total:+.1%})."
            ),
            detail=detail, numbers=numbers,
        )
    if second_sharpe < first_sharpe * 0.5:
        return Check(
            key="out_of_sample", title="Out of sample", question=question, verdict=WARN,
            headline=(
                f"The second half is much weaker: Sharpe {first_sharpe:.2f} then "
                f"{second_sharpe:.2f}. Still positive, but less than half."
            ),
            detail=detail + " Decay of this size is what a fitted rule looks like as it "
                            "meets data it was not shaped on.",
            numbers=numbers,
        )
    return Check(
        key="out_of_sample", title="Out of sample", question=question, verdict=PASS,
        headline=(
            f"Both halves point the same way: Sharpe {first_sharpe:.2f} then "
            f"{second_sharpe:.2f} ({first_total:+.1%} then {second_total:+.1%})."
        ),
        detail=detail, numbers=numbers,
    )


# --------------------------------------------------------------------------
# 7. How much survives trading costs?
# --------------------------------------------------------------------------
def cost_sensitivity(data, *, cost_rate: float, applied_bps: float) -> Check:
    question = "At what level of trading costs does the profit disappear?"
    if data.positions is None or data.asset_returns is None:
        return Check(
            key="costs", title="Cost sensitivity", question=question, verdict=NA,
            headline="Not available: costs cannot be re-run on an equity curve.",
            detail=(
                "This tool cannot tell whether the file already has fees and slippage "
                "in it. If it does not, every number in this report is optimistic by "
                "an amount nobody has measured."
            ),
        )

    positions = data.positions.to_numpy(dtype=float)
    # Log space, locally: the breakeven below is closed form only because log
    # returns add, and `net_log = gross - cost * turnover` is then exact rather
    # than a search. Everything the reader sees is converted back with expm1.
    log_returns = np.nan_to_num(np.log1p(data.asset_returns.to_numpy(dtype=float)))
    gross = float(np.sum(positions[:-1] * log_returns[1:]))
    turnover = float(np.sum(np.abs(np.diff(positions, prepend=positions[0]))[:-1]))

    if turnover <= 0:
        return Check(key="costs", title="Cost sensitivity", question=question, verdict=NA,
                     headline="The position never changes, so there is no turnover to cost.")
    if gross <= 0:
        return Check(
            key="costs", title="Cost sensitivity", question=question, verdict=FAIL,
            headline="The strategy loses money before any cost is charged.",
            numbers={"gross_log_return": gross, "turnover": turnover},
        )

    # net_log = gross - cost_rate * turnover, exactly, so breakeven is closed form.
    breakeven_rate = gross / turnover
    breakeven_bps = breakeven_rate * 10_000.0
    numbers = {
        "applied_bps": applied_bps,
        "breakeven_bps": breakeven_bps,
        "turnover": turnover,
        "gross_log_return": gross,
        "net_at_25bps": float(np.expm1(gross - 0.0025 * turnover)),
        "net_at_50bps": float(np.expm1(gross - 0.0050 * turnover)),
        "net_at_100bps": float(np.expm1(gross - 0.0100 * turnover)),
    }
    detail = (
        "Breakeven is the one-way cost, as a fraction of each unit of turnover, at "
        "which the whole profit is eaten. Retail crypto round-trips commonly land "
        "between 20 and 60 bps one way once spread and slippage are counted, and a "
        "market order in a fast tape can cost several times that."
    )
    if breakeven_bps < 25:
        return Check(
            key="costs", title="Cost sensitivity", question=question, verdict=FAIL,
            headline=(
                f"The profit dies at {breakeven_bps:.0f} bps of one-way cost, which is "
                "below what retail execution actually costs."
            ),
            detail=detail, numbers=numbers,
        )
    if breakeven_bps < 60:
        return Check(
            key="costs", title="Cost sensitivity", question=question, verdict=WARN,
            headline=(
                f"Thin margin: the profit dies at {breakeven_bps:.0f} bps one way, "
                "inside the range retail execution reaches on a bad day."
            ),
            detail=detail, numbers=numbers,
        )
    return Check(
        key="costs", title="Cost sensitivity", question=question, verdict=PASS,
        headline=f"Survives to {breakeven_bps:.0f} bps of one-way cost before the profit is gone.",
        detail=detail, numbers=numbers,
    )


# --------------------------------------------------------------------------
# 8. Did it beat simply holding the thing?
# --------------------------------------------------------------------------
def versus_holding(data, *, periods_per_year: int) -> Check:
    question = "Did it beat buying the asset and doing nothing?"
    net = data.net_returns.dropna()

    if data.benchmark_returns is not None:
        bench = data.benchmark_returns.reindex(net.index).fillna(0.0)
        label = "the benchmark column in the file"
    elif data.asset_returns is not None:
        bench = data.asset_returns.reindex(net.index).fillna(0.0)
        label = "buying and holding the asset in the file"
    else:
        return Check(
            key="benchmark", title="Versus holding", question=question, verdict=NA,
            headline="No benchmark available: add a `close` or `benchmark` column.",
            detail=(
                "A strategy that made 200% in a market that made 600% destroyed value, "
                "and no risk metric computed on the strategy alone will say so."
            ),
        )

    strategy_total = float((1 + net).prod() - 1)
    bench_total = float((1 + bench).prod() - 1)
    strategy_sharpe = sharpe(net, periods_per_year)
    bench_sharpe = sharpe(bench, periods_per_year)
    numbers = {
        "benchmark": label,
        "strategy_return": strategy_total,
        "benchmark_return": bench_total,
        "strategy_sharpe": strategy_sharpe,
        "benchmark_sharpe": bench_sharpe,
        "strategy_max_drawdown": max_drawdown(data.equity),
        "benchmark_max_drawdown": max_drawdown((1 + bench).cumprod()),
    }
    detail = (
        "Two ways to win, and they are different claims: more money (return), or the "
        "same money with a calmer path (Sharpe, drawdown). A rule that is only in the "
        "market half the time should be judged on the second."
    )
    if strategy_total <= bench_total and strategy_sharpe <= bench_sharpe:
        return Check(
            key="benchmark", title="Versus holding", question=question, verdict=FAIL,
            headline=(
                f"Loses on both counts against {label}: {strategy_total:+.1%} vs "
                f"{bench_total:+.1%}, Sharpe {strategy_sharpe:.2f} vs {bench_sharpe:.2f}."
            ),
            detail=detail, numbers=numbers,
        )
    if strategy_total <= bench_total:
        return Check(
            key="benchmark", title="Versus holding", question=question, verdict=WARN,
            headline=(
                f"Less money, smoother ride: {strategy_total:+.1%} vs {bench_total:+.1%}, "
                f"but Sharpe {strategy_sharpe:.2f} vs {bench_sharpe:.2f}."
            ),
            detail=detail + " Whether that is a win depends on whether the smoother path "
                            "lets you hold it, which no backtest can tell you.",
            numbers=numbers,
        )
    return Check(
        key="benchmark", title="Versus holding", question=question, verdict=PASS,
        headline=(
            f"Beats {label}: {strategy_total:+.1%} vs {bench_total:+.1%}, "
            f"Sharpe {strategy_sharpe:.2f} vs {bench_sharpe:.2f}."
        ),
        detail=detail, numbers=numbers,
    )


# --------------------------------------------------------------------------
# 9. How few days carry the whole thing?
# --------------------------------------------------------------------------
def concentration(data) -> Check:
    question = "How much of the profit comes from a handful of days?"
    net = data.net_returns.dropna()
    if len(net) < 60:
        return Check(key="concentration", title="Concentration", question=question,
                     verdict=NA, headline="Too few observations.")

    total = float((1 + net).prod() - 1)
    ordered = net.sort_values(ascending=False)
    top_n = max(1, int(round(len(net) * 0.01)))
    without_top = net.drop(ordered.index[:top_n])
    total_without = float((1 + without_top).prod() - 1)
    best_day = float(ordered.iloc[0])
    numbers = {
        "total_return": total,
        "top_days_removed": top_n,
        "total_return_without_top_days": total_without,
        "best_single_day": best_day,
        "share_of_days_removed": top_n / len(net),
    }
    detail = (
        "Removing the best 1% of days is not a prediction that you would have missed "
        "them. It is a measure of how thin the evidence is: a result carried by five "
        "days out of five hundred is five observations, and the other 495 are noise "
        "around them."
    )
    if total > 0 and total_without <= 0:
        return Check(
            key="concentration", title="Concentration", question=question, verdict=FAIL,
            headline=(
                f"Everything comes from {top_n} day(s). Remove the best 1% and "
                f"{total:+.1%} becomes {total_without:+.1%}."
            ),
            detail=detail, numbers=numbers,
        )
    if total > 0 and total_without < total * 0.35:
        return Check(
            key="concentration", title="Concentration", question=question, verdict=WARN,
            headline=(
                f"Concentrated: the best {top_n} day(s) carry most of it - "
                f"{total:+.1%} falls to {total_without:+.1%} without them."
            ),
            detail=detail, numbers=numbers,
        )
    return Check(
        key="concentration", title="Concentration", question=question, verdict=PASS,
        headline=(
            f"Broadly earned: without the best {top_n} day(s), {total:+.1%} is still "
            f"{total_without:+.1%}."
        ),
        detail=detail, numbers=numbers,
    )


# --------------------------------------------------------------------------
# 10. Could a person actually have held this?
# --------------------------------------------------------------------------
def survivability(data, *, periods_per_year: int) -> Check:
    question = "Could a human being have held this to the end?"
    equity = data.equity
    drawdown = max_drawdown(equity)
    underwater = longest_underwater(equity)
    worst_day = float(data.net_returns.min())
    numbers = {
        "max_drawdown": drawdown,
        "longest_underwater_days": underwater,
        "worst_single_day": worst_day,
        "periods_per_year": periods_per_year,
    }
    detail = (
        "Every number in this report assumes the rule was followed without "
        "interruption. A drawdown deep enough or long enough that the person running "
        "it would have stopped is a backtest of a strategy nobody executed."
    )
    if drawdown <= -0.50 or underwater > 2 * periods_per_year:
        return Check(
            key="survivability", title="Survivability", question=question, verdict=FAIL,
            headline=(
                f"Hard to hold: worst drawdown {drawdown:.0%}, and it spent "
                f"{underwater} days below a previous high."
            ),
            detail=detail, numbers=numbers,
        )
    if drawdown <= -0.30 or underwater > periods_per_year:
        return Check(
            key="survivability", title="Survivability", question=question, verdict=WARN,
            headline=(
                f"Demanding: worst drawdown {drawdown:.0%}, longest stretch under water "
                f"{underwater} days."
            ),
            detail=detail, numbers=numbers,
        )
    return Check(
        key="survivability", title="Survivability", question=question, verdict=PASS,
        headline=(
            f"Tolerable path: worst drawdown {drawdown:.0%}, longest stretch under "
            f"water {underwater} days."
        ),
        detail=detail, numbers=numbers,
    )
