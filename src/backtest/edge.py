"""Does a trading rule have an edge, or was it lucky?

The backtest module could compare strategies but not judge them. It reported
Sharpe ratios and differences against buy-and-hold, and left the reader to
decide whether 1.42 against 1.14 means anything. Everywhere else this project
insists a number without an interval is not a result; the one module that
touches money was exempt.

It is not exempt any more. This asks the only question a backtest can answer:

    would the SAME RULE, applied at random times, have done as well?

WHY RE-TIMING RATHER THAN RESHUFFLING RETURNS
---------------------------------------------
The obvious null is to shuffle the price series and re-run. It is the wrong
one, and flatteringly so: shuffling destroys volatility clustering, momentum
and the upward drift all at once, so almost any strategy beats it and every
rule looks skilled.

The null here keeps the price series exactly as it happened and moves the
STRATEGY instead, circularly shifting its position series. That preserves, by
construction:

  * how much of the time the rule is invested,
  * how many round trips it makes, so trading costs are unchanged,
  * the autocorrelation of the position itself - a slow trend follower stays
    slow, a twitchy rule stays twitchy,

and destroys only the alignment between the rule and the price. What is left
is the question worth asking: is this rule's timing better than arbitrary
timing of the same kind?

A rule that is invested 27% of the time and happens to catch bull markets will
beat buy-and-hold on Sharpe whether or not its logic works. Under this null it
only wins if catching them was not luck.

WHAT A GOOD RESULT HERE STILL DOES NOT MEAN
-------------------------------------------
Passing says the rule's timing beat random timing on this history. It does not
say the rule will work, that the future resembles the sample, or that the
result survives having tried many rules - if several are tested, the p-values
need the same Benjamini-Hochberg correction as everything else here, and
`validation.multiple_testing` is where that lives.

THE TRAP THIS TEST SETS, AND WHY `fragility` EXISTS
---------------------------------------------------
A permutation p-value looks precise no matter how few events produced it. Run
on the halving rule, this test returns p=0.0095 - while the event study on the
same four halvings reports p=0.30 with an interval from -190% to +441%. Both
are correct, and the difference is the whole lesson.

The event study's unit of observation is the EVENT, so four events give it four
observations and an interval wide enough to say so. This test's null is about
PLACEMENT, so 2000 draws give a p-value to three decimals - from the same four
events. The draws multiply the arrangements, not the evidence.

Leave-one-out on that result: removing the 2012 halving takes p from 0.0095 to
0.1369. BTC rose roughly a hundredfold in the year after it, and that single
window carries the entire finding.

So `edge_test` is never the last word. `fragility` refits the test with each
episode of the rule removed in turn, and a result that one omission destroys
is reported as fragile no matter what its p-value says. A precise number built
on one observation is the exact failure this project exists to catch, and this
module was capable of producing it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Enough for a p-value to be readable to about a thousandth, and cheap: one
# draw is a vector multiply, not a refit.
DEFAULT_PERMUTATIONS = 2000


@dataclass
class EdgeResult:
    statistic: str
    observed: float
    null_mean: float
    null_p95: float
    p_value: float
    n_permutations: int
    exposure: float
    n_trades: int

    def summary(self) -> str:
        return (
            f"{self.statistic}: observed {self.observed:.3f} vs {self.null_mean:.3f} "
            f"from random timing (p95 {self.null_p95:.3f}) | p={self.p_value:.4f} "
            f"| {self.exposure:.0%} invested, {self.n_trades} round trips"
        )

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value < 0.05)


def _sharpe(returns: np.ndarray, periods_per_year: int = 365) -> float:
    if returns.size < 2:
        return float("nan")
    deviation = returns.std(ddof=1)
    if deviation == 0:
        return float("nan")
    return float(returns.mean() / deviation * np.sqrt(periods_per_year))


def strategy_returns(
    positions: np.ndarray, asset_returns: np.ndarray, cost_rate: float = 0.0
) -> np.ndarray:
    """Returns of holding `positions` through `asset_returns`, net of turnover cost.

    The position is applied to the NEXT day's return. Multiplying a position by
    the same day's return is the classic backtest look-ahead: it credits the
    rule with a move it used to decide.
    """
    held = positions[:-1]
    realised = asset_returns[1:]
    gross = held * realised
    if cost_rate:
        turnover = np.abs(np.diff(positions, prepend=positions[0]))[:-1]
        return gross - turnover * cost_rate
    return gross


def count_round_trips(positions: np.ndarray) -> int:
    changes = np.abs(np.diff(positions, prepend=0.0))
    return int(np.ceil(changes.sum() / 2.0))


def edge_test(
    positions: pd.Series | np.ndarray,
    asset_returns: pd.Series | np.ndarray,
    *,
    cost_rate: float = 0.0,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 20260905,
    statistic: str = "sharpe",
    periods_per_year: int = 365,
) -> EdgeResult:
    """Compare the rule's timing against random timings of the same rule.

    One-sided: only "better than random" is evidence of an edge. A rule that is
    reliably WORSE than random timing is interesting too, but it is a different
    claim and inverting it after the fact would be reading the sign off the
    data.
    """
    positions = np.asarray(pd.Series(positions).astype(float).to_numpy())
    asset_returns = np.asarray(pd.Series(asset_returns).astype(float).to_numpy())
    if positions.size != asset_returns.size:
        raise ValueError(
            f"positions and returns must align: {positions.size} vs {asset_returns.size}"
        )
    if positions.size < 50:
        raise ValueError(f"need at least 50 observations, got {positions.size}")

    finite = np.isfinite(asset_returns)
    asset_returns = np.where(finite, asset_returns, 0.0)

    def score(series: np.ndarray) -> float:
        realised = strategy_returns(series, asset_returns, cost_rate)
        if statistic == "sharpe":
            return _sharpe(realised, periods_per_year)
        if statistic == "total_return":
            return float(np.expm1(np.nansum(realised)))
        raise ValueError(f"unknown statistic {statistic!r}")

    observed = score(positions)

    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, positions.size, size=n_permutations)
    null = np.array([score(np.roll(positions, int(shift))) for shift in shifts])
    null = null[np.isfinite(null)]

    if null.size == 0 or not np.isfinite(observed):
        return EdgeResult(statistic, observed, np.nan, np.nan, np.nan, 0,
                          float(np.mean(np.abs(positions) > 0)), count_round_trips(positions))

    # +1 top and bottom: the observed arrangement is one the null can produce,
    # and leaving it out allows p=0, which is never true.
    p_value = (1 + np.sum(null >= observed)) / (1 + null.size)
    return EdgeResult(
        statistic=statistic,
        observed=float(observed),
        null_mean=float(null.mean()),
        null_p95=float(np.percentile(null, 95)),
        p_value=float(p_value),
        n_permutations=int(null.size),
        exposure=float(np.mean(np.abs(positions) > 0)),
        n_trades=count_round_trips(positions),
    )


def episodes(positions: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous stretches where the rule holds a position.

    The natural unit of evidence for a trading rule: one entry, one exit, one
    thing that either worked or did not. Counting days instead would say a
    rule invested for four years has 1460 observations, when it has as many as
    it made decisions.
    """
    active = np.abs(np.asarray(positions, dtype=float)) > 0
    if not active.any():
        return []
    edges = np.diff(active.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


@dataclass
class FragilityResult:
    n_episodes: int
    full_p_value: float
    worst_p_value: float
    worst_episode: int
    per_episode: pd.DataFrame

    @property
    def fragile(self) -> bool:
        """Does removing one episode take the result out of significance?"""
        return bool(
            np.isfinite(self.full_p_value)
            and self.full_p_value < 0.05 <= self.worst_p_value
        )


def fragility(
    positions: pd.Series | np.ndarray,
    asset_returns: pd.Series | np.ndarray,
    **kwargs,
) -> FragilityResult:
    """Re-run the edge test with each episode removed in turn.

    Answers the question a p-value cannot: how much of this rests on one
    trade? A rule whose significance disappears when a single episode is
    dropped has one observation supporting it, however many decimal places the
    p-value carries.
    """
    positions = pd.Series(positions).astype(float)
    values = positions.to_numpy()
    blocks = episodes(values)
    full = edge_test(positions, asset_returns, **kwargs)

    rows = []
    for index, (start, end) in enumerate(blocks):
        without = values.copy()
        without[start:end] = 0.0
        if not (np.abs(without) > 0).any():
            continue  # nothing left to test
        result = edge_test(pd.Series(without, index=positions.index), asset_returns, **kwargs)
        rows.append({
            "episode": index,
            "start": positions.index[start] if hasattr(positions.index, "__getitem__") else start,
            "days": end - start,
            "p_value": result.p_value,
            "observed": result.observed,
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return FragilityResult(len(blocks), full.p_value, full.p_value, -1, table)
    worst = table.loc[table["p_value"].idxmax()]
    return FragilityResult(
        n_episodes=len(blocks),
        full_p_value=full.p_value,
        worst_p_value=float(worst["p_value"]),
        worst_episode=int(worst["episode"]),
        per_episode=table,
    )


def verdict(result: EdgeResult) -> str:
    """One sentence, phrased so it cannot be quoted as more than it is."""
    if not np.isfinite(result.p_value):
        return "INCONCLUSIVE - the permutation test did not run."
    if result.significant:
        return (
            f"TIMING BEATS CHANCE (p={result.p_value:.4f}) - randomly re-timing "
            "this rule reproduces its result less than 5% of the time. Read the "
            "fragility line before believing it: this p-value is as precise as "
            "the number of draws and as strong as the number of episodes, and "
            "those are not the same thing. It also does not survive being one "
            "of many rules tried unless corrected for how many."
        )
    return (
        f"NO EDGE (p={result.p_value:.4f}) - the same rule applied at random "
        f"times scores {result.null_mean:.3f} on average against {result.observed:.3f} "
        "observed. Whatever this rule earned, its timing is not the reason."
    )
