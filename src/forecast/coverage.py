"""Does the interval contain the outcome as often as it claims?

An interval is a testable promise. "90% confidence" means the price lands
inside nine times in ten, and if it does not, the number is decoration. This
module walks the model forward through history, records every hit and miss, and
tests the hit rate against the nominal level.

The verdict gates the tool. `run.py range` refuses to quote an interval whose
coverage failed here, because a badly calibrated interval is worse than none:
it looks like knowledge.

TWO TRAPS, BOTH ALREADY PAID FOR ELSEWHERE IN THIS PROJECT
----------------------------------------------------------
1. OVERLAPPING WINDOWS. The 10-day outcome starting today and the one starting
   tomorrow share nine days. Treating consecutive checks as independent
   observations inflates the sample roughly h-fold and makes any binomial test
   meaningless - the same error the forecast module fixed by thinning 4644
   daily predictions to 155 non-overlapping ones. Coverage is therefore tested
   on every h-th check only, and the count of independent observations is what
   gets reported.

2. LOOK-AHEAD IN THE FIT. The model that produces the interval for day t must
   be fitted on data ending at t. Refitting on the whole sample and then
   "checking" history is a way of proving that a model fits data it has already
   seen. The walk here refits periodically on a trailing window and never lets
   the fit see past its own forecast origin.

WHAT THIS TEST STILL CANNOT SEE
-------------------------------
Thinning removes the overlap but not the dependence. Volatility clusters, so a
violent quarter contains several consecutive non-overlapping windows that all
miss together, and a calm one contains several that all hit. The binomial test
treats them as independent coin flips, which they are not quite, so its p-value
is slightly optimistic - a real miscalibration confined to one regime is harder
to detect than the number suggests.

The honest reading is therefore asymmetric. NOT CALIBRATED is strong evidence:
the test had to overcome its own optimism to say it. CALIBRATED means "no
detectable failure", not "correct". Nothing here fixes that; a block bootstrap
over volatility regimes would, and would need more independent windows than 15
years of daily data provides at a 10-day horizon.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from forecast.volatility import fit_garch, simulate_horizon, update_state

# Refitting daily would be 4000 fits for one report and changes almost nothing:
# GARCH parameters move slowly. The variance recursion is still updated every
# day between refits, so today's sigma is always current - only the parameters
# are stale, by at most this many days.
DEFAULT_REFIT_EVERY = 30

# A trailing window rather than everything since 2011. Fifteen years of BTC
# spans volatility regimes with nothing in common, and a single fit over all of
# them is pushed to near-unit-root persistence. Four years is long enough to
# identify the parameters and short enough to stay in one regime.
DEFAULT_WINDOW = 1460


@dataclass
class CoverageResult:
    table: pd.DataFrame
    horizon: int
    n_independent: int
    n_checks: int

    @property
    def calibrated(self) -> bool:
        return bool(self.table["within_tolerance"].all())

    def summary(self) -> str:
        if self.table.empty:
            return "no coverage data"
        worst = self.table.loc[self.table["p_value"].idxmin()]
        state = "CALIBRATED" if self.calibrated else "NOT CALIBRATED"
        return (
            f"{state} | horizon {self.horizon}d | {self.n_independent} independent "
            f"windows | worst level {worst['level']:.0%}: "
            f"{worst['observed']:.1%} covered (p={worst['p_value']:.3f})"
        )


def rolling_intervals(
    prices: pd.Series,
    horizon: int,
    *,
    levels=(0.5, 0.68, 0.90, 0.95),
    window: int = DEFAULT_WINDOW,
    refit_every: int = DEFAULT_REFIT_EVERY,
    n_paths: int = 4000,
    seed: int = 20260905,
    drift: str | float = "fitted",
) -> pd.DataFrame:
    """Walk forward, refitting periodically, and record every interval and outcome."""
    prices = pd.Series(prices).dropna().sort_index()
    log_price = np.log(prices.astype(float))
    returns = log_price.diff().dropna()
    if len(returns) < window + horizon + 10:
        raise ValueError(
            f"need at least {window + horizon + 10} returns for a walk with "
            f"window={window}, got {len(returns)}"
        )

    rows = []
    parameters = None
    last_refit = None
    # The last usable origin is the one whose outcome still exists: its price
    # position is `position + 1`, and that plus `horizon` must stay inside
    # log_price, whose length is len(returns) + 1.
    for position in range(window, len(returns) - horizon):
        if (position - window) % refit_every == 0 or parameters is None:
            parameters = fit_garch(returns.iloc[position - window : position])
            last_refit = position

        # Parameters age slowly and are refitted rarely; the conditional
        # variance ages every day and is rolled forward here. Reusing the fit
        # unchanged - which is what this loop did at first - leaves the
        # interval built on a variance up to `refit_every` days stale, which is
        # worst precisely when a volatility regime is changing.
        fit = update_state(parameters, returns.iloc[last_refit:position])

        origin = returns.index[position]
        # `returns` is one shorter than `log_price` - the first day has no
        # return - so returns.index[i] is log_price.index[i + 1]. Mixing the two
        # index spaces measures an (h-1)-day move and calls it h days, which
        # inflates coverage because a shorter move is easier to contain. The
        # origin's position in price space is therefore made explicit.
        price_position = position + 1
        assert log_price.index[price_position] == origin
        realised = float(
            log_price.iloc[price_position + horizon] - log_price.iloc[price_position]
        )
        # "fitted" takes the drift from the same trailing window as the
        # variance, so no value from beyond the forecast origin enters it.
        step_drift = fit.mean_return if drift == "fitted" else float(drift)
        draws = simulate_horizon(
            fit, horizon, n_paths=n_paths, seed=seed + position, drift=step_drift
        )
        row = {"date": origin, "realised": realised, "position": position}
        for level in levels:
            tail = (1.0 - level) / 2.0
            low, high = np.quantile(draws, [tail, 1.0 - tail])
            row[f"low_{level}"] = low
            row[f"high_{level}"] = high
            row[f"hit_{level}"] = bool(low <= realised <= high)
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


def thin_to_independent(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Keep every h-th row, so no two outcomes share a day.

    Consecutive h-day windows overlap in h-1 days. Counting them all as
    observations is the mistake that turns a coin flip into a certainty.
    """
    if frame.empty:
        return frame
    return frame.iloc[::horizon]


def coverage_report(
    frame: pd.DataFrame,
    horizon: int,
    *,
    levels=(0.5, 0.68, 0.90, 0.95),
    alpha: float = 0.05,
) -> CoverageResult:
    """Test the hit rate at each level against what it promised."""
    independent = thin_to_independent(frame, horizon)
    rows = []
    for level in levels:
        column = f"hit_{level}"
        if column not in independent.columns:
            continue
        hits = int(independent[column].sum())
        n = int(len(independent))
        observed = hits / n if n else np.nan
        # Two-sided: an interval that covers 99% when it promised 90% is also
        # miscalibrated - it is uselessly wide, and saying so is honest.
        p_value = float(stats.binomtest(hits, n, level).pvalue) if n else np.nan
        rows.append({
            "level": level,
            "observed": observed,
            "hits": hits,
            "n": n,
            "p_value": p_value,
            "within_tolerance": bool(np.isfinite(p_value) and p_value >= alpha),
        })
    table = pd.DataFrame(rows)
    return CoverageResult(
        table=table,
        horizon=horizon,
        n_independent=int(len(independent)),
        n_checks=int(len(frame)),
    )


def verdict(result: CoverageResult) -> str:
    """One sentence, phrased so it cannot be quoted as more than it is."""
    if result.table.empty:
        return "INCONCLUSIVE - no coverage data."
    if result.calibrated:
        return (
            f"CALIBRATED - across {result.n_independent} non-overlapping "
            f"{result.horizon}-day windows every level covered the outcome as "
            "often as it promised. The interval is a usable statement about "
            "how far the price moves. It says nothing about which way."
        )
    failed = result.table[~result.table["within_tolerance"]]
    detail = ", ".join(
        f"{row['level']:.0%} covered {row['observed']:.1%}" for _, row in failed.iterrows()
    )
    return (
        f"NOT CALIBRATED - {detail}. The interval does not keep its promise, so "
        "no number derived from it should be quoted."
    )
