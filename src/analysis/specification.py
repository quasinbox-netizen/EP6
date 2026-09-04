"""Specification curve: vary the analytical choices, not the hypothesis.

The hypothesis scan in `validation` asks WHAT is measured - which window, which
event category, which macro phase. It holds four decisions fixed the whole
time:

    estimation window   -250..-31 days
    abnormal return     yes, against that window
    return type         logarithmic
    price series        stitched across three exchanges

Every one of the 26 hypotheses inherits all four. If any is wrong, all 26
results are wrong together, and nothing else in this project would notice. This
module varies those four (and the horizon) across a grid and reports the whole
distribution of answers instead of the one produced by the defaults.

The null result this project reports attracts exactly one objection - "you
picked the wrong window" - and this is the answer to it.

WHY THE COUNT OF SIGNIFICANT SPECIFICATIONS IS NOT A TEST
---------------------------------------------------------
It is tempting to say: 160 specifications, 5% significant by chance, so ~8
hits means nothing. That reasoning is wrong, and wrong in the direction this
project exists to catch.

The 160 specifications are re-analyses of the same four halvings. They are
almost perfectly correlated - changing the estimation window from -250 to -120
does not produce an independent experiment. Under the null the count is
therefore not binomial with p=0.05; it is closer to all-or-nothing, because the
specifications largely rise and fall together. A curve with 40 hits is not
"3x more than chance". It may be one lucky sample seen 240 times.

So inference comes from a permutation test over the ENTIRE curve (Simonsohn,
Simmons & Nelson). The four event dates are shifted together to a random place
in the price history and the whole curve is recomputed. Repeating that builds
the null distribution of curve-level statistics - the median effect and the
number of significant specifications - against which the observed curve is
compared. That accounts for the correlation because the null curves carry
exactly the same correlation.

Shifting the dates TOGETHER matters. Halvings sit about four years apart in a
market that trended up over the sample; scattering four independent random
dates would destroy that structure and produce an easy null to beat. The shift
is circular, and because the halvings span 4161 days of a 5496-day index, about
three quarters of the draws wrap - keeping the gaps circularly rather than on
the calendar. `shifted_dates` explains why that is accepted rather than fixed,
and `permutation_test` reports the wrapped share so it is visible in the
output instead of buried here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
import pandas as pd

from analysis.event_study import event_study, log_returns

# The estimation windows are not arbitrary. -250..-31 is the project default
# (roughly a year, ending a month before the event). -120..-31 is a shorter
# regime-local baseline. -90..-11 ends closer to the event, which is the more
# aggressive choice: it lets pre-event drift into the baseline and so subtracts
# part of any anticipation effect.
ESTIMATION_WINDOWS: tuple[tuple[int, int], ...] = ((-250, -31), (-120, -31), (-90, -11))
HORIZONS: tuple[int, ...] = (7, 30, 90, 180, 365)
RETURN_TYPES: tuple[str, ...] = ("log", "simple")
ABNORMAL: tuple[bool, ...] = (True, False)

ALPHA = 0.05


@dataclass(frozen=True)
class Specification:
    price_series: str
    horizon: int
    estimation_window: tuple[int, int]
    abnormal: bool
    return_type: str

    def label(self) -> str:
        window = "none" if not self.abnormal else f"{self.estimation_window[0]}..{self.estimation_window[1]}"
        return (
            f"{self.price_series}/{self.horizon}d/{self.return_type}/"
            f"{'abn' if self.abnormal else 'raw'}/{window}"
        )


def build_grid(price_series: list[str]) -> list[Specification]:
    """Every combination of the analytical choices.

    When `abnormal` is False the estimation window has no effect, so those
    combinations would be duplicates. They are collapsed to one - counting the
    same specification three times would inflate the denominator and make the
    share of significant results look smaller than it is.
    """
    grid: list[Specification] = []
    for series, horizon, return_type, abnormal in product(
        price_series, HORIZONS, RETURN_TYPES, ABNORMAL
    ):
        windows = ESTIMATION_WINDOWS if abnormal else (ESTIMATION_WINDOWS[0],)
        for window in windows:
            grid.append(
                Specification(
                    price_series=series,
                    horizon=horizon,
                    estimation_window=window,
                    abnormal=abnormal,
                    return_type=return_type,
                )
            )
    return grid


def returns_cache(price_frames: dict[str, pd.DataFrame]) -> dict[tuple[str, str], pd.Series]:
    """Return series for every (price series, return type) pair, computed once.

    There are eight of these and 160 specifications, so without the cache each
    series is rebuilt twenty times per curve - and the curve itself runs 201
    times under permutation.
    """
    return {
        (name, return_type): log_returns(frame, return_type=return_type)
        for name, frame in price_frames.items()
        for return_type in RETURN_TYPES
    }


def run_specification(
    prices: pd.DataFrame,
    event_dates,
    spec: Specification,
    *,
    returns: pd.Series | None = None,
) -> dict:
    """One specification. Returns the CAR at its horizon with the t-interval."""
    result = event_study(
        prices,
        event_dates,
        pre=30,
        post=spec.horizon,
        abnormal=spec.abnormal,
        estimation_window=spec.estimation_window,
        return_type=spec.return_type,
        n_boot=0,
        returns=returns,
    )
    row = {**asdict(spec), "label": spec.label(), "n_events": result.n_events}
    # A single event yields a CAR but no interval and no p-value, so it is not
    # a specification result - it is one observation. Returning NaN keeps it out
    # of the curve entirely. Letting its CAR through would move the median while
    # never being able to count as significant, which biases the curve toward
    # whatever the shortest exchange history happened to catch. Under
    # permutation this matters: a shifted block can leave Binance with one
    # halving in range.
    if result.table.empty or spec.horizon not in result.table.index or result.n_events < 2:
        return {**row, "car": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_value": np.nan, "significant": False}
    line = result.table.loc[spec.horizon]
    p_value = float(line["car_p_value"])
    return {
        **row,
        "car": float(line["car"]),
        "ci_low": float(line["car_ci_low"]),
        "ci_high": float(line["car_ci_high"]),
        "p_value": p_value,
        "significant": bool(np.isfinite(p_value) and p_value < ALPHA),
    }


def run_curve(
    price_frames: dict[str, pd.DataFrame], event_dates, *, grid=None, cache=None
) -> pd.DataFrame:
    """Run every specification and return one row each, sorted by effect size."""
    grid = grid or build_grid(sorted(price_frames))
    cache = returns_cache(price_frames) if cache is None else cache
    rows = [
        run_specification(
            price_frames[spec.price_series],
            event_dates,
            spec,
            returns=cache.get((spec.price_series, spec.return_type)),
        )
        for spec in grid
        if spec.price_series in price_frames
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("car").reset_index(drop=True)


def curve_statistics(curve: pd.DataFrame) -> dict:
    """The two numbers the permutation test is run on, plus description."""
    usable = curve[np.isfinite(curve["car"])]
    if usable.empty:
        return {"n_specs": 0, "median_car": np.nan, "n_significant": 0,
                "share_significant": np.nan, "share_positive": np.nan,
                "n_events_min": 0, "n_events_max": 0}
    return {
        "n_specs": int(len(usable)),
        "median_car": float(usable["car"].median()),
        "n_significant": int(usable["significant"].sum()),
        "share_significant": float(usable["significant"].mean()),
        "share_positive": float((usable["car"] > 0).mean()),
        "n_events_min": int(usable["n_events"].min()),
        "n_events_max": int(usable["n_events"].max()),
    }


def shifted_dates(
    event_dates: pd.DatetimeIndex, index: pd.DatetimeIndex, shift: int
) -> pd.DatetimeIndex:
    """Move the events together along the price index, wrapping around.

    This is a CIRCULAR shift, and the distinction matters more here than it
    usually does. The four halvings span 4161 days inside an index of 5496, so
    only 1335 shifts - 24% of them - move the block without running off the
    end. The other 76% wrap, and a wrapped draw preserves the gaps only
    circularly: some events land at the start of the history while the rest sit
    at the end, so the plain calendar spacing between them is not preserved.

    That is a property of the method, not an accident to be papered over. It is
    accepted for the same reason the rest of this project accepts circular
    shifts: the alternative - scattering four independent random dates - would
    destroy the block structure in EVERY draw rather than in three quarters of
    them, and would produce a null far too easy to beat. Restricting to the
    1335 non-wrapping shifts is the other option, and it is worse: every null
    draw would then overlap heavily with the real halving positions, so the
    null would be built largely out of the arrangement it is meant to test
    against.

    `permutation_test` reports how many draws wrapped, so the reader can see
    what the null was actually made of instead of taking this docstring's word.
    """
    positions = np.array([index.searchsorted(day) for day in event_dates])
    return pd.DatetimeIndex(index[(positions + shift) % len(index)]).sort_values()


def permutation_test(
    price_frames: dict[str, pd.DataFrame],
    event_dates,
    observed: dict,
    *,
    n_permutations: int = 200,
    seed: int = 20260904,
    grid=None,
) -> dict:
    """Null distribution of the curve, by shifting the events as a block.

    Two-sided on the median effect (a consistently negative curve is as much a
    finding as a positive one) and one-sided on the count of significant
    specifications, where only "more than the null produces" is evidence.
    """
    # The shift runs along the LONGEST history, not whichever frame happens to
    # come first in the dict. Taking a short one - Binance starts in 2017 - is
    # silently wrong twice over: the real events would fall outside it, so
    # searchsorted returns len(index) and the modulo places them somewhere
    # unrelated, and the null would explore a fraction of the history the
    # observed curve was measured on. Nothing would raise; the p-value would
    # just be against the wrong null.
    reference = max(price_frames.values(), key=len)
    index = pd.DatetimeIndex(
        pd.DatetimeIndex(pd.to_datetime(reference["date"])).normalize().unique()
    ).sort_values()
    event_dates = pd.DatetimeIndex(pd.to_datetime(event_dates)).normalize().sort_values()
    grid = grid or build_grid(sorted(price_frames))

    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, len(index), size=n_permutations)
    cache = returns_cache(price_frames)

    # How many draws wrapped past the end of the index. A wrapped draw keeps the
    # gaps between events only circularly, so this is the honest measure of how
    # much of the null preserved the calendar spacing. See `shifted_dates`.
    positions = np.array([index.searchsorted(day) for day in event_dates])
    n_wrapped = int(sum((positions + int(s) >= len(index)).any() for s in shifts))

    null_median, null_significant = [], []
    for shift in shifts:
        placebo = shifted_dates(event_dates, index, int(shift))
        stats = curve_statistics(run_curve(price_frames, placebo, grid=grid, cache=cache))
        if stats["n_specs"] == 0:
            continue
        null_median.append(stats["median_car"])
        null_significant.append(stats["n_significant"])

    null_median = np.array(null_median, dtype=float)
    null_significant = np.array(null_significant, dtype=float)
    if null_median.size == 0:
        return {"n_permutations": 0, "median_p_value": np.nan, "significant_count_p_value": np.nan}

    # +1 in numerator and denominator: the observed curve is one of the possible
    # arrangements, and omitting it can produce p=0, which is never true.
    median_p = (1 + np.sum(np.abs(null_median) >= abs(observed["median_car"]))) / (1 + null_median.size)
    count_p = (1 + np.sum(null_significant >= observed["n_significant"])) / (1 + null_significant.size)
    return {
        "n_permutations": int(null_median.size),
        "n_wrapped": n_wrapped,
        "share_wrapped": float(n_wrapped / len(shifts)) if len(shifts) else np.nan,
        "median_p_value": float(median_p),
        "significant_count_p_value": float(count_p),
        "null_median_car_p50": float(np.median(null_median)),
        "null_median_car_p95": float(np.percentile(np.abs(null_median), 95)),
        "null_significant_mean": float(null_significant.mean()),
        "null_significant_p95": float(np.percentile(null_significant, 95)),
    }


def verdict(observed: dict, permutation: dict) -> str:
    """One sentence, stated so it cannot be quoted as more than it is."""
    median_p = permutation.get("median_p_value", np.nan)
    count_p = permutation.get("significant_count_p_value", np.nan)
    if not np.isfinite(median_p) or not np.isfinite(count_p):
        return "INCONCLUSIVE - the permutation test did not run."
    if median_p < ALPHA or count_p < ALPHA:
        return (
            f"ROBUST - the curve is stronger than random placement of the same "
            f"four dates (median p={median_p:.3f}, count p={count_p:.3f}). "
            "Report which specifications drive it before believing it."
        )
    return (
        f"NOT ROBUST - placing the same four dates at random in the price "
        f"history reproduces this curve (median p={median_p:.3f}, "
        f"count p={count_p:.3f}). The result does not depend on the "
        "specification because there is no result to depend on it."
    )
