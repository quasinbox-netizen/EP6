"""Sample splits: by cycle, walk-forward, with embargo.

Three traps this module closes:

1. A random split of a time series makes no sense - it would train on the
   future. Every split here is chronological.
2. The target `fwd_return_90d` on day t contains prices from t+90. If t is the
   last training day and t+1 the first test day, the sets overlap. Hence the
   EMBARGO: a gap equal to the target horizon, cut out between the sets.
3. Splitting by halving cycle is natural for this project but has a price:
   there are five cycles, so the test set is one or two of them. "It works on
   cycle 4" is one observation, not proof. That is what walk-forward fixes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from features.halving import CONFIRMED_HALVINGS


@dataclass(frozen=True)
class Split:
    name: str
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    embargo: pd.DatetimeIndex

    def describe(self) -> dict:
        def span(index: pd.DatetimeIndex) -> str:
            if len(index) == 0:
                return "-"
            return f"{index.min().date()} .. {index.max().date()}"

        return {
            "split": self.name,
            "train_days": len(self.train),
            "train_span": span(self.train),
            "test_days": len(self.test),
            "test_span": span(self.test),
            "embargo_days": len(self.embargo),
        }


def cycle_of(dates: pd.DatetimeIndex) -> pd.Series:
    dates = pd.DatetimeIndex(dates)
    values = [int((CONFIRMED_HALVINGS <= day).sum()) for day in dates]
    return pd.Series(values, index=dates, name="cycle_index")


def cycle_split(
    index: pd.DatetimeIndex,
    train_cycles: list[int],
    test_cycles: list[int],
    *,
    embargo_days: int = 0,
) -> Split:
    """Split by halving cycle number, with an embargo between the sets."""
    index = pd.DatetimeIndex(pd.to_datetime(index)).normalize().sort_values()
    cycles = cycle_of(index)
    train = index[cycles.isin(train_cycles).to_numpy()]
    test = index[cycles.isin(test_cycles).to_numpy()]

    embargo = pd.DatetimeIndex([])
    if embargo_days > 0 and len(train) and len(test):
        boundary = train.max()
        embargo = train[train > boundary - pd.Timedelta(days=embargo_days)]
        train = train[train <= boundary - pd.Timedelta(days=embargo_days)]
        # The embargo is cut from the training side; the test set starts past
        # the original boundary, so no test day is lost.
        test = test[test > boundary]

    name = f"cycles {train_cycles} -> {test_cycles}"
    return Split(name=name, train=train, test=test, embargo=embargo)


def walk_forward_splits(
    index: pd.DatetimeIndex,
    *,
    train_days: int = 730,
    test_days: int = 365,
    step_days: int | None = None,
    embargo_days: int = 90,
    expanding: bool = True,
) -> list[Split]:
    """Walk-forward validation: train on the past, test on the next window.

    `expanding=True` grows the training window at every step, which is the
    honest analogue of how you would actually operate. `expanding=False` gives
    a sliding window of fixed length.

    Keep `step_days >= test_days` or consecutive test windows overlap, which
    breaks the independence assumption of any test taken across folds.
    """
    index = pd.DatetimeIndex(pd.to_datetime(index)).normalize().sort_values()
    step = step_days or test_days
    splits: list[Split] = []
    start = 0
    fold = 0

    while True:
        train_end = start + train_days
        test_start = train_end + embargo_days
        test_end = test_start + test_days
        if test_start >= len(index):
            break
        train_slice = index[(0 if expanding else start) : train_end]
        embargo_slice = index[train_end:test_start]
        test_slice = index[test_start : min(test_end, len(index))]
        if len(test_slice) == 0 or len(train_slice) == 0:
            break
        splits.append(
            Split(
                name=f"fold {fold}",
                train=train_slice,
                test=test_slice,
                embargo=embargo_slice,
            )
        )
        fold += 1
        start += step
        if test_end >= len(index):
            break
    return splits


def window_effect(values: pd.Series, mask: pd.Series) -> float:
    """Difference of means: inside the window minus outside. No test, just the effect.

    Walk-forward does not need a per-fold p-value - inference comes from the
    distribution of effects ACROSS folds, not from a single window. Running
    permutations for every fold would be expensive and add nothing.
    """
    frame = pd.DataFrame({"value": values, "mask": mask}).dropna()
    if frame.empty:
        return float("nan")
    inside = frame.loc[frame["mask"].astype(bool), "value"]
    outside = frame.loc[~frame["mask"].astype(bool), "value"]
    if inside.empty or outside.empty:
        return float("nan")
    return float(inside.mean() - outside.mean())


def sign_agreement_test(train_effects, test_effects) -> dict:
    """Does the sign of the effect survive the move from training to test?

    This is the heart of walk-forward. Under the null (no relationship) the
    sign in the test window is a coin flip, so the number of agreeing folds is
    binomial with p=0.5. Two-sided, because consistently flipping the sign is
    also information - and also not chance.

    We additionally report the mean out-of-sample effect with a t-test across
    folds. Test windows are disjoint (embargo), so treating them as independent
    observations is defensible - unlike treating days that way.
    """
    pairs = [
        (a, b)
        for a, b in zip(train_effects, test_effects)
        if np.isfinite(a) and np.isfinite(b)
    ]
    n = len(pairs)
    if n == 0:
        return {
            "n_folds": 0, "n_same_sign": 0, "sign_agreement": np.nan,
            "sign_p_value": np.nan, "mean_test_effect": np.nan,
            "test_effect_t_stat": np.nan, "test_effect_p_value": np.nan,
        }

    same = sum(1 for a, b in pairs if np.sign(a) == np.sign(b) and a != 0)
    sign_p = float(stats.binomtest(same, n, 0.5).pvalue) if n else np.nan

    test_values = np.array([b for _, b in pairs], dtype=float)
    if n > 1 and test_values.std(ddof=1) > 0:
        t_stat = float(test_values.mean() / (test_values.std(ddof=1) / np.sqrt(n)))
        t_p = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    else:
        t_stat = t_p = np.nan

    return {
        "n_folds": n,
        "n_same_sign": same,
        "sign_agreement": same / n,
        "sign_p_value": sign_p,
        "mean_test_effect": float(test_values.mean()),
        "test_effect_t_stat": t_stat,
        "test_effect_p_value": t_p,
    }


def assert_no_overlap(split: Split, *, horizon_days: int = 0) -> None:
    """Check the sets are disjoint and the gap covers the target horizon."""
    overlap = split.train.intersection(split.test)
    if len(overlap):
        raise AssertionError(f"{split.name}: sets overlap ({len(overlap)} days)")
    if len(split.train) == 0 or len(split.test) == 0:
        return
    gap = (split.test.min() - split.train.max()).days
    if gap <= horizon_days:
        raise AssertionError(
            f"{split.name}: the gap of {gap} days does not cover the target horizon "
            f"of {horizon_days} days - the last training days contain prices from "
            "the test set"
        )


def split_frame(frame: pd.DataFrame, split: Split) -> tuple[pd.DataFrame, pd.DataFrame]:
    return frame.loc[frame.index.isin(split.train)], frame.loc[frame.index.isin(split.test)]


def replicate_finding(
    train_result: dict, test_result: dict, *, alpha: float = 0.05
) -> dict:
    """Did a finding from the training set hold up on the test set?

    We require three things at once: the same sign, significance out of sample,
    and that the effect did not shrink to a fraction. Sign alone is far too
    weak - with two possible signs it agrees half the time by chance.
    """
    train_effect = train_result.get("difference", np.nan)
    test_effect = test_result.get("difference", np.nan)
    same_sign = bool(np.sign(train_effect) == np.sign(test_effect)) and np.isfinite(test_effect)
    significant = bool(test_result.get("p_value", 1.0) < alpha)
    retained = (
        float(abs(test_effect) / abs(train_effect)) if train_effect not in (0, np.nan) else np.nan
    )
    return {
        "train_effect": train_effect,
        "test_effect": test_effect,
        "same_sign": same_sign,
        "significant_out_of_sample": significant,
        "effect_retained": retained,
        "replicated": bool(same_sign and significant and np.isfinite(retained) and retained > 0.5),
    }
