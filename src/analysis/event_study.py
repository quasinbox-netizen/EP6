"""Event study - a description of how price behaved around events, not a signal.

This module answers one question: how did the return in the N days after an
event compare with the rest of the sample - and it always reports a confidence
interval and the number of events. With four halvings, no result may be quoted
as a number without its interval; four observations are four observations.

Three methodological decisions that separate this from a naive average:

1. The "abnormal" return is measured against an estimation window BEFORE the
   event (default -250..-31 days), not against the full-sample mean. The
   full-sample mean already contains whatever the event is supposed to explain.
2. The unit of observation is the EVENT, not the day. The uncertainty comes
   from having had 4 halvings, not from having had 120 days. Inference
   therefore rests on the spread of CAR BETWEEN events (t distribution with
   n-1 degrees of freedom). At n=4 the critical value is 3.18, not 1.96 - as
   it should be. The percentile bootstrap is reported alongside, but at n<10 it
   is far too narrow (measured in tests/test_phase3: it produced ~30% false
   discoveries instead of 5%), so it is not used for inference.
3. The "window vs rest of sample" test uses permutation by circular shift of
   the mask. A plain t-test on daily returns assumes independence, which prices
   do not have, and systematically overstates significance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_ESTIMATION_WINDOW = (-250, -31)


@dataclass
class EventStudyResult:
    """The result: a table indexed by offset plus metadata."""

    table: pd.DataFrame
    n_events: int
    used_events: pd.DatetimeIndex
    skipped_events: pd.DatetimeIndex
    abnormal: bool
    estimation_window: tuple[int, int] | None
    car_summary: dict = field(default_factory=dict)

    def summary(self) -> str:
        car = self.car_summary
        if not car:
            return f"n={self.n_events}, no CAR summary"
        return (
            f"n={self.n_events} | CAR({car['offset']}d) = {car['car']:+.1%} "
            f"[{car['ci_low']:+.1%}, {car['ci_high']:+.1%}] "
            f"| p={car['p_value']:.3f}"
        )


def log_returns(
    prices: pd.DataFrame,
    price_column: str = "close",
    *,
    return_type: str = "log",
) -> pd.Series:
    """Daily returns. Log by default; `return_type="simple"` gives arithmetic.

    The choice is usually invisible and usually harmless - over a day the two
    differ in the third decimal. It stops being harmless once returns are
    summed: cumulating log returns is exact, cumulating simple ones is not, and
    across a 365-day window on an asset this volatile the gap is large. Both
    are defensible and published work uses both, which is precisely why the
    specification curve varies it instead of this module choosing for you.
    """
    frame = prices.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame = frame.drop_duplicates(subset="date", keep="last").set_index("date")
    values = frame[price_column].astype(float)
    if return_type == "log":
        series = np.log(values).diff()
    elif return_type == "simple":
        series = values.pct_change()
    else:
        raise ValueError(f"unknown return_type {return_type!r}; use 'log' or 'simple'")
    series.name = f"{return_type}_return"
    series = series.sort_index()
    # The first day has no return - diff() and pct_change() both leave NaN
    # there. Keeping it is not harmless: an event window that reaches the start
    # of the series then contains a NaN, and the CAR is built with nancumsum,
    # which silently treats it as a zero return rather than as missing. Only
    # LEADING NaNs are dropped. An internal gap must stay, because the window
    # matrix is positional and dropping a middle row would quietly slide the
    # window across the gap instead of reporting it.
    first_valid = series.first_valid_index()
    return series if first_valid is None else series.loc[first_valid:]


def event_window_matrix(
    returns: pd.Series,
    event_dates,
    *,
    pre: int = 30,
    post: int = 90,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Matrix of returns: one row per event, one column per day offset.

    Events without a complete window are skipped and reported separately -
    silently truncating a window turns a study of 4 events into a study of 3.5.
    """
    returns = returns.sort_index()
    index = returns.index
    offsets = np.arange(-pre, post + 1)
    rows, used, skipped = [], [], []

    for event in pd.DatetimeIndex(pd.to_datetime(event_dates)).normalize():
        position = index.searchsorted(event)
        if position >= len(index) or index[position] != event:
            skipped.append(event)
            continue
        start, stop = position - pre, position + post
        if start < 0 or stop >= len(index):
            skipped.append(event)
            continue
        rows.append(returns.iloc[start : stop + 1].to_numpy())
        used.append(event)

    matrix = pd.DataFrame(rows, index=pd.DatetimeIndex(used), columns=offsets)
    return matrix, pd.DatetimeIndex(skipped)


def _estimation_means(
    returns: pd.Series, event_dates: pd.DatetimeIndex, window: tuple[int, int]
) -> pd.Series:
    """Mean daily return in the estimation window preceding each event."""
    index = returns.sort_index().index
    values = []
    for event in event_dates:
        position = index.searchsorted(event)
        start = position + window[0]
        stop = position + window[1]
        if start < 0:
            values.append(np.nan)
            continue
        values.append(float(np.nanmean(returns.iloc[max(start, 0) : stop + 1].to_numpy())))
    return pd.Series(values, index=event_dates, name="estimation_mean")


def event_study(
    prices: pd.DataFrame,
    event_dates,
    *,
    pre: int = 30,
    post: int = 90,
    abnormal: bool = True,
    estimation_window: tuple[int, int] = DEFAULT_ESTIMATION_WINDOW,
    n_boot: int = 5000,
    seed: int = 20260901,
    price_column: str = "close",
    return_type: str = "log",
    returns: pd.Series | None = None,
) -> EventStudyResult:
    """Mean return path around an event, with a confidence interval.

    Returns a table indexed by day offset: mean daily return, cumulative
    (CAR), the CAR confidence interval, and the number of events.
    """
    # `returns` lets a caller supply the series it already computed. The
    # specification curve calls this function tens of thousands of times over
    # eight distinct return series, and recomputing them each time was 45% of
    # the runtime. When it is given, `prices`, `price_column` and `return_type`
    # are unused - the caller owns that choice and must label the result with
    # the choice it actually made.
    if returns is None:
        returns = log_returns(prices, price_column, return_type=return_type)
    matrix, skipped = event_window_matrix(returns, event_dates, pre=pre, post=post)
    if matrix.empty:
        return EventStudyResult(
            table=pd.DataFrame(),
            n_events=0,
            used_events=pd.DatetimeIndex([]),
            skipped_events=skipped,
            abnormal=abnormal,
            estimation_window=estimation_window if abnormal else None,
        )

    values = matrix.to_numpy(dtype=float)
    if abnormal:
        baseline = _estimation_means(returns, matrix.index, estimation_window)
        # An event whose estimation window reaches back past the start of the
        # series has no baseline. Subtracting NaN makes that event's whole row
        # NaN, and the row would then survive into the results as a FLAT ZERO
        # event: nancumsum treats missing as no-change, so the CAR is pulled
        # toward zero and the spread across events is computed against a
        # fabricated observation. The event is therefore dropped, exactly as an
        # event with an incomplete window is - a study of 3 events reported as
        # 3, never as 4 with one silently blank.
        usable = np.isfinite(baseline.to_numpy())
        if not usable.all():
            skipped = skipped.append(matrix.index[~usable])
            matrix = matrix.loc[usable]
            values = values[usable]
            baseline = baseline[usable]
            if matrix.empty:
                return EventStudyResult(
                    table=pd.DataFrame(),
                    n_events=0,
                    used_events=pd.DatetimeIndex([]),
                    skipped_events=skipped.sort_values(),
                    abnormal=abnormal,
                    estimation_window=estimation_window,
                )
        values = values - baseline.to_numpy()[:, None]

    offsets = matrix.columns.to_numpy()
    n_events = values.shape[0]
    mean_by_offset = np.nanmean(values, axis=0)
    std_by_offset = (
        np.nanstd(values, axis=0, ddof=1) if n_events > 1 else np.zeros_like(mean_by_offset)
    )

    # CAR is computed on the "after" side only - offset 0 is the event day.
    post_mask = offsets >= 0
    car_per_event = np.nancumsum(values[:, post_mask], axis=1)  # (events x offsets)
    mean_car_post = car_per_event.mean(axis=0)

    # Inference across events: t distribution with n-1 degrees of freedom.
    if n_events > 1:
        sd_car_post = car_per_event.std(axis=0, ddof=1)
        se_car_post = sd_car_post / np.sqrt(n_events)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat_post = np.where(se_car_post > 0, mean_car_post / se_car_post, np.nan)
        p_post = 2.0 * stats.t.sf(np.abs(t_stat_post), df=n_events - 1)
        t_crit = float(stats.t.ppf(0.975, df=n_events - 1))
        ci_low_post = mean_car_post - t_crit * se_car_post
        ci_high_post = mean_car_post + t_crit * se_car_post
    else:
        nan_like = np.full(mean_car_post.shape, np.nan)
        sd_car_post = nan_like
        t_stat_post = nan_like
        p_post = nan_like
        ci_low_post = nan_like
        ci_high_post = nan_like

    # Bootstrap across events - reported for comparison, not for decisions.
    # n_boot=0 skips it. The specification curve runs this function ~48,000
    # times under permutation, and 5,000 resamples of 4 events per run is the
    # whole cost of that job while contributing nothing: the curve reads the
    # t-interval, never the bootstrap one.
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        boot_indices = rng.integers(0, n_events, size=(n_boot, n_events))
        boot_car = car_per_event[boot_indices].mean(axis=1)
        boot_low_post = np.percentile(boot_car, 2.5, axis=0)
        boot_high_post = np.percentile(boot_car, 97.5, axis=0)
    else:
        boot_low_post = np.full(mean_car_post.shape, np.nan)
        boot_high_post = np.full(mean_car_post.shape, np.nan)

    def _expand(post_values: np.ndarray) -> np.ndarray:
        full = np.full(mean_by_offset.shape, np.nan)
        full[post_mask] = post_values
        return full

    table = pd.DataFrame(
        {
            "offset_days": offsets,
            "mean_daily_return": mean_by_offset,
            "std_across_events": std_by_offset,
            "car": _expand(mean_car_post),
            "car_sd_across_events": _expand(sd_car_post),
            "car_ci_low": _expand(ci_low_post),
            "car_ci_high": _expand(ci_high_post),
            "car_t_stat": _expand(t_stat_post),
            "car_p_value": _expand(np.clip(p_post, 0.0, 1.0)),
            "car_boot_ci_low": _expand(boot_low_post),
            "car_boot_ci_high": _expand(boot_high_post),
            "n_events": n_events,
        }
    ).set_index("offset_days")

    last_offset = int(offsets.max())
    car_summary = {
        "offset": last_offset,
        "car": float(table.loc[last_offset, "car"]),
        "ci_low": float(table.loc[last_offset, "car_ci_low"]),
        "ci_high": float(table.loc[last_offset, "car_ci_high"]),
        "p_value": float(table.loc[last_offset, "car_p_value"]),
        "t_stat": float(table.loc[last_offset, "car_t_stat"]),
        "boot_ci_low": float(table.loc[last_offset, "car_boot_ci_low"]),
        "boot_ci_high": float(table.loc[last_offset, "car_boot_ci_high"]),
    }

    return EventStudyResult(
        table=table,
        n_events=n_events,
        used_events=matrix.index,
        skipped_events=skipped,
        abnormal=abnormal,
        estimation_window=estimation_window if abnormal else None,
        car_summary=car_summary,
    )


def circular_shift_test(
    values: pd.Series,
    mask: pd.Series,
    *,
    n_permutations: int = 5000,
    seed: int = 20260901,
) -> dict:
    """"Window vs rest of sample" test, robust to autocorrelation.

    Null hypothesis: the mask is unrelated to returns. We draw circular shifts
    of the mask, which preserves both the autocorrelation of the returns and
    the structure of the mask itself (window length, count and clustering). A
    plain permutation of days would destroy the latter and overstate
    significance.
    """
    aligned = pd.DataFrame({"value": values, "mask": mask}).dropna()
    if aligned.empty or aligned["mask"].sum() == 0:
        return {
            "n_in": 0, "n_out": int(len(aligned)), "mean_in": np.nan, "mean_out": np.nan,
            "difference": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        }

    y = aligned["value"].to_numpy(dtype=float)
    m = aligned["mask"].to_numpy().astype(bool)
    n = len(y)

    observed = float(y[m].mean() - y[~m].mean()) if (~m).any() else np.nan

    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, n, size=n_permutations)
    null_diffs = np.empty(n_permutations)
    positions = np.arange(n)
    for i, shift in enumerate(shifts):
        shifted = m[(positions - shift) % n]
        if shifted.all() or not shifted.any():
            null_diffs[i] = np.nan
            continue
        null_diffs[i] = y[shifted].mean() - y[~shifted].mean()
    null_diffs = null_diffs[~np.isnan(null_diffs)]

    p_value = float((np.abs(null_diffs) >= abs(observed)).mean()) if len(null_diffs) else np.nan
    return {
        "n_in": int(m.sum()),
        "n_out": int((~m).sum()),
        "mean_in": float(y[m].mean()),
        "mean_out": float(y[~m].mean()),
        "difference": observed,
        "p_value": p_value,
        "ci_low": float(np.percentile(null_diffs, 2.5)) if len(null_diffs) else np.nan,
        "ci_high": float(np.percentile(null_diffs, 97.5)) if len(null_diffs) else np.nan,
    }


def window_scan(
    frame: pd.DataFrame,
    mask_columns: list[str],
    target_column: str,
    *,
    n_permutations: int = 2000,
    seed: int = 20260901,
) -> pd.DataFrame:
    """The same test across many windows - the input to multiple-testing correction.

    NOTE: every extra row in this table is another hypothesis tested. The raw
    p-values it produces are NOT suitable for inference; pass them through
    validation.multiple_testing.
    """
    rows = []
    for i, column in enumerate(mask_columns):
        if column not in frame.columns:
            continue
        result = circular_shift_test(
            frame[target_column],
            frame[column].astype(float) > 0,
            n_permutations=n_permutations,
            seed=seed + i,
        )
        result["hypothesis"] = column
        result["target"] = target_column
        rows.append(result)
    if not rows:
        return pd.DataFrame()
    columns = ["hypothesis", "target", "n_in", "n_out", "mean_in", "mean_out",
               "difference", "p_value"]
    return pd.DataFrame(rows).loc[:, columns]
