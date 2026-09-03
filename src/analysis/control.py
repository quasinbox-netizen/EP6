"""Control group - a placebo test for supposedly "cyclical" effects.

The question this module answers: is what we see around a halving an effect of
the halving, or simply what every risk asset was doing at the time? A halving
is a Bitcoin-ONLY event, so the NASDAQ in the same window has no way of
knowing about it. If it reacts the same way, we are measuring the common market
rather than a reward halving.

The test is PAIRED across events (difference in differences): for each halving
we compute Bitcoin's CAR and the control's CAR over the same window, and infer
from the distribution of their difference. Pairing matters - the 2020 halving
fell in the middle of the pandemic rebound, which lifted both assets.
Comparing two separate means would lose that fact; the paired difference
removes it.

Calendar: the NASDAQ trades about 252 days a year, Bitcoin 365. Every window
here is a CALENDAR window - the control series is mapped onto the full calendar
by carrying the last known close forward (`ffill`, strictly backward-looking).
Without that, "365 days after the halving" would mean almost 17 months for the
NASDAQ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from analysis.event_study import (
    DEFAULT_ESTIMATION_WINDOW,
    _estimation_means,
    event_study,
    event_window_matrix,
    log_returns,
)


@dataclass
class ControlComparison:
    """Result of comparing the studied asset with a control asset."""

    treatment_name: str
    control_name: str
    table: pd.DataFrame
    per_event: pd.DataFrame
    n_events: int
    summary_at: dict

    def verdict(self, alpha: float = 0.05) -> str:
        row = self.summary_at
        if self.n_events < 2:
            return f"n={self.n_events} - too few events for any conclusion"
        if not np.isfinite(row["difference_p_value"]):
            return "not enough data for the difference test"
        if row["difference_p_value"] < alpha:
            return (
                f"Difference {self.treatment_name} - {self.control_name} = "
                f"{row['difference']:+.1%} (p={row['difference_p_value']:.3f}) - "
                "the effect is NOT common to both assets"
            )
        return (
            f"Difference {self.treatment_name} - {self.control_name} = "
            f"{row['difference']:+.1%} [{row['difference_ci_low']:+.1%}, "
            f"{row['difference_ci_high']:+.1%}], p={row['difference_p_value']:.3f} - "
            "indistinguishable from what the control group did"
        )


def to_calendar(prices: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
    """Map a series onto the full daily calendar.

    Non-trading days receive the last known close (`ffill`). This is strictly
    backward-looking: on Saturday we know Friday's close, not Monday's.

    One side effect to keep in mind: weekends carry a zero return, so the daily
    mean and volatility are computed over 365 rather than 252 days. For CAR
    this is irrelevant - the sum of returns over a window depends only on the
    closes at its ends - and CAR is exactly the unit of comparison here.
    """
    frame = prices.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame = frame.drop_duplicates(subset="date", keep="last").set_index("date")
    frame = frame.sort_index()
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    out = frame.reindex(full_index).ffill()
    out.index.name = "date"
    return out.reset_index()


def per_event_car(
    prices: pd.DataFrame,
    event_dates,
    *,
    post: int,
    pre: int = 30,
    abnormal: bool = True,
    estimation_window: tuple[int, int] = DEFAULT_ESTIMATION_WINDOW,
    price_column: str = "close",
) -> pd.Series:
    """CAR for each event separately, in calendar days.

    Returns a series indexed by event date - the raw material for the paired
    test.
    """
    returns = log_returns(prices, price_column)
    matrix, _ = event_window_matrix(returns, event_dates, pre=pre, post=post)
    if matrix.empty:
        return pd.Series(dtype=float)

    values = matrix.to_numpy(dtype=float)
    if abnormal:
        baseline = _estimation_means(returns, matrix.index, estimation_window)
        values = values - baseline.to_numpy()[:, None]

    offsets = matrix.columns.to_numpy()
    post_values = values[:, offsets >= 0]
    return pd.Series(np.nansum(post_values, axis=1), index=matrix.index, name="car")


def compare_with_control(
    treatment_prices: pd.DataFrame,
    control_prices: pd.DataFrame,
    event_dates,
    *,
    treatment_name: str = "BTC",
    control_name: str = "NASDAQ",
    post: int = 365,
    pre: int = 30,
    abnormal: bool = True,
    horizons: list[int] | None = None,
    align_calendar: bool = True,
) -> ControlComparison:
    """Difference in differences: treatment CAR minus control CAR, across events.

    `horizons` are the points at which we report (default 30/90/180 and the
    full window). For each one we compute the paired difference and a t-test
    with n-1 degrees of freedom - the same as in event_study, because the unit
    of observation is still the event.
    """
    horizons = sorted({h for h in (horizons or [30, 90, 180, post]) if h <= post})
    control = to_calendar(control_prices) if align_calendar else control_prices

    rows = []
    per_event_frames = {}
    for horizon in horizons:
        treatment_car = per_event_car(
            treatment_prices, event_dates, post=horizon, pre=pre, abnormal=abnormal
        )
        control_car = per_event_car(
            control, event_dates, post=horizon, pre=pre, abnormal=abnormal
        )
        shared = treatment_car.index.intersection(control_car.index)
        if len(shared) == 0:
            continue

        paired = pd.DataFrame(
            {
                treatment_name: treatment_car.loc[shared],
                control_name: control_car.loc[shared],
            }
        )
        paired["difference"] = paired[treatment_name] - paired[control_name]
        per_event_frames[horizon] = paired

        n = len(paired)
        difference = float(paired["difference"].mean())
        if n > 1:
            se = float(paired["difference"].std(ddof=1) / np.sqrt(n))
            t_stat = difference / se if se > 0 else np.nan
            p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if se > 0 else np.nan
            t_crit = float(stats.t.ppf(0.975, df=n - 1))
            ci_low, ci_high = difference - t_crit * se, difference + t_crit * se
        else:
            t_stat = p_value = ci_low = ci_high = np.nan

        rows.append(
            {
                "horizon_days": horizon,
                "n_events": n,
                f"car_{treatment_name}": float(paired[treatment_name].mean()),
                f"car_{control_name}": float(paired[control_name].mean()),
                "difference": difference,
                "difference_ci_low": ci_low,
                "difference_ci_high": ci_high,
                "difference_t_stat": t_stat,
                "difference_p_value": p_value,
            }
        )

    table = pd.DataFrame(rows).set_index("horizon_days") if rows else pd.DataFrame()
    longest = max(per_event_frames) if per_event_frames else None
    return ControlComparison(
        treatment_name=treatment_name,
        control_name=control_name,
        table=table,
        per_event=per_event_frames.get(longest, pd.DataFrame()),
        n_events=int(table.loc[longest, "n_events"]) if longest is not None else 0,
        summary_at=table.loc[longest].to_dict() if longest is not None else {},
    )


def placebo_event_study(
    control_prices: pd.DataFrame,
    event_dates,
    *,
    post: int = 365,
    pre: int = 30,
    n_boot: int = 2000,
    seed: int = 20260901,
):
    """Event study on the control asset - same code, different asset.

    If halvings also "work" on the NASDAQ, the problem is in the method or the
    sample, not in Bitcoin.
    """
    return event_study(
        to_calendar(control_prices),
        event_dates,
        pre=pre,
        post=post,
        n_boot=n_boot,
        seed=seed,
    )
