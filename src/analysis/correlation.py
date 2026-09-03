"""Correlations and regressions with autocorrelation-robust standard errors.

Default OLS standard errors assume independent residuals. On daily financial
data that assumption is false, and the consequence is one-sided: t statistics
come out too large, so we "find" relationships that are not there. Every
regression here therefore uses Newey-West (HAC) errors.

Rolling correlations are for LOOKING at how a relationship moves over time
(Bitcoin's correlation with the S&P was near zero until 2020 and clearly
positive afterwards), not for inference - which is why they carry no p-values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def rolling_correlation(
    left: pd.Series, right: pd.Series, window: int = 90, *, min_periods: int | None = None
) -> pd.Series:
    aligned = pd.DataFrame({"left": left, "right": right}).dropna()
    return (
        aligned["left"]
        .rolling(window, min_periods=min_periods or window // 2)
        .corr(aligned["right"])
        .rename(f"corr_{window}d")
    )


def hac_lags(n_obs: int) -> int:
    """Newey-West rule of thumb: number of lags ~ 4 * (n/100)^(2/9)."""
    return max(1, int(np.floor(4 * (n_obs / 100.0) ** (2.0 / 9.0))))


def hac_regression(
    target: pd.Series, predictors: pd.DataFrame, *, lags: int | None = None
) -> pd.DataFrame:
    """OLS with HAC errors. Returns a coefficient table, not a model object."""
    data = pd.concat([target.rename("__y"), predictors], axis=1).dropna()
    if data.empty or len(data) <= predictors.shape[1] + 1:
        return pd.DataFrame()
    y = data["__y"]
    X = sm.add_constant(data.drop(columns="__y"), has_constant="add")
    model = sm.OLS(y, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags or hac_lags(len(data))}
    )
    table = pd.DataFrame(
        {
            "coefficient": model.params,
            "std_error": model.bse,
            "t_stat": model.tvalues,
            "p_value": model.pvalues,
        }
    )
    table.attrs["n_obs"] = int(model.nobs)
    table.attrs["r_squared"] = float(model.rsquared)
    table.attrs["hac_lags"] = int(model.cov_kwds["maxlags"])
    return table


def lead_lag_correlation(
    left: pd.Series, right: pd.Series, max_lag: int = 30, *, step: int = 5
) -> pd.DataFrame:
    """Correlation at various offsets: does `right` lead `left`?

    A positive `lag` means `right` is shifted forward, i.e. we are asking
    whether its PAST values line up with the current `left`. Only positive lags
    have predictive meaning; negative ones are shown for contrast, because they
    are easy to mistake for a signal.
    """
    rows = []
    for lag in range(-max_lag, max_lag + 1, step):
        shifted = right.shift(lag)
        aligned = pd.DataFrame({"left": left, "right": shifted}).dropna()
        if len(aligned) < 30:
            continue
        rows.append(
            {
                "lag_days": lag,
                "correlation": float(aligned["left"].corr(aligned["right"])),
                "n_obs": int(len(aligned)),
                "predictive": lag > 0,
            }
        )
    return pd.DataFrame(rows)


def regime_returns(
    returns: pd.Series, regime: pd.Series, *, periods_per_year: int = 365
) -> pd.DataFrame:
    """Return statistics split by macro regime - descriptive, not a test."""
    aligned = pd.DataFrame({"return": returns, "regime": regime}).dropna()
    if aligned.empty:
        return pd.DataFrame()
    grouped = aligned.groupby("regime")["return"]
    table = pd.DataFrame(
        {
            "days": grouped.size(),
            "mean_daily": grouped.mean(),
            "volatility_annual": grouped.std() * np.sqrt(periods_per_year),
            "share_positive": grouped.apply(lambda s: float((s > 0).mean())),
        }
    )
    table["annualized_return"] = table["mean_daily"] * periods_per_year
    return table.sort_values("days", ascending=False)
