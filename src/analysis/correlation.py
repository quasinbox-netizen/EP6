"""Korelacje i regresje z bledami odpornymi na autokorelacje.

Domyslne bledy standardowe OLS zakladaja niezaleznosc reszt. Na dziennych
danych finansowych to zalozenie jest falszywe, a skutek jest jednostronny:
statystyki t sa zawyzone, wiec "znajdujemy" zaleznosci, ktorych nie ma.
Dlatego kazda regresja tutaj uzywa bledow Neweya-Westa (HAC).

Korelacje kroczace sluza do OGLADANIA niestabilnosci zaleznosci w czasie
(korelacja BTC z SPX byla bliska zera do 2020 r. i wyraznie dodatnia
pozniej), a nie do wnioskowania - dlatego nie maja p-value.
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
    """Regula Neweya-Westa: liczba opoznien ~ 4 * (n/100)^(2/9)."""
    return max(1, int(np.floor(4 * (n_obs / 100.0) ** (2.0 / 9.0))))


def hac_regression(
    target: pd.Series, predictors: pd.DataFrame, *, lags: int | None = None
) -> pd.DataFrame:
    """OLS z bledami HAC. Zwraca tabele wspolczynnikow, nie obiekt modelu."""
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
    """Korelacja przy roznych przesunieciach: czy `right` wyprzedza `left`.

    Dodatni `lag` oznacza, ze `right` jest przesuniety w przod, czyli
    sprawdzamy, czy jego PRZESZLE wartosci wspolgraja z biezacym `left`.
    Tylko dodatnie opoznienia maja sens predykcyjny; ujemne pokazujemy dla
    kontrastu, bo latwo pomylic je z sygnalem.
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
    """Statystyki zwrotow w podziale na rezimy makro - opis, nie test."""
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
