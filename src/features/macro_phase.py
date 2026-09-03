"""Macro phase - liquidity and rate regimes, built only from data known that day.

Everything here rests on one primitive: `asof_series`. For each calendar day it
takes the latest observation whose PUBLICATION date is not later than that day.
Transformations (year-over-year, 3-month change) are computed first on the
observation axis and only then mapped onto days - a year-over-year value becomes
known when its later component is published.

Regime thresholds use EXPANDING medians, not the full-sample median. A
full-history median is the classic quiet look-ahead: in 2015 we did not know
the median of 2015-2026.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LIQUIDITY_SOURCES = ("m2_yoy", "dxy_chg_3m_inv")
RATES_SOURCES = ("fed_funds_chg_3m", "us13w_chg_3m", "us10y_chg_3m")

MIN_HISTORY_DAYS = 365  # a year of history before we call a regime at all


def asof_series(
    macro: pd.DataFrame,
    series: str,
    index: pd.DatetimeIndex,
    *,
    transform=None,
) -> pd.Series:
    """The value of `series` known on each day of `index`.

    `transform` receives a series indexed by observation date and returns a
    series with the same index (e.g. year-over-year change). Mapping onto
    calendar days happens afterwards, by publication date.
    """
    index = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    subset = macro[macro["series"] == series]
    if subset.empty:
        return pd.Series(np.nan, index=index, name=series)

    subset = subset.sort_values("date")
    observations = pd.Series(
        subset["value"].to_numpy(),
        index=pd.DatetimeIndex(subset["date"]).normalize(),
        name=series,
    )
    available = pd.Series(
        pd.DatetimeIndex(subset["available_from"]).normalize(),
        index=observations.index,
    )
    if transform is not None:
        observations = transform(observations)

    frame = (
        pd.DataFrame({"value": observations, "available_from": available})
        .dropna(subset=["value"])
        .sort_values("available_from")
    )
    if frame.empty:
        return pd.Series(np.nan, index=index, name=series)

    merged = pd.merge_asof(
        pd.DataFrame({"day": index}),
        frame.reset_index(drop=True)[["available_from", "value"]],
        left_on="day",
        right_on="available_from",
        direction="backward",
    )
    return pd.Series(merged["value"].to_numpy(), index=index, name=series)


def _yoy(periods: int):
    def transform(series: pd.Series) -> pd.Series:
        return series.pct_change(periods)

    return transform


def _diff(periods: int):
    def transform(series: pd.Series) -> pd.Series:
        return series.diff(periods)

    return transform


def macro_features(macro: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Raw macro features, point-in-time, without regime labels."""
    index = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    out = pd.DataFrame(index=index)
    out.index.name = "date"
    if macro.empty:
        return out

    present = set(macro["series"].unique())

    # Monthly series: year-over-year means 12 observations back.
    for name in ("m2", "indpro", "unrate"):
        if name in present:
            out[f"{name}_yoy"] = asof_series(macro, name, index, transform=_yoy(12))

    # Daily series: change over roughly 63 sessions (a quarter).
    for name in ("fed_funds", "us13w", "us10y", "dxy", "spx", "gold"):
        if name in present:
            out[f"{name}_level"] = asof_series(macro, name, index)
            out[f"{name}_chg_3m"] = asof_series(macro, name, index, transform=_diff(63))
    if "dxy_chg_3m" in out.columns:
        # A weaker dollar means looser financial conditions, hence the sign flip.
        out["dxy_chg_3m_inv"] = -out["dxy_chg_3m"]
    return out


def _expanding_sign_regime(series: pd.Series, labels: tuple[str, str]) -> pd.Series:
    """Classify against an expanding median - no knowledge of the future."""
    threshold = series.expanding(min_periods=MIN_HISTORY_DAYS).median()
    regime = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = series.notna() & threshold.notna()
    regime[valid & (series >= threshold)] = labels[0]
    regime[valid & (series < threshold)] = labels[1]
    return regime


def macro_phase(
    macro: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    liquidity_source: str | None = None,
    rates_source: str | None = None,
) -> pd.DataFrame:
    """Macro features plus a phase label.

    The phase is the product of two axes:
      liquidity (expanding / contracting) x rates (rising / falling).
    If data for either axis is missing the phase is empty (NA) - no label is
    better than a label invented from a proxy that is not there.

    By default we take the first available source from the priority list (M2
    before the dollar proxy). Passing `liquidity_source` / `rates_source`
    explicitly allows comparing the real-M2 version against the proxy - see
    pipeline.macro_phase_comparison.
    """
    features = macro_features(macro, index)
    if liquidity_source is not None and liquidity_source not in features.columns:
        raise ValueError(f"no column {liquidity_source} - is that series in the database?")
    if rates_source is not None and rates_source not in features.columns:
        raise ValueError(f"no column {rates_source} - is that series in the database?")
    liquidity_source = liquidity_source or next(
        (c for c in LIQUIDITY_SOURCES if c in features.columns), None
    )
    rates_source = rates_source or next(
        (c for c in RATES_SOURCES if c in features.columns), None
    )

    if liquidity_source:
        features["liquidity_regime"] = _expanding_sign_regime(
            features[liquidity_source], ("expanding", "contracting")
        )
    if rates_source:
        features["rates_regime"] = _expanding_sign_regime(
            features[rates_source], ("rising", "falling")
        )

    if liquidity_source and rates_source:
        combined = features["liquidity_regime"].astype("object") + "_" + features[
            "rates_regime"
        ].astype("object")
        features["macro_phase"] = combined.where(
            features["liquidity_regime"].notna() & features["rates_regime"].notna()
        )
    else:
        features["macro_phase"] = pd.Series(pd.NA, index=features.index, dtype="object")

    features.attrs["liquidity_source"] = liquidity_source
    features.attrs["rates_source"] = rates_source
    return features
