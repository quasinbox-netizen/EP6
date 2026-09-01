"""Faza makro - rezim plynnosci i stop, liczony wylacznie z danych znanych w danym dniu.

Cala logika opiera sie na jednym prymitywie: `asof_series`. Dla kazdego dnia
kalendarzowego bierze ostatnia obserwacje, ktorej data PUBLIKACJI nie jest
pozniejsza niz ten dzien. Transformacje (r/r, zmiana 3M) liczone sa najpierw
na osi obserwacji, a dopiero potem mapowane na dni - wartosc r/r staje sie
znana wtedy, kiedy opublikowano jej pozniejszy skladnik.

Progi rezimow uzywaja median ROZSZERZAJACYCH SIE (expanding), nie pelnej
proby. Mediana z calej historii to klasyczny, cichy look-ahead: w 2015 r.
nie znalismy mediany z lat 2015-2026.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LIQUIDITY_SOURCES = ("m2_yoy", "dxy_chg_3m_inv")
RATES_SOURCES = ("fed_funds_chg_3m", "us13w_chg_3m", "us10y_chg_3m")

MIN_HISTORY_DAYS = 365  # zanim uznamy rezim, potrzebujemy roku historii


def asof_series(
    macro: pd.DataFrame,
    series: str,
    index: pd.DatetimeIndex,
    *,
    transform=None,
) -> pd.Series:
    """Wartosc serii `series` znana na kazdy dzien z `index`.

    `transform` dostaje szereg indeksowany data obserwacji i zwraca szereg
    o tym samym indeksie (np. zmiana r/r). Mapowanie na dni kalendarzowe
    odbywa sie po dacie publikacji.
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
    """Surowe cechy makro w ujeciu punkt-w-czasie (bez klasyfikacji rezimu)."""
    index = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    out = pd.DataFrame(index=index)
    out.index.name = "date"
    if macro.empty:
        return out

    present = set(macro["series"].unique())

    # Serie miesieczne: r/r to 12 obserwacji wstecz.
    for name in ("m2", "indpro", "unrate"):
        if name in present:
            out[f"{name}_yoy"] = asof_series(macro, name, index, transform=_yoy(12))

    # Serie dzienne: zmiana w oknie ~63 sesji (kwartal).
    for name in ("fed_funds", "us13w", "us10y", "dxy", "spx", "gold"):
        if name in present:
            out[f"{name}_level"] = asof_series(macro, name, index)
            out[f"{name}_chg_3m"] = asof_series(macro, name, index, transform=_diff(63))
    if "dxy_chg_3m" in out.columns:
        # Slabszy dolar = luzniejsze warunki finansowe, stad znak przeciwny.
        out["dxy_chg_3m_inv"] = -out["dxy_chg_3m"]
    return out


def _expanding_sign_regime(series: pd.Series, labels: tuple[str, str]) -> pd.Series:
    """Klasyfikacja wzgledem mediany rozszerzajacej sie (bez wiedzy o przyszlosci)."""
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
    """Cechy makro + etykieta fazy.

    Faza to iloczyn dwoch osi:
      plynnosc (expanding / contracting) x stopy (rising / falling).
    Jesli brakuje danych do ktorejkolwiek osi, faza jest pusta (NA) - lepszy
    brak etykiety niz etykieta zmyslona z proxy, ktorego nie ma.

    Domyslnie wybieramy pierwsze dostepne zrodlo z listy priorytetow
    (M2 przed proxy dolarowym). Jawne podanie `liquidity_source` /
    `rates_source` pozwala porownac wersje na prawdziwym M2 z wersja na
    proxy - patrz pipeline.macro_phase_comparison.
    """
    features = macro_features(macro, index)
    if liquidity_source is not None and liquidity_source not in features.columns:
        raise ValueError(f"brak kolumny {liquidity_source} - czy ta seria jest w bazie?")
    if rates_source is not None and rates_source not in features.columns:
        raise ValueError(f"brak kolumny {rates_source} - czy ta seria jest w bazie?")
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
