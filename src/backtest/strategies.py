"""Strategie - kazda to funkcja zwracajaca DOCELOWA pozycje na dany dzien.

Zadna z nich nie przesuwa sygnalu w czasie; opoznienie wykonania naklada
silnik (engine.run_backtest). Dublowanie przesuniecia jest bledem tak
samo jak jego brak.

Strategie sa celowo prymitywne. To nie sa propozycje inwestycyjne, tylko
sposob na sprawdzenie, czy wzorzec znaleziony w event study przetrwa
zderzenie z kosztami i z baseline'em kup-i-trzymaj.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.halving import CONFIRMED_HALVINGS


def buy_and_hold(index: pd.DatetimeIndex) -> pd.Series:
    """Baseline. Kazda strategia musi sie z nia porownac."""
    return pd.Series(1.0, index=index, name="buy_and_hold")


def from_mask(mask: pd.Series, *, position: float = 1.0) -> pd.Series:
    """Pozycja `position` gdy maska jest prawdziwa, zero poza nia."""
    return (mask.astype(float) * position).rename("mask_strategy")


def halving_window(
    index: pd.DatetimeIndex, *, days_after: int = 180, position: float = 1.0
) -> pd.Series:
    """W rynku przez N dni po halvingu, poza rynkiem w pozostale dni.

    Uzywa wylacznie potwierdzonych halvingow z przeszlosci - w dniu t
    wiadomo, ile dni minelo od ostatniego halvingu, i nic wiecej.
    """
    index = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    flag = np.zeros(len(index))
    for halving in CONFIRMED_HALVINGS:
        window = (index >= halving) & (index <= halving + pd.Timedelta(days=days_after))
        flag = np.maximum(flag, window.astype(float))
    return pd.Series(flag * position, index=index, name=f"halving_{days_after}d")


def macro_regime(
    regime: pd.Series, *, long_labels: tuple[str, ...] = ("expanding_falling",)
) -> pd.Series:
    """Pozycja tylko w wybranych fazach makro.

    Domyslnie: rosnaca plynnosc przy spadajacych stopach. To hipoteza do
    sprawdzenia, nie wiedza - w probie sa cztery cykle, wiec kazda faza ma
    garstke niezaleznych epizodow.
    """
    return regime.isin(long_labels).astype(float).rename("macro_regime")


def trend_following(
    close: pd.Series, *, fast: int = 50, slow: int = 200, position: float = 1.0
) -> pd.Series:
    """Klasyczny filtr trendu - uczciwy punkt odniesienia dla strategii cyklicznych.

    Jesli hipoteza halvingowa nie bije nawet przeciecia srednich, to znaczy,
    ze mierzy trend, a nie cykl polowien.
    """
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(float) * position
    signal[slow_ma.isna()] = 0.0
    return signal.rename(f"trend_{fast}_{slow}")


def combine(*signals: pd.Series, mode: str = "all") -> pd.Series:
    """Laczy sygnaly: `all` = iloczyn warunkow, `any` = suma logiczna."""
    if not signals:
        raise ValueError("brak sygnalow do polaczenia")
    frame = pd.concat(signals, axis=1).fillna(0.0)
    if mode == "all":
        combined = frame.min(axis=1)
    elif mode == "any":
        combined = frame.max(axis=1)
    else:
        raise ValueError("mode musi byc 'all' albo 'any'")
    return combined.rename(f"combined_{mode}")
