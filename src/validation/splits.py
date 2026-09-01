"""Podzialy proby: po cyklach, kroczaco, z embargiem.

Trzy pulapki, ktore ten modul zamyka:

1. Losowy podzial szeregu czasowego nie ma sensu - uczylby sie na
   przyszlosci. Dlatego kazdy podzial jest chronologiczny.
2. Cel `fwd_return_90d` w dniu t zawiera ceny z t+90. Jesli t jest ostatnim
   dniem treningu, a t+1 pierwszym dniem testu, zbiory zachodza na siebie.
   Stad EMBARGO: luka rowna horyzontowi celu, wycinana miedzy zbiorami.
3. Podzial po cyklach halvingowych jest naturalny dla tego projektu, ale ma
   swoja cene: cykli jest piec, wiec zbior testowy to jeden-dwa cykle.
   Wynik "dziala na cyklu 4" to jedna obserwacja, nie dowod.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    """Podzial po numerach cykli halvingowych, z embargiem miedzy zbiorami."""
    index = pd.DatetimeIndex(pd.to_datetime(index)).normalize().sort_values()
    cycles = cycle_of(index)
    train = index[cycles.isin(train_cycles).to_numpy()]
    test = index[cycles.isin(test_cycles).to_numpy()]

    embargo = pd.DatetimeIndex([])
    if embargo_days > 0 and len(train) and len(test):
        boundary = train.max()
        cutoff = boundary + pd.Timedelta(days=embargo_days)
        embargo = train[train > boundary - pd.Timedelta(days=embargo_days)]
        train = train[train <= boundary - pd.Timedelta(days=embargo_days)]
        test = test[test > boundary]
        _ = cutoff  # embargo wycinamy po stronie treningu, test zaczyna sie za granica

    name = f"cykle {train_cycles} -> {test_cycles}"
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
    """Kroczaca walidacja: ucz na przeszlosci, testuj na kolejnym oknie.

    `expanding=True` powieksza okno treningowe z kazdym krokiem (uczciwy
    odpowiednik tego, jak dziala sie w praktyce). `expanding=False` daje
    okno przesuwne o stalej dlugosci.
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


def assert_no_overlap(split: Split, *, horizon_days: int = 0) -> None:
    """Pilnuje, ze zbiory sa rozlaczne, a luka pokrywa horyzont celu."""
    overlap = split.train.intersection(split.test)
    if len(overlap):
        raise AssertionError(f"{split.name}: zbiory zachodza na siebie ({len(overlap)} dni)")
    if len(split.train) == 0 or len(split.test) == 0:
        return
    gap = (split.test.min() - split.train.max()).days
    if gap <= horizon_days:
        raise AssertionError(
            f"{split.name}: luka {gap} dni nie pokrywa horyzontu celu {horizon_days} dni - "
            "ostatnie dni treningu zawieraja ceny ze zbioru testowego"
        )


def split_frame(frame: pd.DataFrame, split: Split) -> tuple[pd.DataFrame, pd.DataFrame]:
    return frame.loc[frame.index.isin(split.train)], frame.loc[frame.index.isin(split.test)]


def replicate_finding(
    train_result: dict, test_result: dict, *, alpha: float = 0.05
) -> dict:
    """Czy wynik z treningu potwierdzil sie na tescie?

    Wymagamy trzech rzeczy naraz: tego samego znaku efektu, istotnosci na
    tescie i tego, zeby efekt nie skurczyl sie do ulamka. Sam znak to za malo -
    przy dwoch mozliwych znakach trafia sie w polowie przypadkow.
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
