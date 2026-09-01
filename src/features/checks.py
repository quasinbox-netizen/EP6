"""Wykrywanie look-ahead bias metoda punkt-w-czasie.

Idea testu: dla wybranego dnia t budujemy cechy DWA razy - raz majac
wylacznie dane opublikowane do dnia t, raz majac cala historie do dzis.
Wiersz t musi wyjsc identycznie. Jesli sie rozni, to znaczy, ze jakas
cecha w dniu t korzysta z informacji, ktora wtedy jeszcze nie istniala.

To mocniejszy test niz porownanie "przycietej" ramki z pelna: wychwytuje
takze cechy liczone na calej probie (mediany, z-score, normalizacje),
ktore przy zwyklym przycinaniu wygladaja niewinnie.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RELATIVE_TOLERANCE = 1e-9


def _comparable(value) -> object:
    if isinstance(value, float) and np.isnan(value):
        return "NaN"
    if value is None or value is pd.NaT or value is pd.NA:
        return "NaN"
    return value


def _differs(full_value, pit_value) -> bool:
    a, b = _comparable(full_value), _comparable(pit_value)
    if a == "NaN" or b == "NaN":
        return a != b
    if isinstance(a, (int, float, np.floating, np.integer)) and isinstance(
        b, (int, float, np.floating, np.integer)
    ):
        return not np.isclose(float(a), float(b), rtol=RELATIVE_TOLERANCE, atol=1e-12)
    return a != b


def pointwise_lookahead_report(
    build_fn,
    test_dates: list[str] | pd.DatetimeIndex,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Porownuje najswiezszy wiersz z budowy punkt-w-czasie z wersja pelna.

    `build_fn(as_of)` ma zwrocic ramke cech indeksowana data; `as_of=None`
    oznacza pelna historie. Dla dnia t bierzemy OSTATNI wiersz, jaki dalo sie
    wtedy policzyc (bar dnia t zamyka sie dopiero o polnocy, wiec zwykle jest
    to t-1) i porownujemy go z tym samym dniem policzonym z pelna historia.

    Zwraca ramke rozbieznosci - pusta ramka to wynik pozytywny.
    """
    full = build_fn(None)
    if full.empty:
        raise ValueError("pelna budowa cech zwrocila pusta ramke")
    checked = columns or [c for c in full.columns if not c.startswith("fwd_return_")]

    findings = []
    for day in pd.DatetimeIndex(pd.to_datetime(test_dates)).normalize():
        point_in_time = build_fn(day)
        if point_in_time.empty:
            findings.append(
                {
                    "date": day,
                    "column": "<caly wiersz>",
                    "full_value": "obecny" if day in full.index else "brak",
                    "point_in_time_value": "pusta ramka",
                }
            )
            continue
        target = point_in_time.index.max()
        if target not in full.index:
            findings.append(
                {
                    "date": target,
                    "column": "<caly wiersz>",
                    "full_value": "brak",
                    "point_in_time_value": "obecny",
                }
            )
            continue
        for column in checked:
            if column not in point_in_time.columns:
                findings.append(
                    {
                        "date": target,
                        "column": column,
                        "full_value": full.loc[target, column],
                        "point_in_time_value": "<brak kolumny>",
                    }
                )
                continue
            full_value = full.loc[target, column]
            pit_value = point_in_time.loc[target, column]
            if _differs(full_value, pit_value):
                findings.append(
                    {
                        "date": target,
                        "column": column,
                        "full_value": full_value,
                        "point_in_time_value": pit_value,
                    }
                )
    return pd.DataFrame(findings)


def assert_no_lookahead(build_fn, test_dates, *, columns: list[str] | None = None) -> None:
    """Wersja dla testow: podnosi AssertionError z lista winnych kolumn."""
    report = pointwise_lookahead_report(build_fn, test_dates, columns=columns)
    if not report.empty:
        culprits = report["column"].value_counts().to_dict()
        raise AssertionError(
            f"look-ahead bias w {len(report)} komorkach; kolumny: {culprits}"
        )


def targets_are_forward_only(frame: pd.DataFrame, horizon: int) -> bool:
    """Sanity check celu: ostatnie `horizon` dni musi byc puste.

    Jesli fwd_return_Nd ma wartosc na ostatnim dniu proby, to znaczy, ze cel
    zostal policzony wstecz albo przesuniety w zla strone.
    """
    column = f"fwd_return_{horizon}d"
    if column not in frame.columns:
        raise KeyError(f"brak kolumny {column}")
    return bool(frame[column].tail(horizon).isna().all())
