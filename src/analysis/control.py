"""Grupa kontrolna - test placebo dla efektow "cyklicznych".

Pytanie, na ktore ten modul odpowiada: czy to, co widzimy wokol halvingu,
jest efektem halvingu, czy po prostu tym, co robily wtedy wszystkie aktywa
ryzykowne. Halving jest zdarzeniem WYLACZNIE bitcoinowym, wiec NASDAQ w tym
samym oknie nie ma prawa nic o nim wiedziec. Jesli reaguje tak samo, to
znaczy, ze mierzymy wspolny rynek, a nie polowienie nagrody.

Test jest SPAROWANY po zdarzeniach (roznica w roznicach): dla kazdego
halvingu liczymy CAR bitcoina i CAR kontroli w tym samym oknie, a wnioskujemy
z rozkladu ich roznicy. Parowanie ma znaczenie - halving 2020 wypadl w
srodku pandemicznego odbicia, ktore podnioslo oba aktywa. Porownanie dwoch
osobnych srednich zgubiloby ten fakt, roznica sparowana go usuwa.

Kalendarz: NASDAQ handluje sie okolo 252 dni w roku, BTC 365. Wszystkie okna
sa tu KALENDARZOWE - szereg kontrolny przenosimy na pelny kalendarz,
przenoszac ostatnie znane zamkniecie na dni bez sesji (`ffill`, czyli
wylacznie w przod, bez zagladania w przyszlosc). Bez tego "365 dni po
halvingu" znaczyloby dla NASDAQ prawie 17 miesiecy.
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
    """Wynik porownania aktywa badanego z kontrolnym."""

    treatment_name: str
    control_name: str
    table: pd.DataFrame
    per_event: pd.DataFrame
    n_events: int
    summary_at: dict

    def verdict(self, alpha: float = 0.05) -> str:
        row = self.summary_at
        if self.n_events < 2:
            return f"n={self.n_events} - za malo zdarzen na jakikolwiek wniosek"
        if not np.isfinite(row["difference_p_value"]):
            return "brak wystarczajacych danych do testu roznicy"
        if row["difference_p_value"] < alpha:
            return (
                f"Roznica {self.treatment_name} - {self.control_name} = "
                f"{row['difference']:+.1%} (p={row['difference_p_value']:.3f}) - "
                "efekt NIE jest wspolny dla obu aktywow"
            )
        return (
            f"Roznica {self.treatment_name} - {self.control_name} = "
            f"{row['difference']:+.1%} [{row['difference_ci_low']:+.1%}, "
            f"{row['difference_ci_high']:+.1%}], p={row['difference_p_value']:.3f} - "
            "nie da sie odroznic od tego, co robila grupa kontrolna"
        )


def to_calendar(prices: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
    """Przenosi szereg na pelny kalendarz dzienny.

    Dni bez sesji dostaja ostatnie znane zamkniecie (`ffill`). To operacja
    wylacznie wsteczna: w sobote znamy piatkowe zamkniecie, nie poniedzialkowe.

    Skutek uboczny, o ktorym trzeba pamiec: weekendy maja zerowy zwrot, wiec
    srednia dzienna i zmiennosc licza sie na 365, a nie 252 dni. Dla CAR jest
    to bez znaczenia - suma zwrotow w oknie zalezy tylko od zamkniec na jego
    koncach - i wlasnie CAR jest tu jednostka porownania.
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
    """CAR kazdego zdarzenia osobno, w dniach kalendarzowych.

    Zwraca szereg indeksowany data zdarzenia - to jest surowiec dla testu
    sparowanego.
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
    """Roznica w roznicach: CAR badanego minus CAR kontroli, po zdarzeniach.

    `horizons` to punkty, w ktorych raportujemy wynik (domyslnie 30/90/180
    i pelne okno). Dla kazdego liczymy sparowana roznice i test t o n-1
    stopniach swobody - tak samo jak w event_study, bo jednostka obserwacji
    dalej jest zdarzenie.
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
    """Event study na aktywie kontrolnym - ten sam kod, inne aktywo.

    Jesli halvingi "dzialaja" takze na NASDAQ, problem jest w metodzie albo
    w probie, a nie w bitcoinie.
    """
    return event_study(
        to_calendar(control_prices),
        event_dates,
        pre=pre,
        post=post,
        n_boot=n_boot,
        seed=seed,
    )
