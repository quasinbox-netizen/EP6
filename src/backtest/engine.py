"""Silnik backtestu: dzienny, wektorowy, z kosztami i poslizgiem.

Model wykonania - jawnie, bo tu mieszka wiekszosc cichych bledow:

* sygnal dnia t powstaje z danych znanych NA ZAMKNIECIU dnia t,
* zwrot dnia t to zmiana zamkniecia z t-1 na t,
* przy `execution_lag_days = 1` sygnal z dnia t zbiera zwrot dnia t+1,
  czyli wchodzisz po zamknieciu, ktore wlasnie zobaczyles, i zarabiasz
  dopiero ruch nastepnego dnia.

Zerowe opoznienie oznaczaloby, ze sygnal z dnia t zbiera zwrot dnia t -
czyli handel po cenie, ktora dopiero sie ustala. To najczestsza przyczyna
backtestow "dzialajacych" wylacznie w arkuszu; test
test_execution_lag_blocks_same_day_knowledge pokazuje roznice na liczbach.

Konsekwencja dla baseline: kup-i-trzymaj bez kosztow zwraca dokladnie
tyle, ile samo aktywo - pierwszy dzien proby sluzy na wystawienie zlecenia.

Koszt naliczamy od OBROTU (|zmiana pozycji|), a nie od liczby transakcji:
wejscie z 0 na 1 kosztuje tyle samo co wyjscie z 1 na 0, a zmiana 0.5 -> 0.6
kosztuje jedna dziesiata tego. Prowizja i poslizg sa osobnymi parametrami,
bo skaluja sie inaczej - prowizje negocjujesz, poslizgu nie.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    fee_bps: float = 10.0
    slippage_bps: float = 15.0
    execution_lag_days: int = 1
    initial_equity: float = 10_000.0
    periods_per_year: int = 365
    allow_short: bool = False
    max_leverage: float = 1.0

    @property
    def cost_rate(self) -> float:
        """Koszt jednostronny jako ulamek obrotu."""
        return (self.fee_bps + self.slippage_bps) / 10_000.0

    @classmethod
    def from_config(cls, config) -> "BacktestConfig":
        section = config["backtest"]
        return cls(
            fee_bps=float(section["fee_bps"]),
            slippage_bps=float(section["slippage_bps"]),
            execution_lag_days=int(section["execution_lag_days"]),
            initial_equity=float(section["initial_equity"]),
            periods_per_year=int(section["periods_per_year"]),
        )


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.Series
    net_returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    metrics: dict = field(default_factory=dict)
    name: str = "strategia"

    def summary(self) -> str:
        m = self.metrics
        return (
            f"{self.name}: zwrot {m['total_return']:+.1%} | CAGR {m['cagr']:+.1%} "
            f"| Sharpe {m['sharpe']:.2f} | maxDD {m['max_drawdown']:.1%} "
            f"| ekspozycja {m['time_in_market']:.0%} | koszty {m['total_cost']:.1%}"
        )


def _annualize_return(equity: pd.Series, periods_per_year: int) -> float:
    if len(equity) < 2:
        return float("nan")
    years = len(equity) / periods_per_year
    if years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min())


def compute_metrics(
    equity: pd.Series,
    net_returns: pd.Series,
    positions: pd.Series,
    costs: pd.Series,
    *,
    periods_per_year: int = 365,
) -> dict:
    """Metryki ryzyka i zwrotu. Sam zwrot calkowity nic nie mowi o drodze."""
    returns = net_returns.dropna()
    if returns.empty:
        return {}

    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    mean_annual = float(returns.mean() * periods_per_year)
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else np.nan
    drawdown = max_drawdown(equity)
    active = returns[positions.reindex(returns.index).fillna(0) != 0]
    turnover = float(positions.diff().abs().sum())

    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr": _annualize_return(equity, periods_per_year),
        "volatility": volatility,
        "sharpe": float(mean_annual / volatility) if volatility > 0 else np.nan,
        "sortino": float(mean_annual / downside_vol) if downside_vol and downside_vol > 0 else np.nan,
        "max_drawdown": drawdown,
        "calmar": float(_annualize_return(equity, periods_per_year) / abs(drawdown))
        if drawdown < 0
        else np.nan,
        "win_rate": float((active > 0).mean()) if len(active) else np.nan,
        "days": int(len(returns)),
        "time_in_market": float((positions.reindex(returns.index).fillna(0) != 0).mean()),
        "average_position": float(positions.reindex(returns.index).fillna(0).mean()),
        "turnover_total": turnover,
        "turnover_annual": float(turnover / (len(returns) / periods_per_year))
        if len(returns)
        else np.nan,
        "n_position_changes": int((positions.diff().abs() > 1e-12).sum()),
        "total_cost": float(costs.sum()),
    }


def run_backtest(
    prices: pd.DataFrame | pd.Series,
    signal: pd.Series,
    config: BacktestConfig | None = None,
    *,
    name: str = "strategia",
    price_column: str = "close",
) -> BacktestResult:
    """Uruchamia strategie na dziennych zamknieciach.

    `signal` to DOCELOWA pozycja (0..1, lub -1..1 przy allow_short) znana na
    zamknieciu danego dnia. Silnik sam naklada opoznienie wykonania - nie
    przesuwaj sygnalu recznie, bo przesuniesz go dwa razy.
    """
    config = config or BacktestConfig()

    if isinstance(prices, pd.DataFrame):
        frame = prices.copy()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            frame = frame.drop_duplicates(subset="date", keep="last").set_index("date")
        close = frame[price_column].astype(float)
    else:
        close = prices.astype(float)
    close = close.sort_index()

    target = signal.reindex(close.index).astype(float).fillna(0.0)
    lower = -config.max_leverage if config.allow_short else 0.0
    target = target.clip(lower, config.max_leverage)

    positions = target.shift(config.execution_lag_days).fillna(0.0)
    asset_returns = close.pct_change().fillna(0.0)

    gross = positions * asset_returns
    turnover = positions.diff().abs().fillna(positions.abs())
    costs = turnover * config.cost_rate
    net = gross - costs

    equity = config.initial_equity * (1.0 + net).cumprod()
    metrics = compute_metrics(
        equity, net, positions, costs, periods_per_year=config.periods_per_year
    )
    return BacktestResult(
        equity=equity,
        positions=positions,
        net_returns=net,
        gross_returns=gross,
        costs=costs,
        metrics=metrics,
        name=name,
    )


def compare(results: list[BacktestResult]) -> pd.DataFrame:
    """Tabela porownawcza. Baseline zawsze na liscie - inaczej nie ma z czym porownac."""
    rows = []
    for result in results:
        row = {"strategia": result.name}
        row.update(result.metrics)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("strategia")
    preferred = [
        "total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar",
        "win_rate", "time_in_market", "turnover_annual", "total_cost", "days",
    ]
    ordered = [c for c in preferred if c in table.columns]
    return table.loc[:, ordered + [c for c in table.columns if c not in ordered]]


def excess_over_baseline(strategy: BacktestResult, baseline: BacktestResult) -> dict:
    """Roznica wzgledem baseline - jedyna liczba, ktora naprawde interesuje.

    Strategia, ktora zarabia mniej niz kup-i-trzymaj przy wiekszym ryzyku,
    jest gorsza nawet gdy jej zwrot jest dodatni.
    """
    return {
        "return_difference": strategy.metrics["total_return"] - baseline.metrics["total_return"],
        "cagr_difference": strategy.metrics["cagr"] - baseline.metrics["cagr"],
        "sharpe_difference": strategy.metrics["sharpe"] - baseline.metrics["sharpe"],
        "drawdown_difference": strategy.metrics["max_drawdown"] - baseline.metrics["max_drawdown"],
        "beats_baseline_return": strategy.metrics["total_return"] > baseline.metrics["total_return"],
        "beats_baseline_sharpe": strategy.metrics["sharpe"] > baseline.metrics["sharpe"],
    }
