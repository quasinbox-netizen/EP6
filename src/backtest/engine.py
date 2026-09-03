"""Backtest engine: daily, vectorised, with costs and slippage.

The execution model, stated explicitly because this is where most silent
errors live:

* the signal for day t is formed from information known AT THE CLOSE of day t,
* the return of day t is the change from the close of t-1 to the close of t,
* with `execution_lag_days = 1` the signal from day t earns the return of
  day t+1 - you enter at the close you just observed and only capture the next
  day's move.

Zero lag would mean the signal from day t earns the return of day t - trading
at a price that has not settled yet. That is the most common reason a backtest
"works" only in a spreadsheet; test_execution_lag_blocks_same_day_knowledge
shows the difference in numbers.

Consequence for the baseline: with no costs, buy-and-hold returns exactly what
the asset returned - the first day of the sample is spent placing the order.

Cost is charged on TURNOVER (|change in position|), not per trade: going from
0 to 1 costs the same as going from 1 to 0, and a change from 0.5 to 0.6 costs
a tenth of that. Fees and slippage are separate parameters because they scale
differently - you can negotiate fees, you cannot negotiate slippage.
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
        """One-way cost as a fraction of turnover."""
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
    name: str = "strategy"

    def summary(self) -> str:
        m = self.metrics
        return (
            f"{self.name}: return {m['total_return']:+.1%} | CAGR {m['cagr']:+.1%} "
            f"| Sharpe {m['sharpe']:.2f} | maxDD {m['max_drawdown']:.1%} "
            f"| exposure {m['time_in_market']:.0%} | costs {m['total_cost']:.1%}"
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
    """Risk and return metrics. Total return alone says nothing about the path."""
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
    name: str = "strategy",
    price_column: str = "close",
) -> BacktestResult:
    """Run a strategy on daily closes.

    `signal` is the TARGET position (0..1, or -1..1 with allow_short) known at
    the close of that day. The engine applies the execution lag itself - do not
    shift the signal by hand or you will shift it twice.
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
    """Comparison table. Always include the baseline - otherwise there is nothing to compare to."""
    rows = []
    for result in results:
        row = {"strategy": result.name}
        row.update(result.metrics)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("strategy")
    preferred = [
        "total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar",
        "win_rate", "time_in_market", "turnover_annual", "total_cost", "days",
    ]
    ordered = [c for c in preferred if c in table.columns]
    return table.loc[:, ordered + [c for c in table.columns if c not in ordered]]


def excess_over_baseline(strategy: BacktestResult, baseline: BacktestResult) -> dict:
    """Difference against the baseline - the only number that really matters.

    A strategy that earns less than buy-and-hold at greater risk is worse, even
    when its return is positive.
    """
    return {
        "return_difference": strategy.metrics["total_return"] - baseline.metrics["total_return"],
        "cagr_difference": strategy.metrics["cagr"] - baseline.metrics["cagr"],
        "sharpe_difference": strategy.metrics["sharpe"] - baseline.metrics["sharpe"],
        "drawdown_difference": strategy.metrics["max_drawdown"] - baseline.metrics["max_drawdown"],
        "beats_baseline_return": strategy.metrics["total_return"] > baseline.metrics["total_return"],
        "beats_baseline_sharpe": strategy.metrics["sharpe"] > baseline.metrics["sharpe"],
    }
