"""A virtual portfolio, run in public, so the claim can be checked by anyone.

Every backtest in this project is written by someone who has already seen the
outcome. A portfolio that starts on a stated date, follows a stated rule and
publishes every trade is the version of that claim which cannot be tuned
afterwards - the same reason the prediction ledger exists, applied to a
position rather than an interval.

What decides:

* DIRECTION comes from a mechanical rule - by default the 50/200 crossover.
  The rest of this project exists to show that no timing rule here beats its
  own random-timing version, and this one is no exception. It is not included
  because it works; it is included because it is transparent, and because a
  portfolio needs some rule to be a portfolio at all.
* SIZE comes from the volatility forecast, which is the one component with
  demonstrated skill: target volatility divided by forecast volatility, with
  a rebalance band so it does not retrade on every wobble.

The two answer different questions. Sizing says how much, and can be right.
Direction says which way, and on this data cannot. Publishing them together
without saying which is which would be the dishonest version of this page.

Costs are charged on every change in weight. A paper portfolio that trades
for free is a marketing document.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Below this, a change in weight is drift rather than a decision, and printing
# it as a trade would bury the entries and exits people came to read.
TRADE_THRESHOLD = 0.02


@dataclass
class PaperRun:
    equity: pd.Series
    weights: pd.Series
    hold: pd.Series          # buy-and-hold on the same capital, for comparison
    trades: pd.DataFrame
    capital: float
    cost_rate: float
    # The close the portfolio was last valued at; a live quote replaces it in
    # the browser to revalue the open position.
    last_price: float = 0.0

    @property
    def current_weight(self) -> float:
        return float(self.weights.iloc[-1]) if len(self.weights) else 0.0

    @property
    def units(self) -> float:
        """Coins held right now, valued at the last close."""
        if self.equity.empty or self.last_price <= 0:
            return 0.0
        return float(self.equity.iloc[-1] * self.current_weight / self.last_price)

    def summary(self) -> str:
        if self.equity.empty:
            return "no history"
        total = self.equity.iloc[-1] / self.capital - 1
        bench = self.hold.iloc[-1] / self.capital - 1
        return (
            f"{self.equity.index[0]:%Y-%m-%d} to {self.equity.index[-1]:%Y-%m-%d}: "
            f"{total:+.1%} against {bench:+.1%} for holding, "
            f"{len(self.trades)} trades"
        )


def target_weights(
    direction: pd.Series,
    size: pd.Series,
    *,
    max_weight: float = 1.0,
) -> pd.Series:
    """How much of the portfolio should be in BTC on each day.

    Direction is 0 or 1 and decides whether to be in at all; size decides how
    much when in. Multiplying rather than choosing one of them is the point:
    a rule with no edge cannot make the position bigger, only present or
    absent.
    """
    combined = direction.astype(float).clip(0.0, 1.0) * size.reindex(direction.index).fillna(0.0)
    return combined.clip(0.0, max_weight).rename("weight")


def run_paper(
    close: pd.Series,
    weights: pd.Series,
    *,
    capital: float = 10_000.0,
    cost_rate: float = 0.0025,
    start: pd.Timestamp | None = None,
) -> PaperRun:
    """Walk the portfolio day by day, charging for every change in weight.

    `weights` must already be executable on the day they are indexed - the
    engine's lag convention, not the signal's. Feeding tomorrow's weight into
    today would hand the portfolio a return it could not have earned, which is
    the single most common way a paper account beats the market.
    """
    close = close.astype(float).sort_index()
    weights = weights.reindex(close.index).fillna(0.0).clip(0.0, 1.0)
    if start is not None:
        close = close.loc[close.index >= pd.Timestamp(start)]
        weights = weights.loc[close.index]
    if close.empty:
        empty = pd.Series(dtype=float)
        return PaperRun(empty, empty, empty, _empty_trades(), capital, cost_rate, 0.0)

    returns = close.pct_change().fillna(0.0)
    equity = np.empty(len(close))
    value = capital
    previous = 0.0
    rows = []

    for i, (day, weight) in enumerate(zip(close.index, weights.to_numpy())):
        # The day's return applies to the weight held coming into it.
        value *= 1.0 + previous * returns.iloc[i]
        change = weight - previous
        if abs(change) > 1e-12:
            value -= value * abs(change) * cost_rate
        if abs(change) > TRADE_THRESHOLD:
            rows.append({
                "date": day,
                "action": "buy" if change > 0 else "sell",
                "price": float(close.iloc[i]),
                "weight_from": round(previous, 4),
                "weight_to": round(float(weight), 4),
                "cost": round(value * abs(change) * cost_rate, 2),
                "equity": round(value, 2),
            })
        previous = float(weight)
        equity[i] = value

    curve = pd.Series(equity, index=close.index, name="equity")
    hold = capital * (1.0 + returns).cumprod()
    hold.name = "buy_and_hold"
    trades = pd.DataFrame(rows) if rows else _empty_trades()
    return PaperRun(curve, weights, hold, trades, capital, cost_rate,
                    float(close.iloc[-1]))


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "action", "price", "weight_from", "weight_to", "cost", "equity"]
    )


def state(run: PaperRun, *, flip_level: float | None, flip_text: str = "") -> dict:
    """What the page shows, and what the browser needs to mark it to market.

    `units` and `cash` are what a live price is applied to. Everything else is
    settled history, so a visitor watching the number move is watching one
    position revalue - not a portfolio quietly trading while nobody looks.
    """
    if run.equity.empty:
        return {"started": None, "equity": 0.0, "weight": 0.0, "units": 0.0, "cash": 0.0}

    last_equity = float(run.equity.iloc[-1])
    weight = run.current_weight
    entry = None
    if weight > 0 and not run.trades.empty:
        # The most recent move into the position, for a cost basis a reader
        # can check against the trade list.
        buys = run.trades[run.trades["action"] == "buy"]
        if not buys.empty:
            entry = float(buys.iloc[-1]["price"])

    return {
        "started": run.equity.index[0].strftime("%Y-%m-%d"),
        "asOf": run.equity.index[-1].strftime("%Y-%m-%d"),
        "capital": run.capital,
        "equity": last_equity,
        "hold": float(run.hold.iloc[-1]),
        "weight": weight,
        "lastPrice": run.last_price,
        # Coins held, so a live quote can revalue the position in the browser.
        "units": run.units,
        "cash": last_equity * (1.0 - weight),
        "entryPrice": entry,
        "trades": len(run.trades),
        "flipLevel": flip_level,
        "flipText": flip_text,
        "costRate": run.cost_rate,
    }
