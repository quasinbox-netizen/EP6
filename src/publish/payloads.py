"""Shaping results into the JSON the HTML panels read.

The panels take a JSON blob and draw it. Building that blob is
pixels-and-labels work, not research: it cuts a window, thins a series so a
canvas is not asked to draw five thousand segments, and turns dates into
array positions. Every number passing through was computed by src/ before it
got here.

Shared by the dashboard and the static site for the same reason as the
charts: two copies would drift, and the drift would be invisible.
"""
from __future__ import annotations

import pandas as pd

# Five years of daily closes. Two fitted a panel more comfortably, but the
# rules trade rarely - the trend crossover produced four signals in two years -
# and a replay of four markers is over before it reads as a sequence.
DESK_WINDOW_DAYS = 1825


def desk_payload(bundle: dict) -> dict:
    """Reshape the signal bundle for the panel. No arithmetic on results.

    Everything here is pixels-and-labels work: cut the window, thin it out so
    the canvas is not asked to draw 5,000 segments, and translate dates into
    array positions. Every number passed through is one the pipeline already
    computed.
    """
    close = bundle["price"]
    window = close.iloc[-DESK_WINDOW_DAYS:]
    step = max(1, len(window) // 700)
    shown = window.iloc[::step]
    if shown.index[-1] != window.index[-1]:  # never drop the latest close
        shown = pd.concat([shown, window.iloc[[-1]]])

    def slot(timestamp):
        """Nearest drawn point, or None when the date is off the left edge."""
        stamp = pd.Timestamp(timestamp)
        if stamp < shown.index[0]:
            return None
        position = int(shown.index.searchsorted(stamp))
        return min(position, len(shown) - 1)

    strategies = []
    for item in bundle["strategies"]:
        signal = item["report"]
        held = item["positions"].reindex(shown.index).fillna(0.0).abs() > 0
        holds, start = [], None
        for i, flag in enumerate(held.tolist()):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                holds.append([start, i])
                start = None
        if start is not None:
            holds.append([start, len(held) - 1])

        marks = []
        for row in signal.trades.itertuples():
            entry = slot(row.entry_date)
            if entry is not None:
                marks.append({"i": entry, "side": "buy"})
            if not row.open:
                exit_slot = slot(row.exit_date)
                if exit_slot is not None:
                    marks.append({"i": exit_slot, "side": "sell"})

        metrics = signal.metrics
        strategies.append({
            "name": signal.name,
            "side": signal.side,
            "since": None if signal.since is None else signal.since.strftime("%Y-%m-%d"),
            "days": signal.days,
            "unrealized": _clean(signal.unrealized),
            "trigger": signal.trigger.as_dict(),
            "metrics": {
                "sharpe": _clean(metrics.get("sharpe")),
                "cagr": _clean(metrics.get("cagr")),
                "maxDrawdown": _clean(metrics.get("max_drawdown")),
                "totalReturn": _clean(metrics.get("total_return")),
            },
            "excessSharpe": _clean(item["excess_sharpe"]),
            "tradeCount": int(len(signal.trades)),
            "closedCount": signal.closed_trades,
            "winRateNet": _clean(signal.win_rate_net),
            "trades": [
                {
                    "in": row.entry_date.strftime("%Y-%m-%d"),
                    "out": row.exit_date.strftime("%Y-%m-%d"),
                    "net": _clean(row.return_net),
                    "open": bool(row.open),
                }
                for row in signal.trades.itertuples()
            ],
            "marks": marks,
            "holds": holds,
        })

    return {
        "asOf": bundle["as_of"].strftime("%Y-%m-%d"),
        "sampleDays": int(len(close)),
        "lastClose": float(close.iloc[-1]),
        "costRate": float(bundle["cost_rate"]),
        "price": {
            "d": [stamp.strftime("%Y-%m-%d") for stamp in shown.index],
            "c": [round(float(value), 2) for value in shown.tolist()],
        },
        "strategies": strategies,
    }


def _clean(value):
    """JSON has no NaN. Anything unmeasurable becomes null."""
    if value is None:
        return None
    number = float(value)
    return None if number != number or number in (float("inf"), float("-inf")) else number
