"""Strategy Reality Check - audit an outside strategy with this project's tests.

The lab was built to interrogate one hypothesis of its own. The machinery it
grew for that - a permutation null that re-times a rule instead of reshuffling
prices, a leave-one-episode-out fragility check, a correction for how many
variants were tried, an out-of-sample split with an embargo - answers a
question nobody else was asking about their own backtests.

This package points that machinery at a CSV the user brings in.

Nothing here generates a signal, recommends a position, or forecasts a price.
It takes a track record that already exists and reports how much of it could
be chance. That distinction is deliberate and load-bearing: see LEGAL.md.
"""
from __future__ import annotations

from .loader import AuditInput, LoadError, load_csv  # noqa: F401
from .report import AuditReport, Check, run_audit  # noqa: F401

__all__ = [
    "AuditInput",
    "AuditReport",
    "Check",
    "LoadError",
    "load_csv",
    "run_audit",
]
