"""Running every check and turning them into one verdict.

The verdict is deliberately hard to pass and easy to read. Three grades:

  SURVIVES   every blocking check passed, nothing was skipped for want of data
  UNPROVEN   nothing failed, but something could not be tested or came back
             weak - the common outcome, and not an accusation
  REJECTED   at least one blocking check failed

There is no numeric score. A score invites the user to tune the strategy until
the number goes up, which is the exact behaviour - searching until something
passes - that the multiplicity check exists to punish. A verdict plus the list
of what failed cannot be optimised against in the same way.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import checks as C
from .checks import FAIL, NA, PASS, WARN, Check  # noqa: F401  (re-exported)
from .license import EVAL_PERMUTATIONS, Licence

SURVIVES = "SURVIVES"
UNPROVEN = "UNPROVEN"
REJECTED = "REJECTED"

DEFAULT_PERMUTATIONS = 2000
SCHEMA_VERSION = 1


@dataclass
class AuditReport:
    verdict: str
    summary: str
    checks: list[Check]
    meta: dict = field(default_factory=dict)

    def by_key(self, key: str) -> Check | None:
        for check in self.checks:
            if check.key == key:
                return check
        return None

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.blocking and c.verdict == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.blocking and c.verdict == WARN]

    @property
    def untested(self) -> list[Check]:
        return [c for c in self.checks if c.blocking and c.verdict == NA]

    def as_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "verdict": self.verdict,
            "summary": self.summary,
            "meta": self.meta,
            "checks": [check.as_dict() for check in self.checks],
        }

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, default=str), encoding="utf-8")
        return path


def _decide(blocking: list[Check]) -> tuple[str, str]:
    failed = [c for c in blocking if c.verdict == FAIL]
    warned = [c for c in blocking if c.verdict == WARN]
    skipped = [c for c in blocking if c.verdict == NA]

    if failed:
        names = ", ".join(c.title.lower() for c in failed)
        return REJECTED, (
            f"{len(failed)} of {len(blocking)} checks failed ({names}). At least one "
            "result in this record is better explained by chance, by costs, or by a "
            "handful of days than by the rule."
        )
    if warned or skipped:
        parts = []
        if warned:
            parts.append(f"{len(warned)} came back weak")
        if skipped:
            parts.append(f"{len(skipped)} could not be run on the data provided")
        return UNPROVEN, (
            "Nothing failed outright, but " + " and ".join(parts) + ". This is the "
            "usual outcome and it is not a rejection: it means the record is "
            "consistent with a real edge and also consistent with luck, and this "
            "file cannot tell them apart."
        )
    return SURVIVES, (
        f"All {len(blocking)} checks passed. The record survives re-timing against "
        "chance, the removal of its best trade, the correction for how many variants "
        "were tried, a split of its own history, and realistic costs. That is as far "
        "as a backtest can go; it is not a statement about the future."
    )


def run_audit(
    data,
    *,
    licence: Licence | None = None,
    variants_tried: int | None = None,
    cost_bps: float = 0.0,
    permutations: int = DEFAULT_PERMUTATIONS,
    label: str = "",
) -> AuditReport:
    """Run every check against a loaded track record.

    `licence` only ever narrows what runs: without one the permutation count
    drops. Nothing in the report is falsified for unlicensed users - a
    weakened test is stated as weakened, because a tool whose free tier lies
    is worse than no free tier.
    """
    licence = licence or Licence(valid=False, reason="no licence supplied.")
    if not licence.valid:
        permutations = min(permutations, EVAL_PERMUTATIONS)

    periods = C.infer_periods_per_year(data.net_returns.index)
    cost_rate = float(cost_bps) / 10_000.0

    results = [
        C.describe(data, periods),
        C.retiming(data, permutations=permutations, cost_rate=cost_rate,
                   periods_per_year=periods),
        C.fragility_check(data, permutations=permutations, cost_rate=cost_rate,
                          periods_per_year=periods),
    ]

    # Which p-value the multiplicity correction is applied to. The re-timing
    # p-value when it exists; otherwise there is no hypothesis test to correct
    # and the check says so rather than correcting a bootstrap interval, which
    # is not a p-value and would be nonsense dressed as rigour.
    retimed = results[1]
    primary_p = float(retimed.numbers.get("p_value", np.nan))
    results.append(C.multiplicity(primary_p, variants_tried=variants_tried))

    results.extend([
        C.bootstrap_interval(data, periods_per_year=periods),
        C.out_of_sample(data, periods_per_year=periods),
        C.cost_sensitivity(data, cost_rate=cost_rate, applied_bps=float(cost_bps)),
        C.versus_holding(data, periods_per_year=periods),
        C.concentration(data),
        C.survivability(data, periods_per_year=periods),
    ])

    blocking = [c for c in results if c.blocking]
    verdict, summary = _decide(blocking)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": data.source,
        "label": label or data.source,
        "input_form": data.form,
        "position_convention": data.convention,
        "periods_per_year": periods,
        "cost_bps_applied": float(cost_bps),
        "permutations": permutations,
        "variants_declared": variants_tried,
        "licence_mode": licence.mode,
        "licensee": licence.licensee,
        "observations": int(len(data.net_returns.dropna())),
        "first_day": str(data.span[0].date()),
        "last_day": str(data.span[1].date()),
        "notes": list(data.notes),
        "tool": "Strategy Reality Check",
        "python": platform.python_version(),
        "counts": {
            "passed": sum(1 for c in blocking if c.verdict == PASS),
            "failed": sum(1 for c in blocking if c.verdict == FAIL),
            "weak": sum(1 for c in blocking if c.verdict == WARN),
            "not_run": sum(1 for c in blocking if c.verdict == NA),
            "blocking_total": len(blocking),
        },
    }
    return AuditReport(verdict=verdict, summary=summary, checks=results, meta=meta)
