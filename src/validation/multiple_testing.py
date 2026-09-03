"""Correction for multiple testing.

This is the problem the project exists to handle. With a dozen halving windows,
several event categories and four horizons, we test hundreds of hypotheses. Run
100 tests on pure noise and about five come back "significant" at 0.05 - and it
is those five that end up as headlines about a cyclical pattern.

Two corrections, two philosophies:

* Bonferroni controls the probability of making EVEN ONE type I error (FWER).
  Conservative; the right choice when a single false conclusion is expensive,
  for instance when you intend to put money behind it.
* Benjamini-Hochberg controls the PROPORTION of false discoveries among the
  rejections (FDR). Gentler; the right choice while generating hypotheses for
  further study.

Honesty requires counting every test you ran, not only the ones that made it
into the report. That is why `correct` takes the whole scan table, and why
`n_tests_override` exists for tests you discarded along the way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("bonferroni", "bh", "none")


def _valid_mask(p: np.ndarray) -> np.ndarray:
    """Hypotheses that could not be tested at all, e.g. an empty window.

    They stay in the table as NaN but take no part in the correction: they
    neither inflate the test count nor - more importantly - corrupt the
    monotonicity step of BH, which walks backwards through the sorted list.
    """
    return np.isfinite(p)


def bonferroni(p_values: np.ndarray | pd.Series, n_tests: int | None = None) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    valid = _valid_mask(p)
    n = n_tests or int(valid.sum())
    out = np.full(p.shape, np.nan)
    out[valid] = np.clip(p[valid] * n, 0.0, 1.0)
    return out


def benjamini_hochberg(
    p_values: np.ndarray | pd.Series, n_tests: int | None = None
) -> np.ndarray:
    """BH q-values: monotone and clipped to 1."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = _valid_mask(p)
    if not valid.any():
        return out

    tested = p[valid]
    m = n_tests or int(valid.sum())
    order = np.argsort(tested)
    ranked = tested[order]
    adjusted = ranked * m / (np.arange(1, len(ranked) + 1))
    # Enforce monotonicity from the end, otherwise q can decrease as p rises.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    out[valid] = restored
    return out


def correct(
    scan: pd.DataFrame,
    *,
    method: str = "bh",
    alpha: float = 0.05,
    p_column: str = "p_value",
    n_tests_override: int | None = None,
) -> pd.DataFrame:
    """Attach adjusted p/q values and decisions to a hypothesis scan table."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method} (available: {METHODS})")
    out = scan.copy()
    if out.empty:
        return out
    testable = int(np.isfinite(np.asarray(out[p_column], dtype=float)).sum())
    n_tests = n_tests_override or testable

    if method == "bonferroni":
        out["p_adjusted"] = bonferroni(out[p_column], n_tests)
    elif method == "bh":
        out["p_adjusted"] = benjamini_hochberg(out[p_column], n_tests)
    else:
        out["p_adjusted"] = out[p_column].to_numpy()

    out["significant_raw"] = (out[p_column] < alpha).fillna(False)
    out["significant_adjusted"] = (out["p_adjusted"] < alpha).fillna(False)
    out.attrs["method"] = method
    out.attrs["alpha"] = alpha
    out.attrs["n_tests"] = n_tests
    out.attrs["n_untestable"] = int(len(out) - testable)
    return out.sort_values(p_column)


def expected_false_discoveries(n_tests: int, alpha: float = 0.05) -> float:
    """How many "discoveries" chance alone yields at this many tests."""
    return n_tests * alpha


def summarize(corrected: pd.DataFrame) -> str:
    if corrected.empty:
        return "no hypotheses to summarize"
    n_tests = corrected.attrs.get("n_tests", len(corrected))
    alpha = corrected.attrs.get("alpha", 0.05)
    raw = int(corrected["significant_raw"].sum())
    adjusted = int(corrected["significant_adjusted"].sum())
    untestable = corrected.attrs.get("n_untestable", 0)
    skipped = f" | skipped (no data): {untestable}" if untestable else ""
    return (
        f"{n_tests} hypotheses | significant raw: {raw} "
        f"(chance alone would give ~{expected_false_discoveries(n_tests, alpha):.1f}) "
        f"| after {corrected.attrs.get('method', '?')} correction: {adjusted}{skipped}"
    )
