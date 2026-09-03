"""Detecting look-ahead bias by the point-in-time method.

The idea: for a chosen day t we build the features TWICE - once with only the
data published up to t, once with the whole history. Row t must come out
identical. If it differs, some feature on day t is using information that did
not exist yet.

This is a stronger test than comparing a truncated frame against the full one:
it also catches features computed over the whole sample (medians, z-scores,
normalisations), which look innocent under plain truncation.
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
    """Compare the newest point-in-time row against the full-history version.

    `build_fn(as_of)` must return a feature frame indexed by date; `as_of=None`
    means the whole history. For day t we take the LAST row that could have
    been computed then (the bar for t only closes at midnight, so usually t-1)
    and compare it with the same day computed from the full history.

    Returns a frame of discrepancies - an empty frame is a pass.
    """
    full = build_fn(None)
    if full.empty:
        raise ValueError("the full feature build returned an empty frame")
    checked = columns or [c for c in full.columns if not c.startswith("fwd_return_")]

    findings = []
    for day in pd.DatetimeIndex(pd.to_datetime(test_dates)).normalize():
        point_in_time = build_fn(day)
        if point_in_time.empty:
            findings.append(
                {
                    "date": day,
                    "column": "<whole row>",
                    "full_value": "present" if day in full.index else "absent",
                    "point_in_time_value": "empty frame",
                }
            )
            continue
        target = point_in_time.index.max()
        if target not in full.index:
            findings.append(
                {
                    "date": target,
                    "column": "<whole row>",
                    "full_value": "absent",
                    "point_in_time_value": "present",
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
                        "point_in_time_value": "<column missing>",
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
    """Test-facing version: raises AssertionError listing the guilty columns."""
    report = pointwise_lookahead_report(build_fn, test_dates, columns=columns)
    if not report.empty:
        culprits = report["column"].value_counts().to_dict()
        raise AssertionError(
            f"look-ahead bias in {len(report)} cells; columns: {culprits}"
        )


def targets_are_forward_only(frame: pd.DataFrame, horizon: int) -> bool:
    """Sanity check on the target: the last `horizon` days must be empty.

    If fwd_return_Nd has a value on the last day of the sample, the target was
    computed backwards or shifted the wrong way.
    """
    column = f"fwd_return_{horizon}d"
    if column not in frame.columns:
        raise KeyError(f"no column {column}")
    return bool(frame[column].tail(horizon).isna().all())
