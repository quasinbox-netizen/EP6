"""Baselines a forecast has to beat before it means anything.

This module is the point of the whole forecast package. Bitcoin rose in
roughly 55% of historical 30-day windows, so a model reporting 58% accuracy
has demonstrated almost nothing - and that is how most "BTC prediction models"
are presented: against an implied 50% that nobody actually competes with.

Three baselines, each answering a different objection:

* `always_up`  - "the asset just goes up". Uses the base rate from the
                 TRAINING window, never the test window; using the test base
                 rate would hand the baseline information the model does not
                 have.
* `coin_flip`  - a constant 0.5. The reference point for the Brier score.
* `momentum`   - "the last month continues". The cheapest real predictor, and
                 the one a cyclical story most needs to beat: if a halving
                 model only reproduces momentum, it is measuring trend.

Every baseline returns calibrated probabilities, not labels, so it can be
scored with exactly the same metrics as the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_COLUMN = "return_30d"


def always_up(train_target: pd.Series, index: pd.Index) -> pd.Series:
    """Constant probability equal to the training-window base rate."""
    rate = float(train_target.dropna().mean())
    return pd.Series(rate, index=index, name="always_up")


def coin_flip(index: pd.Index) -> pd.Series:
    """Constant 0.5 - no information at all."""
    return pd.Series(0.5, index=index, name="coin_flip")


def momentum(
    train_frame: pd.DataFrame,
    train_target: pd.Series,
    test_frame: pd.DataFrame,
    *,
    column: str = MOMENTUM_COLUMN,
) -> pd.Series:
    """Probability conditioned on the sign of the trailing return.

    Calibrated on the training window: we measure how often the forward return
    was positive after a positive trailing month and after a negative one, then
    apply those two rates to the test rows. That keeps it a genuine competitor
    rather than a straw man - it gets the same training data the model gets.
    """
    if column not in train_frame.columns or column not in test_frame.columns:
        return pd.Series(np.nan, index=test_frame.index, name="momentum")

    aligned = pd.DataFrame(
        {"signal": train_frame[column], "target": train_target}
    ).dropna()
    if aligned.empty:
        return pd.Series(np.nan, index=test_frame.index, name="momentum")

    positive = aligned.loc[aligned["signal"] > 0, "target"]
    negative = aligned.loc[aligned["signal"] <= 0, "target"]
    overall = float(aligned["target"].mean())
    rate_positive = float(positive.mean()) if len(positive) else overall
    rate_negative = float(negative.mean()) if len(negative) else overall

    signal = test_frame[column]
    out = pd.Series(np.nan, index=test_frame.index, name="momentum")
    out[signal > 0] = rate_positive
    out[signal <= 0] = rate_negative
    return out


def all_baselines(
    train_frame: pd.DataFrame,
    train_target: pd.Series,
    test_frame: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Every baseline, keyed by name, aligned to the test rows."""
    return {
        "always_up": always_up(train_target, test_frame.index),
        "coin_flip": coin_flip(test_frame.index),
        "momentum": momentum(train_frame, train_target, test_frame),
    }
