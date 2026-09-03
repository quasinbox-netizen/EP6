"""Walk-forward evaluation of the forecast - the part that decides anything.

For every fold: fit on the training window, predict on the test window, score
against the baselines. Nothing is ever fitted on data the prediction could not
have seen, and the embargo between the sets covers the target horizon so the
last training labels do not reach into the test window.

The aggregate is deliberately conservative. Predictions from all folds are
pooled and scored once, on non-overlapping rows only, because that is the
number that answers "would this have worked". Per-fold results are reported
alongside so a single lucky window cannot hide inside the average.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forecast.baselines import all_baselines
from forecast.evaluate import (
    brier_score,
    calibration_table,
    compare_against_baselines,
    thin_to_non_overlapping,
    verdict,
)
from forecast.model import RidgeLogistic
from validation.splits import Split, assert_no_overlap, split_frame

# Penalty strengths tried when alpha is selected rather than fixed. Spans four
# orders of magnitude because the right value depends on how many rows the
# expanding training window happens to hold.
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)

# Share of the training window held back, chronologically, to choose alpha.
INNER_VALIDATION_SHARE = 0.25


@dataclass
class ForecastRun:
    horizon: int
    folds: pd.DataFrame
    pooled: pd.DataFrame
    pooled_all_rows: pd.DataFrame
    calibration: pd.DataFrame
    predictions: pd.DataFrame
    weights: pd.Series = field(default_factory=pd.Series)
    n_folds: int = 0

    def summary(self) -> str:
        return verdict(self.pooled)


def select_alpha(
    train_frame: pd.DataFrame,
    train_target: pd.Series,
    *,
    grid=ALPHA_GRID,
    horizon: int = 30,
) -> float:
    """Choose the penalty on an inner split of the TRAINING window only.

    This exists because a fixed alpha is not a fair test of the question. With
    60-odd standardised features, alpha=1 is effectively no penalty at all: the
    fit chases collinear directions, produces probabilities near 0 and 1, and
    scores worse than a coin flip. Reporting "no edge" from a model tuned that
    badly would prove nothing about the data.

    The inner split is chronological and the last `horizon` rows of the inner
    training part are dropped, so the labels used to pick alpha do not overlap
    the labels used to score it. Alpha is never chosen on the outer test
    window - that would be the classic way to manufacture an edge.
    """
    index = train_frame.index.sort_values()
    cut = int(len(index) * (1 - INNER_VALIDATION_SHARE))
    if cut <= horizon + 30 or cut >= len(index):
        return 1.0

    inner_train = index[: max(cut - horizon, 1)]
    inner_valid = index[cut:]
    y_train = train_target.reindex(inner_train)
    y_valid = train_target.reindex(inner_valid)
    if y_train.dropna().nunique() < 2 or y_valid.dropna().empty:
        return 1.0

    best_alpha, best_brier = 1.0, float("inf")
    for candidate in grid:
        try:
            model = RidgeLogistic(alpha=candidate).fit(
                train_frame.loc[inner_train], y_train
            )
        except ValueError:
            continue
        probability = model.predict_proba(train_frame.loc[inner_valid])
        brier = brier_score(y_valid, probability)
        if np.isfinite(brier) and brier < best_brier:
            best_alpha, best_brier = candidate, brier
    return best_alpha


def target_column(horizon: int) -> str:
    return f"fwd_return_{horizon}d"


def direction_target(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """1 when the forward return over `horizon` days is positive, else 0."""
    column = target_column(horizon)
    if column not in frame.columns:
        raise KeyError(
            f"no column {column} - add_forward_returns must be called with "
            f"horizon {horizon}"
        )
    forward = frame[column]
    return (forward > 0).astype(float).where(forward.notna()).rename("direction")


def run_walk_forward(
    frame: pd.DataFrame,
    splits: list[Split],
    *,
    horizon: int = 30,
    alpha: float | None = None,
) -> ForecastRun:
    """Fit and score the forecast across every fold.

    `alpha=None` (the default) selects the penalty per fold on an inner split
    of that fold's training window. Passing a number fixes it, which is useful
    in tests but not a fair test of the data.
    """
    target = direction_target(frame, horizon)
    fold_rows = []
    prediction_frames = []
    last_weights = pd.Series(dtype=float)

    for split in splits:
        # The embargo must cover the horizon, otherwise training labels contain
        # test-window prices. horizon - 1 because the check is strict.
        assert_no_overlap(split, horizon_days=horizon - 1)
        train_frame, test_frame = split_frame(frame, split)
        train_target = target.reindex(train_frame.index)
        test_target = target.reindex(test_frame.index)
        if train_target.dropna().nunique() < 2 or test_target.dropna().empty:
            continue

        try:
            chosen_alpha = (
                select_alpha(train_frame, train_target) if alpha is None else alpha
            )
            model = RidgeLogistic(alpha=chosen_alpha).fit(train_frame, train_target)
        except ValueError:
            continue
        last_weights = model.weights()

        probability = model.predict_proba(test_frame)
        baselines = all_baselines(train_frame, train_target, test_frame)

        table = compare_against_baselines(test_target, probability, baselines)
        row = {"fold": split.name, "n_train": model.n_train,
               "alpha": chosen_alpha, "converged": model.converged}
        row.update(table.loc["model"].to_dict())
        fold_rows.append(row)

        part = pd.DataFrame({"target": test_target, "model": probability})
        for name, values in baselines.items():
            part[name] = values
        part["fold"] = split.name
        prediction_frames.append(part)

    if not prediction_frames:
        empty = pd.DataFrame()
        return ForecastRun(horizon, empty, empty, empty, empty, empty, last_weights, 0)

    predictions = pd.concat(prediction_frames).sort_index()
    baseline_names = [c for c in predictions.columns
                      if c not in ("target", "model", "fold")]

    # Non-overlapping rows are the honest sample; every row is reported too so
    # the difference between the two is visible rather than hidden.
    thinned = thin_to_non_overlapping(predictions.index, horizon)
    pooled = compare_against_baselines(
        predictions.loc[thinned, "target"],
        predictions.loc[thinned, "model"],
        {name: predictions.loc[thinned, name] for name in baseline_names},
    )
    pooled_all = compare_against_baselines(
        predictions["target"],
        predictions["model"],
        {name: predictions[name] for name in baseline_names},
    )

    return ForecastRun(
        horizon=horizon,
        folds=pd.DataFrame(fold_rows),
        pooled=pooled,
        pooled_all_rows=pooled_all,
        calibration=calibration_table(
            predictions.loc[thinned, "target"], predictions.loc[thinned, "model"]
        ),
        predictions=predictions,
        weights=last_weights,
        n_folds=len(fold_rows),
    )


def latest_prediction(
    frame: pd.DataFrame, *, horizon: int = 30, alpha: float | None = None
) -> dict:
    """Fit on everything with a known label and predict the most recent day.

    This is the closest thing to a live forecast the project produces, and it
    must be read with the walk-forward verdict next to it: if the model has no
    edge out of sample, this number is decoration.
    """
    target = direction_target(frame, horizon)
    labelled = target.dropna().index
    if len(labelled) == 0:
        return {"error": "no labelled rows"}

    train_frame = frame.loc[labelled]
    if alpha is None:
        alpha = select_alpha(train_frame, target.reindex(labelled), horizon=horizon)
    model = RidgeLogistic(alpha=alpha).fit(train_frame, target.reindex(labelled))
    probability = model.predict_proba(frame)
    usable = probability.dropna()
    if usable.empty:
        return {"error": "no row with complete features"}

    day = usable.index.max()
    return {
        "as_of": day,
        "horizon_days": horizon,
        "probability_up": float(usable.loc[day]),
        "train_base_rate": model.base_rate,
        "n_train": model.n_train,
        "alpha": float(alpha),
        # The base rate is the number to compare against: a probability equal
        # to it carries no information beyond "this asset usually went up".
        "edge_over_base_rate": float(usable.loc[day] - model.base_rate),
    }
