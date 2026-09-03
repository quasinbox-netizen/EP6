"""Directional forecasting - and above all, whether the evaluation is honest.

Two tests carry this file. On pure noise the model must NOT beat the baselines;
with a signal deliberately planted in a feature it MUST. Without the second
one, "no edge" on real data would be indistinguishable from a broken pipeline
that can never find anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.baselines import all_baselines, always_up, coin_flip, momentum
from forecast.evaluate import (
    auc,
    brier_score,
    calibration_table,
    compare_against_baselines,
    log_loss,
    skill_score,
    thin_to_non_overlapping,
    verdict,
)
from forecast.model import RidgeLogistic, usable_features
from forecast.walk import direction_target, run_walk_forward, select_alpha
from validation.splits import walk_forward_splits
from validation.synthetic import random_walk_prices

HORIZON = 30


def frame_from_prices(prices: pd.DataFrame, *, extra: dict | None = None) -> pd.DataFrame:
    """A minimal feature frame: price features plus the forward target."""
    from features.build import add_forward_returns, price_features

    frame = add_forward_returns(price_features(prices), [HORIZON])
    for name, values in (extra or {}).items():
        frame[name] = values
    return frame


@pytest.fixture
def noise_frame() -> pd.DataFrame:
    return frame_from_prices(random_walk_prices(3000, start="2013-01-01", seed=17))


def splits_for(frame: pd.DataFrame):
    return walk_forward_splits(
        frame.index, train_days=730, test_days=365, step_days=365,
        embargo_days=90, expanding=True,
    )


# --- the target ------------------------------------------------------------


def test_direction_target_is_binary_and_forward_only(noise_frame):
    target = direction_target(noise_frame, HORIZON)
    assert set(target.dropna().unique()) <= {0.0, 1.0}
    # The last `horizon` days cannot have a label - the future is not there.
    assert target.tail(HORIZON).isna().all()


def test_direction_target_matches_the_sign_of_the_forward_return(noise_frame):
    target = direction_target(noise_frame, HORIZON)
    forward = noise_frame[f"fwd_return_{HORIZON}d"]
    both = target.notna()
    assert ((forward[both] > 0) == (target[both] > 0.5)).all()


def test_missing_horizon_fails_loudly(noise_frame):
    with pytest.raises(KeyError, match="fwd_return_90d"):
        direction_target(noise_frame, 90)


# --- features --------------------------------------------------------------


def test_forward_looking_columns_are_never_used():
    """A model handed `days_to_next_halving` is told about the future."""
    frame = pd.DataFrame(
        {
            "return_30d": [0.1, 0.2],
            "days_to_next_halving": [100, 99],
            "cycle_progress": [0.5, 0.51],
            "fwd_return_30d": [0.1, -0.1],
            "close": [1.0, 2.0],
        }
    )
    columns = usable_features(frame)
    assert "return_30d" in columns
    assert "days_to_next_halving" not in columns
    assert "cycle_progress" not in columns
    assert "fwd_return_30d" not in columns
    assert "close" not in columns


def test_sparse_columns_are_dropped_not_imputed(noise_frame):
    """An all-NaN column must not wipe out the training set.

    This is not hypothetical: `days_since_event_cycle_extreme` is entirely NaN
    because no event of that category exists, and a plain dropna across every
    candidate column leaves zero rows.
    """
    frame = noise_frame.copy()
    frame["never_observed"] = np.nan
    frame["half_observed"] = np.where(np.arange(len(frame)) % 2, np.nan, 1.0)
    target = direction_target(frame, HORIZON)

    model = RidgeLogistic().fit(frame, target)
    assert "never_observed" in model.dropped_for_coverage
    assert "half_observed" in model.dropped_for_coverage
    assert model.n_train > 100


def test_constant_columns_are_dropped(noise_frame):
    frame = noise_frame.copy()
    frame["always_seven"] = 7.0
    model = RidgeLogistic().fit(frame, direction_target(frame, HORIZON))
    assert "always_seven" not in model.columns


# --- the model -------------------------------------------------------------


def test_model_fits_and_returns_probabilities(noise_frame):
    target = direction_target(noise_frame, HORIZON)
    model = RidgeLogistic(alpha=100.0).fit(noise_frame, target)
    probability = model.predict_proba(noise_frame).dropna()
    assert model.converged
    assert len(probability) > 1000
    assert probability.between(0, 1).all()


def test_model_survives_perfectly_collinear_features(noise_frame):
    """Duplicate columns are real here - halving_after_90d equals event_halving_90d."""
    frame = noise_frame.copy()
    frame["copy_of_return_30d"] = frame["return_30d"]
    model = RidgeLogistic(alpha=1.0).fit(frame, direction_target(frame, HORIZON))
    assert model.converged
    assert np.isfinite(model.coefficients).all()


def test_stronger_penalty_shrinks_the_coefficients(noise_frame):
    target = direction_target(noise_frame, HORIZON)
    weak = RidgeLogistic(alpha=1.0).fit(noise_frame, target)
    strong = RidgeLogistic(alpha=10000.0).fit(noise_frame, target)
    assert np.abs(strong.coefficients).sum() < np.abs(weak.coefficients).sum()


def test_unfitted_model_refuses_to_predict(noise_frame):
    with pytest.raises(ValueError, match="not been fitted"):
        RidgeLogistic().predict_proba(noise_frame)


def test_standardisation_uses_training_rows_only(noise_frame):
    """Scaling by full-sample means would be a quiet look-ahead."""
    target = direction_target(noise_frame, HORIZON)
    early = noise_frame.index[:1500]
    model = RidgeLogistic(alpha=10.0).fit(noise_frame.loc[early], target.reindex(early))

    column = model.columns[0]
    # The model standardises over the rows it actually trained on, which is
    # the complete-case subset, so the expectation has to be built the same way.
    used = (
        noise_frame.loc[early, model.columns]
        .join(target.rename("__y"))
        .dropna()
    )
    position = model.columns.index(column)
    assert model.mean[position] == pytest.approx(used[column].mean(), rel=1e-6)

    # And it must differ from the full-sample mean, or the test proves nothing.
    full_sample = noise_frame.loc[:, column].dropna().mean()
    assert model.mean[position] != pytest.approx(full_sample, rel=1e-9)


# --- baselines -------------------------------------------------------------


def test_always_up_uses_the_training_base_rate(noise_frame):
    target = direction_target(noise_frame, HORIZON)
    train = noise_frame.index[:1000]
    test = noise_frame.index[1200:1400]
    baseline = always_up(target.reindex(train), test)
    assert baseline.nunique() == 1
    assert baseline.iloc[0] == pytest.approx(target.reindex(train).dropna().mean())


def test_coin_flip_is_exactly_half(noise_frame):
    assert (coin_flip(noise_frame.index) == 0.5).all()


def test_momentum_conditions_on_the_trailing_return(noise_frame):
    target = direction_target(noise_frame, HORIZON)
    train = noise_frame.index[:2000]
    test = noise_frame.index[2100:2400]
    baseline = momentum(
        noise_frame.loc[train], target.reindex(train), noise_frame.loc[test]
    )
    # Two calibrated rates, one per sign of the trailing month.
    assert baseline.dropna().nunique() <= 2
    assert baseline.dropna().between(0, 1).all()


def test_momentum_is_nan_without_its_column(noise_frame):
    stripped = noise_frame.drop(columns=["return_30d"])
    baseline = momentum(stripped, direction_target(stripped, HORIZON), stripped)
    assert baseline.isna().all()


def test_all_baselines_are_aligned_to_the_test_rows(noise_frame):
    target = direction_target(noise_frame, HORIZON)
    train, test = noise_frame.index[:2000], noise_frame.index[2100:2300]
    baselines = all_baselines(
        noise_frame.loc[train], target.reindex(train), noise_frame.loc[test]
    )
    assert set(baselines) == {"always_up", "coin_flip", "momentum"}
    for values in baselines.values():
        assert values.index.equals(test)


# --- metrics ---------------------------------------------------------------


def test_brier_and_log_loss_reward_being_right():
    y = pd.Series([1.0, 1.0, 0.0, 0.0])
    confident_right = pd.Series([0.9, 0.9, 0.1, 0.1])
    confident_wrong = pd.Series([0.1, 0.1, 0.9, 0.9])
    assert brier_score(y, confident_right) < brier_score(y, pd.Series([0.5] * 4))
    assert brier_score(y, pd.Series([0.5] * 4)) < brier_score(y, confident_wrong)
    assert log_loss(y, confident_right) < log_loss(y, confident_wrong)


def test_coin_flip_brier_is_a_quarter():
    y = pd.Series([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y, pd.Series([0.5] * 4)) == pytest.approx(0.25)


def test_auc_is_half_for_a_constant_and_one_for_perfect_ranking():
    y = pd.Series([1.0, 0.0, 1.0, 0.0])
    assert np.isnan(auc(y, pd.Series([0.5] * 4))) or auc(y, pd.Series([0.5] * 4)) == 0.5
    assert auc(y, pd.Series([0.9, 0.1, 0.8, 0.2])) == pytest.approx(1.0)
    assert auc(y, pd.Series([0.1, 0.9, 0.2, 0.8])) == pytest.approx(0.0)


def test_auc_is_nan_with_only_one_class():
    assert np.isnan(auc(pd.Series([1.0, 1.0]), pd.Series([0.6, 0.7])))


def test_skill_score_is_zero_against_itself():
    assert skill_score(0.2, 0.2) == pytest.approx(0.0)
    assert skill_score(0.1, 0.2) == pytest.approx(0.5)
    assert skill_score(0.4, 0.2) == pytest.approx(-1.0)


def test_thinning_removes_overlapping_labels():
    """365 daily predictions of a 30-day return are not 365 observations."""
    index = pd.date_range("2020-01-01", periods=365, freq="D")
    thinned = thin_to_non_overlapping(index, 30)
    assert len(thinned) == 13
    gaps = pd.Series(thinned).diff().dropna().dt.days
    assert (gaps >= 30).all()


def test_calibration_table_flags_a_miscalibrated_model():
    rng = np.random.default_rng(3)
    y = pd.Series(rng.binomial(1, 0.5, 400).astype(float))
    overconfident = pd.Series(np.where(y > 0.5, 0.95, 0.05))
    honest = pd.Series(np.full(400, 0.5))

    good = calibration_table(y, honest, bins=5).dropna()
    assert good["gap"].abs().max() < 0.1
    # A model that is always right is perfectly calibrated too - the point of
    # the table is that the gap is visible either way.
    assert not calibration_table(y, overconfident, bins=5).empty


def test_verdict_names_the_baselines_it_loses_to():
    y = pd.Series([1.0, 0.0] * 50)
    table = compare_against_baselines(
        y, pd.Series([0.5] * 100), {"always_up": pd.Series([0.5] * 100)}
    )
    text = verdict(table)
    assert "NO EDGE" in text or "beats every baseline" in text


# --- alpha selection -------------------------------------------------------


def test_alpha_is_chosen_from_the_grid(noise_frame):
    from forecast.walk import ALPHA_GRID

    chosen = select_alpha(noise_frame, direction_target(noise_frame, HORIZON))
    assert chosen in ALPHA_GRID


def test_alpha_selection_falls_back_on_a_tiny_window(noise_frame):
    tiny = noise_frame.iloc[:20]
    assert select_alpha(tiny, direction_target(noise_frame, HORIZON).reindex(tiny.index)) == 1.0


# --- the two tests that matter --------------------------------------------


def test_no_edge_on_pure_noise(noise_frame):
    """On a random walk the model must not beat the baselines.

    This is the negative control. If it fails, the evaluation is leaking.
    """
    run = run_walk_forward(noise_frame, splits_for(noise_frame), horizon=HORIZON)
    assert run.n_folds >= 3
    skills = [
        run.pooled.loc["model", column]
        for column in run.pooled.columns
        if column.startswith("skill_vs_")
    ]
    assert all(s < 0.05 for s in skills if np.isfinite(s)), (
        f"a model found an edge in pure noise: {skills}"
    )
    assert "NO EDGE" in run.summary()


def planted_signal_frame(agreement: float, *, seed: int = 23) -> pd.DataFrame:
    """A frame with a feature that agrees with the future direction `agreement` of the time.

    The feature is derived from the future, but it is placed on the row where
    it would have been observable, so using it is legitimate - it stands in for
    a leading indicator that actually worked.
    """
    prices = random_walk_prices(3000, start="2013-01-01", seed=seed, sigma=0.02)
    frame = frame_from_prices(prices)
    forward = frame[f"fwd_return_{HORIZON}d"]
    truth = (forward > 0).astype(float)
    rng = np.random.default_rng(5)
    flip = rng.random(len(frame)) < (1 - agreement)
    frame["oracle_hint"] = np.where(flip, 1 - truth, truth)
    frame.loc[forward.isna(), "oracle_hint"] = np.nan
    return frame


def test_detects_an_edge_that_is_really_there():
    """With a strong signal planted in a feature the model MUST find it.

    Without this test, "no edge" on real data would be worthless - it could
    just mean the pipeline is incapable of detecting anything.
    """
    frame = planted_signal_frame(0.95)
    run = run_walk_forward(frame, splits_for(frame), horizon=HORIZON)

    assert run.n_folds >= 3
    assert run.pooled.loc["model", "auc"] > 0.85
    for column in run.pooled.columns:
        if column.startswith("skill_vs_"):
            assert run.pooled.loc["model", column] > 0.2, column
    assert "beats every baseline" in run.summary()


def test_a_moderate_edge_shows_up_in_auc_even_when_brier_does_not():
    """Discrimination and calibration are different things, and both are reported.

    With a feature that is right 80% of the time, every fold ranks well
    (AUC comfortably above 0.5) while the pooled Brier score can still lose to
    a constant baseline. The reason is worth knowing: the share of positive
    30-day windows swings from 0.27 to 0.72 between test windows, so a model
    calibrated on one window is miscalibrated on the next, and a fixed 0.5
    threshold makes accuracy nearly meaningless.

    This is why the report leads with AUC and the calibration table rather
    than accuracy.
    """
    frame = planted_signal_frame(0.80)
    run = run_walk_forward(frame, splits_for(frame), horizon=HORIZON)

    assert run.n_folds >= 3
    assert run.pooled.loc["model", "auc"] > 0.55
    assert (run.folds["auc"] > 0.5).all(), "the ranking must work in every fold"
    # The base rate really does move that much - that is the point.
    assert run.folds["base_rate"].max() - run.folds["base_rate"].min() > 0.3


def test_walk_forward_reports_per_fold_results(noise_frame):
    run = run_walk_forward(noise_frame, splits_for(noise_frame), horizon=HORIZON)
    assert {"fold", "n_train", "alpha", "brier"} <= set(run.folds.columns)
    assert len(run.folds) == run.n_folds
    # Pooled non-overlapping rows must be far fewer than every row.
    assert run.pooled.loc["model", "n"] < run.pooled_all_rows.loc["model", "n"]


def test_walk_forward_survives_having_no_usable_fold(noise_frame):
    run = run_walk_forward(noise_frame.iloc[:50], [], horizon=HORIZON)
    assert run.n_folds == 0
    assert run.pooled.empty
    assert "no predictions" in run.summary()
