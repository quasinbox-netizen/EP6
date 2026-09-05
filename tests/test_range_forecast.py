"""The range forecast, and the proof that it can be wrong.

An interval is a promise, so the tests here are mostly about catching it
breaking that promise. The two that matter most are the recovery test (given
data from a known model, does the fit find it?) and the miscalibration test
(given an interval that is deliberately too narrow, does coverage say so?).
Without the second, a CALIBRATED verdict would mean nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.coverage import (
    coverage_report,
    rolling_intervals,
    thin_to_independent,
    verdict,
)
from forecast.volatility import (
    MAX_DF,
    MIN_DF,
    SCALE,
    fit_garch,
    price_interval,
    simulate_horizon,
)


def garch_series(n=3000, omega=0.05, alpha=0.10, beta=0.85, df=6.0, seed=3):
    """Returns from a known GARCH(1,1)-t, in the units the fitter expects."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_t(df, size=n) / np.sqrt(df / (df - 2.0))
    variance = omega / (1 - alpha - beta)
    out = np.empty(n)
    for t in range(n):
        out[t] = np.sqrt(variance) * shocks[t]
        variance = omega + alpha * out[t] ** 2 + beta * variance
    return pd.Series(out / SCALE)


def prices_from(returns: pd.Series, start=100.0) -> pd.Series:
    index = pd.bdate_range("2010-01-04", periods=len(returns) + 1, freq="D")
    return pd.Series(start * np.exp(np.concatenate([[0.0], returns.cumsum()])), index=index)


# --- the fit ---------------------------------------------------------------


def test_recovers_the_parameters_it_was_generated_from():
    """Power: on data from a known model the fit must find it.

    Loose tolerances on purpose - 3000 points does not pin GARCH parameters
    tightly, and pretending otherwise would make this test fail for the wrong
    reason. What must be recovered is the shape: high persistence, most of it
    in beta, fat tails.
    """
    fit = fit_garch(garch_series(n=3000, alpha=0.10, beta=0.85, df=6.0))
    assert fit.converged
    assert 0.80 < fit.persistence < 0.99, f"persistence {fit.persistence}"
    assert fit.beta > fit.alpha, "the model should put most persistence in beta"
    assert MIN_DF < fit.df < 20, f"df {fit.df} - fat tails were generated"


def test_persistence_stays_inside_the_stationary_region():
    """The reparameterisation must make this impossible to violate.

    An earlier version enforced alpha + beta < 1 by returning a huge value from
    the objective. L-BFGS-B walked straight into the cliff and returned
    alpha + beta = 1.000 with converged=False - an IGARCH fit with no long-run
    variance. Every start must now stay inside by construction.
    """
    for seed in range(4):
        fit = fit_garch(garch_series(n=800, seed=seed))
        assert fit.persistence < 1.0
        assert fit.alpha > 0 and fit.beta > 0
        assert np.isfinite(fit.long_run_variance)


def test_a_calm_series_gets_a_narrower_interval_than_a_violent_one():
    """The whole premise: volatility clusters, so recent calm should show up."""
    calm = fit_garch(garch_series(n=1500, omega=0.01, seed=11))
    wild = fit_garch(garch_series(n=1500, omega=0.50, seed=11))
    calm_width = np.diff(np.quantile(simulate_horizon(calm, 10, n_paths=8000), [0.05, 0.95]))
    wild_width = np.diff(np.quantile(simulate_horizon(wild, 10, n_paths=8000), [0.05, 0.95]))
    assert wild_width > calm_width * 1.5


def test_too_few_observations_is_refused_not_guessed():
    with pytest.raises(ValueError, match="at least 100"):
        fit_garch(pd.Series(np.random.default_rng(0).normal(0, 0.01, 50)))


# --- the simulation --------------------------------------------------------


def test_mean_reversion_breaks_sqrt_of_time_scaling_in_both_directions():
    """Why simulation replaces sqrt(h) scaling.

    Under GARCH the conditional variance decays towards its long-run level, so
    the h-day spread depends on where today sits relative to that level:

        today unusually violent -> variance falls  -> h-day < sqrt(h) x 1-day
        today unusually calm    -> variance rises  -> h-day > sqrt(h) x 1-day

    sqrt(h) is only right when today happens to sit AT the long-run level,
    which is why an earlier version of this test failed: it asserted a gap on a
    fitted series whose final variance happened to land there, and got sqrt(10)
    to within 2%. The state is set explicitly here instead of hoping for it.
    """
    import dataclasses

    base = fit_garch(garch_series(n=2000, seed=5))
    long_run = base.long_run_variance

    def ratio(variance_multiple: float) -> float:
        state = dataclasses.replace(
            base,
            last_variance=long_run * variance_multiple,
            last_shock=np.sqrt(long_run * variance_multiple),
        )
        one = np.std(simulate_horizon(state, 1, n_paths=30000))
        ten = np.std(simulate_horizon(state, 10, n_paths=30000))
        return float(ten / (one * np.sqrt(10)))

    assert ratio(9.0) < 0.97, "after a violent stretch the horizon must scale sub-sqrt"
    assert ratio(0.1) > 1.03, "after a calm stretch it must scale super-sqrt"


def test_intervals_are_nested_and_ordered():
    fit = fit_garch(garch_series(n=1200, seed=8))
    frame = price_interval(fit, 50_000.0, 10, levels=(0.5, 0.68, 0.90, 0.95))
    assert (frame["low"] < frame["high"]).all()
    assert frame["low"].is_monotonic_decreasing, "wider level must have a lower floor"
    assert frame["high"].is_monotonic_increasing, "wider level must have a higher ceiling"


def test_zero_drift_centres_the_interval_on_todays_price():
    """`drift=0.0` must still do exactly what it says.

    It is no longer the default - it failed coverage, missing upward 41 times
    against 19 downward - but it stays available so the comparison between the
    two remains runnable rather than a claim in a docstring.
    """
    fit = fit_garch(garch_series(n=1200, seed=9))
    frame = price_interval(fit, 50_000.0, 10, n_paths=40000, drift=0.0)
    assert frame.attrs["median"] == pytest.approx(50_000.0, rel=0.02)


def test_horizon_must_be_positive():
    fit = fit_garch(garch_series(n=600, seed=2))
    with pytest.raises(ValueError, match="at least 1 day"):
        simulate_horizon(fit, 0)


# --- coverage --------------------------------------------------------------


def test_thinning_removes_the_overlap():
    frame = pd.DataFrame({"x": range(100)})
    thinned = thin_to_independent(frame, 10)
    assert len(thinned) == 10
    assert list(thinned["x"]) == list(range(0, 100, 10))


def test_coverage_detects_an_interval_that_is_too_narrow():
    """Without this, a CALIBRATED verdict would be unfalsifiable.

    Halving every interval must be caught. If the test cannot see a two-fold
    error it cannot see anything, and the gate on `run.py range` would be
    ornamental.
    """
    rng = np.random.default_rng(4)
    n = 600
    realised = rng.normal(0, 0.10, n)
    frame = pd.DataFrame({"realised": realised})
    for level in (0.5, 0.90):
        half = 0.10 * 1.96 * level / 2  # far too narrow at every level
        frame[f"low_{level}"] = -half
        frame[f"high_{level}"] = half
        frame[f"hit_{level}"] = (frame["realised"] >= -half) & (frame["realised"] <= half)

    result = coverage_report(frame, horizon=1, levels=(0.5, 0.90))
    assert not result.calibrated
    assert "NOT CALIBRATED" in verdict(result)


def test_coverage_accepts_an_interval_that_keeps_its_promise():
    rng = np.random.default_rng(6)
    n = 800
    realised = rng.normal(0, 0.10, n)
    frame = pd.DataFrame({"realised": realised})
    for level in (0.5, 0.90):
        edge = 0.10 * abs(np.percentile(rng.normal(0, 1, 200000), 100 * (1 - (1 - level) / 2)))
        frame[f"low_{level}"] = -edge
        frame[f"high_{level}"] = edge
        frame[f"hit_{level}"] = (frame["realised"] >= -edge) & (frame["realised"] <= edge)

    result = coverage_report(frame, horizon=1, levels=(0.5, 0.90))
    assert result.calibrated, result.table.to_string()
    assert "CALIBRATED" in verdict(result)


def test_coverage_counts_independent_windows_not_days():
    """The reported n must be the thinned one.

    Overlapping windows inflate the sample h-fold and make the binomial test
    claim a precision it does not have - the same error the forecast module
    fixed by thinning 4644 daily predictions to 155.
    """
    n = 500
    frame = pd.DataFrame({"realised": np.zeros(n)})
    frame["low_0.9"] = -1.0
    frame["high_0.9"] = 1.0
    frame["hit_0.9"] = True
    result = coverage_report(frame, horizon=10, levels=(0.9,))
    assert result.n_checks == n
    assert result.n_independent == 50
    assert int(result.table.iloc[0]["n"]) == 50


def test_a_too_wide_interval_also_fails():
    """Two-sided on purpose: covering 100% when promising 50% is not success."""
    n = 400
    frame = pd.DataFrame({"realised": np.zeros(n)})
    frame["low_0.5"] = -10.0
    frame["high_0.5"] = 10.0
    frame["hit_0.5"] = True
    result = coverage_report(frame, horizon=1, levels=(0.5,))
    assert not result.calibrated, "an interval covering everything is miscalibrated"


def test_rolling_intervals_never_use_a_return_from_inside_the_window(monkeypatch):
    """Look-ahead check: the fit must not see the days it is predicting.

    Recorded by capturing the last date handed to the fitter at each step and
    requiring it to precede the outcome window.
    """
    seen = []
    import forecast.coverage as module

    real_fit = module.fit_garch

    def spy(returns):
        seen.append(returns.index[-1])
        return real_fit(returns)

    monkeypatch.setattr(module, "fit_garch", spy)
    prices = prices_from(garch_series(n=700, seed=12))
    frame = rolling_intervals(prices, 5, window=400, refit_every=100, n_paths=200)

    assert not frame.empty
    assert seen, "the fitter was never called"
    # Every fit ends at or before the first origin it is used for.
    assert min(frame.index) >= min(seen)


def test_walk_refuses_a_sample_too_short_for_its_window():
    prices = prices_from(garch_series(n=300, seed=13))
    with pytest.raises(ValueError, match="need at least"):
        rolling_intervals(prices, 10, window=1000)


def test_realised_move_spans_exactly_the_forecast_horizon():
    """The outcome must be an h-day move, not an (h-1)-day one.

    `returns` is one element shorter than the price series, so a position that
    indexes returns is one behind the same position in prices. Mixing the two
    silently measures a shorter move than the interval was built for, and
    because a shorter move is easier to contain, coverage comes out too high -
    the model would appear to pass its own test by being graded on an easier
    question. This pins the outcome against a directly computed h-day move.
    """
    returns = garch_series(n=700, seed=21)
    prices = prices_from(returns)
    horizon = 10

    frame = rolling_intervals(prices, horizon, window=400, refit_every=200, n_paths=200)
    log_price = np.log(prices)

    for origin in list(frame.index)[:20]:
        position = log_price.index.get_loc(origin)
        expected = float(log_price.iloc[position + horizon] - log_price.iloc[position])
        assert frame.loc[origin, "realised"] == pytest.approx(expected, abs=1e-12)


def test_a_shorter_horizon_would_have_shown_up_as_higher_coverage():
    """Why the off-by-one mattered rather than being cosmetic.

    Grading a 10-day interval on a 9-day move is easier, so the mistake pushes
    coverage up. This states the direction of that bias explicitly, so the fix
    cannot be undone without the suite noticing.
    """
    returns = garch_series(n=3000, seed=22)
    prices = prices_from(returns)
    log_price = np.log(prices)
    positions = range(400, len(log_price) - 21)

    def moves(h):
        return np.array([log_price.iloc[p + h] - log_price.iloc[p] for p in positions])

    ten = moves(10)
    edge = float(np.quantile(np.abs(ten), 0.90))
    # 5 against 10 rather than 9 against 10. The bias is real at 9 days but too
    # small to separate from sampling noise on one series - an earlier version
    # of this test compared them and got 0.90 against 0.90. What must be pinned
    # is the direction of the effect, and a clear gap pins it without depending
    # on a seed.
    assert (np.abs(moves(5)) <= edge).mean() > (np.abs(ten) <= edge).mean() + 0.02


def test_update_state_matches_a_full_refit_of_the_variance_path():
    """Rolling the variance forward must equal fitting through those days.

    The walk refits parameters rarely and rolls the variance daily. If the roll
    is wrong the interval silently uses a variance that belongs to a different
    day - which is the bug this replaced, where the fit was reused untouched
    for up to 30 days and the comment claimed otherwise.
    """
    from forecast.volatility import update_state, _variance_path, SCALE
    import dataclasses

    returns = garch_series(n=900, seed=31)
    fit = fit_garch(returns.iloc[:800])
    rolled = update_state(fit, returns.iloc[800:850])

    # Recompute the same recursion directly from the parameters.
    values = (np.asarray(returns.iloc[:850], dtype=float) * SCALE) - fit.mean_return * SCALE
    path = _variance_path(values, fit.omega, fit.alpha, fit.beta)
    assert rolled.last_variance == pytest.approx(path[-1], rel=1e-6)
    assert rolled.last_shock == pytest.approx(values[-1], rel=1e-9)
    # Parameters must be untouched - only the state moved. Compared field by
    # field because the fit carries a numpy array of residuals, and `==` on the
    # whole dataclass then returns an array rather than a bool.
    for name in ("omega", "alpha", "beta", "df", "mean_return", "n_observations"):
        assert getattr(rolled, name) == getattr(fit, name), name
    assert rolled.residuals is fit.residuals


def test_update_state_widens_the_interval_after_a_violent_stretch():
    """The reason the roll matters at all.

    A frozen variance cannot react to a crash beginning after the last refit.
    Feeding in a run of large moves must widen the forecast; if it does not,
    the daily update is decoration.
    """
    from forecast.volatility import update_state

    returns = garch_series(n=800, seed=32)
    fit = fit_garch(returns)
    shock_run = pd.Series([0.12, -0.15, 0.11, -0.13, 0.14])

    calm_width = np.diff(np.quantile(simulate_horizon(fit, 10, n_paths=8000), [0.05, 0.95]))
    after = update_state(fit, shock_run)
    wild_width = np.diff(np.quantile(simulate_horizon(after, 10, n_paths=8000), [0.05, 0.95]))
    assert wild_width > calm_width * 1.2


def test_fitted_drift_is_the_window_mean_and_not_zero():
    """The drift must come from the same window the variance did.

    Zero drift was tried first and failed coverage with a 41-to-19 upward miss
    split; what replaced it is a location parameter estimated on the trailing
    window, never anything from beyond the forecast origin.
    """
    returns = garch_series(n=900, seed=33) + 0.002  # a clear positive trend
    fit = fit_garch(returns)
    assert fit.mean_return == pytest.approx(float(returns.mean()), rel=1e-6)

    centred = np.median(simulate_horizon(fit, 10, n_paths=20000, drift=0.0))
    drifted = np.median(simulate_horizon(fit, 10, n_paths=20000, drift=fit.mean_return))
    assert drifted > centred
