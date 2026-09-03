"""Phase 3 - event study.

Two kinds of test, both necessary:
* NEGATIVE - on pure noise the method must not find patterns,
* POWER    - when we inject a known effect, the method must find it.
A negative test alone would also be passed by code that never detects anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.correlation import (
    hac_lags,
    hac_regression,
    lead_lag_correlation,
    regime_returns,
    rolling_correlation,
)
from analysis.event_study import (
    circular_shift_test,
    event_study,
    event_window_matrix,
    log_returns,
    window_scan,
)
from validation.synthetic import inject_drift, random_walk_prices


@pytest.fixture
def noise() -> pd.DataFrame:
    return random_walk_prices(3000, start="2013-01-01", seed=42)


@pytest.fixture
def anchors() -> list[str]:
    return ["2014-01-15", "2015-06-10", "2017-03-20", "2019-08-05", "2020-11-11"]


# --- the window contract -------------------------------------------------------


def test_window_matrix_has_expected_shape(noise, anchors):
    matrix, skipped = event_window_matrix(log_returns(noise), anchors, pre=30, post=90)
    assert matrix.shape == (5, 121)
    assert list(matrix.columns[:2]) == [-30, -29]
    assert matrix.columns[30] == 0
    assert len(skipped) == 0


def test_events_without_full_window_are_skipped_not_truncated(noise):
    """An event without a complete window drops out and is reported."""
    early = ["2013-01-05", "2016-01-01"]
    matrix, skipped = event_window_matrix(log_returns(noise), early, pre=30, post=90)
    assert len(matrix) == 1
    assert pd.Timestamp("2013-01-05") in skipped


def test_event_not_present_in_index_is_skipped(noise):
    matrix, skipped = event_window_matrix(log_returns(noise), ["2050-01-01"], pre=5, post=5)
    assert matrix.empty
    assert pd.Timestamp("2050-01-01") in skipped


# --- negative tests: noise -----------------------------------------------------


def test_no_false_pattern_on_pure_noise(anchors):
    """On a random walk the p-value distribution must be ~uniform.

    A single draw is not a test: at a nominal 5%, one noise sample in twenty
    WILL give a significant result, and that is correct. So we check the median
    p-value across 20 independent samples - for a uniform distribution it is 0.5.
    """
    p_values = []
    for seed in range(20):
        prices = random_walk_prices(3000, start="2013-01-01", seed=500 + seed)
        result = event_study(prices, anchors, pre=30, post=90, n_boot=500, seed=seed)
        assert result.n_events == 5
        p_values.append(result.car_summary["p_value"])
    assert float(np.median(p_values)) > 0.25


def test_confidence_interval_is_wide_when_events_are_few(anchors):
    """At n=5 the confidence interval must be wide - that is not a defect."""
    prices = random_walk_prices(3000, start="2013-01-01", seed=42)
    result = event_study(prices, anchors, pre=30, post=90, n_boot=1000)
    width = result.car_summary["ci_high"] - result.car_summary["ci_low"]
    assert width > 0.30, "a confidence interval from 5 events cannot be narrow"
    # The percentile bootstrap is narrower at such small n - hence it does not decide.
    boot_width = result.car_summary["boot_ci_high"] - result.car_summary["boot_ci_low"]
    assert boot_width < width


def test_false_positive_rate_stays_near_nominal():
    """Across 40 random samples ~5% should be significant, not 30%."""
    rejections = 0
    trials = 40
    for seed in range(trials):
        prices = random_walk_prices(1500, start="2015-01-01", seed=1000 + seed)
        events = pd.to_datetime(["2016-03-01", "2017-05-15", "2018-02-20"])
        result = event_study(prices, events, pre=30, post=60, n_boot=1000, seed=seed)
        if result.car_summary["p_value"] < 0.05:
            rejections += 1
    # With 40 samples at a nominal 5% we allow up to 5 rejections (~12%).
    assert rejections <= 5, f"too many false discoveries: {rejections}/{trials}"


def test_circular_shift_test_is_calm_on_noise(noise):
    returns = log_returns(noise).dropna()
    mask = pd.Series(False, index=returns.index)
    mask.iloc[500:560] = True
    mask.iloc[1200:1260] = True
    result = circular_shift_test(returns, mask, n_permutations=2000)
    assert result["p_value"] > 0.05


# --- power tests: an injected effect -------------------------------------------


def test_detects_injected_drift(anchors):
    """An injected drift that clearly dominates the noise must be detected.

    We use a calmer series (sigma 1.5%/day): at Bitcoin-scale volatility of
    around 4% a day, five events are NOT enough to detect a 0.6%/day drift -
    that is not a flaw in the method, just the real power limit at n=5.
    """
    calm = random_walk_prices(3000, start="2013-01-01", seed=42, sigma=0.015)
    seeded = inject_drift(calm, anchors, window=60, daily_drift=0.006)
    result = event_study(seeded, anchors, pre=30, post=60, n_boot=2000)
    assert result.car_summary["car"] > 0.20
    assert result.car_summary["p_value"] < 0.05
    assert result.car_summary["ci_low"] > 0


def test_five_events_cannot_detect_a_drift_buried_in_btc_scale_noise(noise, anchors):
    """An honest power limit: at Bitcoin volatility, 5 events are too few.

    This test exists so that nobody "improves" the method into finding effects
    the data cannot support.
    """
    seeded = inject_drift(noise, anchors, window=60, daily_drift=0.006)
    result = event_study(seeded, anchors, pre=30, post=60, n_boot=2000)
    assert result.car_summary["p_value"] > 0.05


def test_abnormal_return_removes_pre_event_trend():
    """A trend already present before the event must not count as the effect."""
    trending = random_walk_prices(2000, start="2015-01-01", seed=5, mu=0.003, sigma=0.02)
    events = ["2017-01-10", "2018-04-20", "2019-09-30"]
    raw = event_study(trending, events, pre=30, post=60, abnormal=False, n_boot=1500)
    adjusted = event_study(trending, events, pre=30, post=60, abnormal=True, n_boot=1500)
    assert raw.car_summary["car"] > adjusted.car_summary["car"]
    assert abs(adjusted.car_summary["car"]) < 0.10


def test_circular_shift_test_detects_real_window_effect(noise):
    returns = log_returns(noise).dropna()
    mask = pd.Series(False, index=returns.index)
    for start in (300, 900, 1500, 2100):
        mask.iloc[start : start + 90] = True
    boosted = returns + mask.astype(float) * 0.012
    result = circular_shift_test(boosted, mask, n_permutations=2000)
    assert result["difference"] > 0.008
    assert result["p_value"] < 0.05


# --- scanning many windows -----------------------------------------------------


def test_window_scan_returns_one_row_per_hypothesis(noise):
    returns = log_returns(noise)
    frame = pd.DataFrame({"target": returns})
    for i, window in enumerate((30, 60, 90)):
        flag = pd.Series(0, index=returns.index)
        flag.iloc[100 + i * 400 : 100 + i * 400 + window] = 1
        frame[f"window_{window}d"] = flag
    scan = window_scan(
        frame, ["window_30d", "window_60d", "window_90d"], "target", n_permutations=500
    )
    assert len(scan) == 3
    assert set(scan["hypothesis"]) == {"window_30d", "window_60d", "window_90d"}
    assert scan["p_value"].between(0, 1).all()


def test_window_scan_ignores_missing_columns(noise):
    frame = pd.DataFrame({"target": log_returns(noise)})
    assert window_scan(frame, ["nie_istnieje"], "target").empty


# --- correlations --------------------------------------------------------------


def test_rolling_correlation_tracks_a_regime_change():
    rng = np.random.default_rng(3)
    index = pd.date_range("2015-01-01", periods=1200, freq="D")
    base = pd.Series(rng.normal(0, 0.02, 1200), index=index)
    other = base.copy()
    other.iloc[:600] = rng.normal(0, 0.02, 600)  # first half: no relationship
    correlation = rolling_correlation(base, other, window=120)
    assert abs(correlation.iloc[300]) < 0.3
    assert correlation.iloc[-1] > 0.9


def test_hac_regression_reports_wider_errors_than_naive_ols():
    """HAC errors must be wider when the residuals are autocorrelated."""
    import statsmodels.api as sm

    rng = np.random.default_rng(11)
    n = 800
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)))
    noise = pd.Series(np.zeros(n))
    for i in range(1, n):
        noise.iloc[i] = 0.9 * noise.iloc[i - 1] + rng.normal(0, 1)
    y = 0.05 * x + noise

    hac = hac_regression(y, pd.DataFrame({"x": x}))
    naive = sm.OLS(y, sm.add_constant(pd.DataFrame({"x": x}))).fit()
    assert hac.loc["x", "std_error"] > naive.bse["x"]
    assert hac.attrs["hac_lags"] == hac_lags(n)


def test_lead_lag_marks_only_positive_lags_as_predictive():
    rng = np.random.default_rng(7)
    index = pd.date_range("2018-01-01", periods=600, freq="D")
    driver = pd.Series(rng.normal(0, 1, 600), index=index)
    follower = driver.shift(10).fillna(0) + rng.normal(0, 0.3, 600)
    table = lead_lag_correlation(follower, driver, max_lag=20, step=5)
    best = table.loc[table["correlation"].idxmax()]
    assert best["lag_days"] == 10
    assert bool(best["predictive"]) is True


def test_regime_returns_splits_by_label():
    index = pd.date_range("2019-01-01", periods=400, freq="D")
    returns = pd.Series(np.r_[np.full(200, 0.01), np.full(200, -0.005)], index=index)
    regime = pd.Series(["expanding"] * 200 + ["contracting"] * 200, index=index)
    table = regime_returns(returns, regime)
    assert table.loc["expanding", "mean_daily"] == pytest.approx(0.01)
    assert table.loc["contracting", "share_positive"] == 0.0
