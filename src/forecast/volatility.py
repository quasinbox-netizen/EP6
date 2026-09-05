"""How far the price is likely to move - not which way.

This module exists because the rest of the project established that direction
is not predictable here (`forecast` reports NO EDGE, the specification curve is
NOT ROBUST, nothing survives correction). None of that says the SIZE of the
next move is unpredictable, and it is not: volatility clusters. Calm days
follow calm days, violent ones follow violent ones, and that is one of the most
reliably reproduced regularities in asset prices.

So this fits GARCH(1,1) with Student-t innovations and produces an interval:
"in h days the price will be between X and Y with probability p". That is a
falsifiable claim about the future, and `coverage.py` falsifies it - if the 90%
interval does not contain the outcome about 90% of the time out of sample, the
tool says so and refuses to quote a number.

THREE DECISIONS WORTH ARGUING WITH
----------------------------------
1. DRIFT. This started as zero - centre the interval on today's price, because
   fitting a trend would encode the one thing the project showed is not
   predictable. That reasoning is appealing and it FAILED the coverage test.

   Measured, on 403 non-overlapping 10-day windows: the 90% interval covered
   85.1%, and the misses were not symmetric. 41 outcomes finished ABOVE the
   interval against 19 below; in 2017 the split was 12 up against 1 down, in
   2020 9 against 2. The interval was not too narrow, it was mis-centred.

   The mistake in the original reasoning is that zero drift is not the neutral
   choice it feels like. It is the claim that the expected 10-day move is
   exactly zero - as much a directional statement as any other, and one that
   this data rejects. There is no assumption-free option; there is only the
   choice of which assumption to make, and the coverage test is what decides
   between them rather than an argument about which feels more modest.

   The drift now used is the mean daily return over the same trailing window
   the variance is fitted on. It is not a forecast of direction: it is a
   location parameter that was removed to fit the variance and is put back
   rather than silently set to zero. Its cost is real and is visible in the
   report - it lags turning points, and after a top it keeps leaning upward
   until the window rolls past the top.

   `drift=0.0` is still available, and `range --calibrate` scores whichever is
   in use, so the claim that one beats the other stays runnable.

2. SIMULATION, NOT sqrt(h) SCALING. Multiplying a daily sigma by sqrt(h) is
   wrong twice over here. GARCH mean-reverts, so a 10-day-ahead variance is not
   10 times the 1-day one when today is unusually calm or unusually wild; and
   summing h fat-tailed daily shocks produces something closer to normal than
   the daily distribution is, so scaling a t quantile by sqrt(h) overstates the
   tails. Simulating the model forward gets both right for free.

3. STUDENT-t, NOT NORMAL. A normal GARCH gets the 68% interval about right and
   the 95% badly wrong, because the daily shocks have fat tails. Measured on
   this data before any model was fitted: a sqrt-of-time normal interval covers
   67.8% at the nominal 68% and only 86.9% at the nominal 95%. The 95% failure
   is what the t distribution is here to fix.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special

# Returns are fitted in percent. GARCH parameters on raw log returns are around
# 1e-6, which optimisers handle badly; scaling by 100 puts omega near 1e-2.
SCALE = 100.0

# Below 2 a Student-t has no finite variance and the standardisation used here
# divides by zero. 2.05 keeps the optimiser away from that edge; hitting the
# bound is itself informative and is reported.
MIN_DF = 2.05
MAX_DF = 50.0


@dataclass(frozen=True)
class GarchFit:
    omega: float
    alpha: float
    beta: float
    df: float
    last_variance: float
    last_shock: float
    n_observations: int
    converged: bool
    log_likelihood: float
    # Mean daily log return over the fitting window, in raw units. The variance
    # model is fitted on demeaned returns, so this is the location parameter
    # that was taken out - and putting it back is a choice, not a formality.
    # See the drift discussion in the module docstring.
    mean_return: float = 0.0
    # Standardised residuals, eps_t / sigma_t. These are what the simulation
    # draws from by default - see `simulate_horizon`. Keeping them on the fit
    # is what makes the forecast non-parametric in the shock distribution while
    # staying parametric in the variance dynamics.
    residuals: np.ndarray = None

    @property
    def persistence(self) -> float:
        """alpha + beta. At 1.0 shocks never decay and the model has no mean."""
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        if self.persistence >= 1.0:
            return float("nan")
        return self.omega / (1.0 - self.persistence)

    def summary(self) -> str:
        daily = np.sqrt(self.last_variance) / SCALE
        return (
            f"GARCH(1,1)-t | omega={self.omega:.4f} alpha={self.alpha:.3f} "
            f"beta={self.beta:.3f} df={self.df:.1f} | persistence={self.persistence:.3f} "
            f"| today's sigma={daily:.2%}/day"
        )


def _variance_path(returns: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Conditional variance, one step at a time. This is the whole model."""
    n = returns.size
    variance = np.empty(n)
    # Seeding with the sample variance rather than omega/(1-alpha-beta) keeps
    # the recursion finite even while the optimiser is exploring parameters
    # whose persistence exceeds one.
    variance[0] = max(returns.var(), 1e-8)
    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]
    return variance


def _unpack(theta: np.ndarray) -> tuple[float, float, float, float]:
    """Unconstrained parameters -> valid GARCH parameters.

    Every constraint holds by construction rather than by rejection:

        omega       > 0                 exp
        persistence in (0, 1)           logistic, so stationarity is automatic
        alpha, beta > 0, summing to it  a logistic split of the persistence
        df          in (MIN_DF, MAX_DF) logistic

    The first version enforced alpha + beta < 1 by returning 1e10 from the
    objective, which turns the boundary into a cliff. L-BFGS-B cannot see round
    a cliff: it walked into alpha + beta = 1.000 and reported failure. That is
    an IGARCH fit - shocks never decay, there is no long-run variance, and
    every horizon inherits today's volatility forever. Reparameterising removes
    the boundary instead of punishing it.
    """
    omega = np.exp(np.clip(theta[0], -20.0, 10.0))
    persistence = 0.9995 / (1.0 + np.exp(-np.clip(theta[1], -30.0, 30.0)))
    weight = 1.0 / (1.0 + np.exp(-np.clip(theta[2], -30.0, 30.0)))
    df_span = MAX_DF - MIN_DF
    df = MIN_DF + df_span / (1.0 + np.exp(-np.clip(theta[3], -30.0, 30.0)))
    alpha = persistence * weight
    beta = persistence * (1.0 - weight)
    return float(omega), float(alpha), float(beta), float(df)


def _negative_log_likelihood(theta: np.ndarray, returns: np.ndarray) -> float:
    omega, alpha, beta, df = _unpack(theta)
    variance = _variance_path(returns, omega, alpha, beta)
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        return 1e10
    # Standardised Student-t: scaled so the innovation has unit variance, which
    # is what lets sigma keep its meaning as the conditional standard deviation.
    constant = (
        special.gammaln((df + 1) / 2)
        - special.gammaln(df / 2)
        - 0.5 * np.log(np.pi * (df - 2))
    )
    z2 = returns**2 / variance
    log_density = (
        constant - 0.5 * np.log(variance) - ((df + 1) / 2) * np.log1p(z2 / (df - 2))
    )
    total = float(np.sum(log_density))
    return -total if np.isfinite(total) else 1e10


def fit_garch(returns: pd.Series | np.ndarray) -> GarchFit:
    """Fit GARCH(1,1) with Student-t innovations by maximum likelihood."""
    values = np.asarray(pd.Series(returns).dropna(), dtype=float) * SCALE
    if values.size < 100:
        raise ValueError(f"need at least 100 returns to fit GARCH, got {values.size}")
    mean_return = float(values.mean())
    values = values - mean_return  # the model is about spread, not level

    variance = values.var()
    # Several starts, because the likelihood is nearly flat along the
    # persistence direction and a single start can settle in the wrong basin.
    # They differ in what they assume about persistence and tail thickness.
    starts = [
        np.array([np.log(variance * 0.05), 2.9, -2.0, 0.0]),   # persistence .95
        np.array([np.log(variance * 0.20), 1.4, -1.0, 1.0]),   # persistence .80
        np.array([np.log(variance * 0.50), 0.0, 0.0, -1.0]),   # persistence .50
    ]

    best = None
    for start in starts:
        result = optimize.minimize(
            _negative_log_likelihood, start, args=(values,),
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6},
        )
        if best is None or result.fun < best.fun:
            best = result

    omega, alpha, beta, df = _unpack(best.x)
    path = _variance_path(values, omega, alpha, beta)
    residuals = values / np.sqrt(path)
    return GarchFit(
        omega=float(omega), alpha=float(alpha), beta=float(beta), df=float(df),
        last_variance=float(path[-1]),
        last_shock=float(values[-1]),
        n_observations=int(values.size),
        converged=bool(best.success),
        log_likelihood=float(-best.fun),
        residuals=residuals,
        mean_return=mean_return / SCALE,
    )


RESIDUAL_BURN_IN = 250


def extend_residual_pool(fit: GarchFit, history) -> GarchFit:
    """Re-estimate the shock distribution on a longer past than the parameters.

    The two things GARCH needs are not equally hungry for data. The variance
    DYNAMICS - how fast a shock decays - are identified by a few years and are
    better estimated on a recent window, because volatility regimes change. The
    shock DISTRIBUTION is a different matter: its interesting part is the tail,
    the tail is by definition rare, and four years of daily data contains only
    a handful of the days that decide where a 95% interval belongs.

    So the parameters stay fitted on the trailing window and the residual pool
    is recomputed over everything available, using those parameters. Nothing
    from beyond the forecast origin enters - `history` is past data, just more
    of it.

    Standardisation is what makes eras comparable: dividing each return by the
    model's own sigma for that day removes the level of volatility and leaves
    the shape. A violent day in 2013 and a violent day in 2024 look alike after
    that, which is the assumption this rests on and the reason it can fail - if
    the SHAPE of the shocks changed, not just their size, the old days are the
    wrong evidence.

    The first `RESIDUAL_BURN_IN` residuals are dropped: the recursion is seeded
    with a sample variance rather than the true state, so the earliest ones are
    artefacts of that seed.
    """
    values = np.asarray(pd.Series(history).dropna(), dtype=float) * SCALE
    if values.size <= RESIDUAL_BURN_IN + 50:
        return fit
    values = values - fit.mean_return * SCALE
    path = _variance_path(values, fit.omega, fit.alpha, fit.beta)
    pool = (values / np.sqrt(path))[RESIDUAL_BURN_IN:]
    return dataclasses.replace(fit, residuals=pool)


def update_state(fit: GarchFit, new_returns) -> GarchFit:
    """Roll the variance recursion forward without refitting the parameters.

    GARCH has two parts and they age at completely different rates. The
    parameters move slowly and refitting them daily is wasted work. The
    CONDITIONAL VARIANCE moves every day and is the whole point of the model -
    it is what makes today's interval narrower after a calm week and wider
    after a violent one.

    Reusing a fit unchanged between refits freezes both. That is what the first
    version of the coverage walk did, while a comment claimed otherwise: with a
    30-day refit interval the interval quoted on any given day could be built
    on a variance up to a month stale. In a quiet stretch that is harmless. In
    the week a crash starts it is the difference between an interval that has
    widened and one that has not, which is exactly when the interval is being
    relied on.
    """
    values = np.asarray(pd.Series(new_returns).dropna(), dtype=float) * SCALE
    if values.size == 0:
        return fit
    values = values - fit.mean_return * SCALE

    variance = fit.last_variance
    shock = fit.last_shock
    for value in values:
        variance = fit.omega + fit.alpha * shock**2 + fit.beta * variance
        shock = value
    return dataclasses.replace(fit, last_variance=float(variance), last_shock=float(shock))


def annualised_long_run_volatility(fit: GarchFit) -> float:
    """The level volatility decays towards, in the usual annual units.

    Reported because it is the honest way to see whether a fit is sane. A
    persistence of 0.999 makes this number meaningless (the decay takes years),
    and that is worth noticing before quoting an interval built on it.
    """
    variance = fit.long_run_variance
    if not np.isfinite(variance):
        return float("nan")
    return float(np.sqrt(variance) / SCALE * np.sqrt(365.0))


def simulate_horizon(
    fit: GarchFit,
    horizon: int,
    *,
    n_paths: int = 20000,
    seed: int = 20260905,
    drift: float = 0.0,
    shocks: str = "empirical",
) -> np.ndarray:
    """Cumulative log returns over `horizon` days, one per simulated path.

    Simulation rather than a closed form because the quantity wanted is a
    quantile of a SUM of h dependent, fat-tailed shocks. There is no usable
    closed form for that, and the two shortcuts people reach for - sqrt(h)
    scaling and a daily t quantile - are wrong in opposite directions.

    `shocks="empirical"` is filtered historical simulation: the paths are built
    by resampling the fit's own standardised residuals rather than drawing from
    a Student-t. This is the default because the parametric version FAILED its
    coverage test on this data. Fitted t innovations gave 50% and 68% intervals
    that were right (49.6% and 65.0% observed) and 90% and 95% intervals that
    were far too narrow - 84.4% and 90.6%, p = 0.0004 and 0.0002 across 403
    non-overlapping windows.

    The diagnosis is that a Student-t is a shape assumption, and the real
    shocks do not have that shape in the tail: the worst days are worse than
    any single df can express while still fitting the middle of the
    distribution, where most of the likelihood lives. Resampling the residuals
    drops the assumption. The extreme days enter the simulation because they
    happened, at the frequency they happened.

    `shocks="t"` keeps the parametric version, because a claim that one method
    beats another should stay runnable rather than being asserted in a comment.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")
    rng = np.random.default_rng(seed)
    df = fit.df

    if shocks == "empirical":
        pool = fit.residuals
        if pool is None or pool.size < 50:
            raise ValueError(
                "empirical shocks need the fit's standardised residuals; "
                "refit, or pass shocks='t'"
            )
        # Rescaled to unit variance so sigma keeps meaning the conditional
        # standard deviation. The residuals are close to unit variance already
        # - that is what standardising them did - but not exactly, and letting
        # the difference through would quietly bias every interval.
        pool = pool / pool.std()
        draws = rng.integers(0, pool.size, size=(n_paths, horizon))
        shock_matrix = pool[draws]
    elif shocks == "t":
        # Standardised t: unit variance, so sigma stays the conditional s.d.
        shock_matrix = rng.standard_t(df, size=(n_paths, horizon)) / np.sqrt(df / (df - 2.0))
    else:
        raise ValueError(f"unknown shocks={shocks!r}; use 'empirical' or 't'")

    variance = np.full(n_paths, fit.omega + fit.alpha * fit.last_shock**2
                       + fit.beta * fit.last_variance)
    total = np.zeros(n_paths)
    for step in range(horizon):
        sigma = np.sqrt(variance)
        step_return = sigma * shock_matrix[:, step]
        total += step_return
        if step + 1 < horizon:
            variance = fit.omega + fit.alpha * step_return**2 + fit.beta * variance
    return total / SCALE + drift * horizon


def price_interval(
    fit: GarchFit,
    last_price: float,
    horizon: int,
    levels=(0.5, 0.68, 0.90, 0.95),
    **kwargs,
) -> pd.DataFrame:
    """Price quantiles at the horizon, one row per confidence level."""
    draws = simulate_horizon(fit, horizon, **kwargs)
    rows = []
    for level in levels:
        tail = (1.0 - level) / 2.0
        low, high = np.quantile(draws, [tail, 1.0 - tail])
        rows.append({
            "level": level,
            "low": float(last_price * np.exp(low)),
            "high": float(last_price * np.exp(high)),
            "low_pct": float(np.expm1(low)),
            "high_pct": float(np.expm1(high)),
        })
    frame = pd.DataFrame(rows)
    frame.attrs["median"] = float(last_price * np.exp(np.median(draws)))
    frame.attrs["horizon"] = horizon
    return frame
