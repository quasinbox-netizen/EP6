"""Ridge-penalised logistic regression - probability that the forward return is positive.

Why a hand-rolled fit rather than statsmodels' `Logit`: the features here are
genuinely collinear (`halving_after_90d` and `event_halving_90d` are the same
column, several return windows overlap), and both `Logit.fit` and
`fit_regularized` fail on that - the latter tries to invert a singular Hessian
to produce a covariance matrix we never use. This module only needs predicted
probabilities; the inference in this project comes from walk-forward
evaluation, not from coefficient p-values. An L2 penalty makes the objective
strictly convex, so the fit always converges and collinearity stops mattering.

Why logistic regression at all rather than something stronger: the sample holds
four halving cycles. A model with enough capacity to be interesting would learn
the noise and look excellent in training. The evaluation in evaluate.py is the
hard part of this problem, not the model.

Point-in-time discipline: standardisation statistics are computed on the
TRAINING rows only and then applied to the test rows. Scaling by full-sample
means would be a quiet look-ahead, exactly the kind features/checks.py exists
to catch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Columns that require knowing the future and must never enter a forecast.
# The halving schedule is roughly knowable in advance, which is defensible for
# a descriptive study, but a model given `days_to_next_halving` is being told
# something about the future by construction.
FORWARD_LOOKING = ("days_to_next_halving", "cycle_progress")

MIN_VARIANCE = 1e-10
DEFAULT_ALPHA = 1.0
# A feature must be observed on at least this share of the training rows.
# Anything sparser is dropped rather than imputed - see RidgeLogistic.fit.
DEFAULT_MIN_COVERAGE = 0.9


def usable_features(frame: pd.DataFrame) -> list[str]:
    """Numeric, backward-looking predictor columns.

    Excludes targets, the raw price level, and anything forward-looking.
    """
    out = []
    for column in frame.columns:
        if column.startswith("fwd_return_") or column in FORWARD_LOOKING:
            continue
        if column in ("close", "macro_phase", "liquidity_regime", "rates_regime"):
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        out.append(column)
    return out


@dataclass
class RidgeLogistic:
    """Logistic regression with an L2 penalty on the slopes (not the intercept)."""

    alpha: float = DEFAULT_ALPHA
    min_coverage: float = DEFAULT_MIN_COVERAGE
    columns: list[str] = field(default_factory=list)
    dropped_for_coverage: list[str] = field(default_factory=list)
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    coefficients: np.ndarray | None = None
    intercept: float = 0.0
    converged: bool = False
    n_train: int = 0
    base_rate: float = float("nan")

    # --- internals --------------------------------------------------------

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale

    @staticmethod
    def _objective(theta: np.ndarray, X: np.ndarray, y: np.ndarray, alpha: float):
        """Penalised negative log likelihood and its gradient.

        Written out rather than autodiffed so the gradient is exact and the
        optimiser converges in a few dozen iterations.
        """
        intercept, weights = theta[0], theta[1:]
        z = intercept + X @ weights
        # log(1 + exp(z)) computed stably for large |z|.
        log_terms = np.logaddexp(0.0, z)
        loss = float(np.sum(log_terms - y * z) + 0.5 * alpha * np.dot(weights, weights))

        probabilities = 1.0 / (1.0 + np.exp(-z))
        residual = probabilities - y
        gradient = np.empty_like(theta)
        gradient[0] = float(np.sum(residual))
        gradient[1:] = X.T @ residual + alpha * weights
        return loss, gradient

    # --- public API -------------------------------------------------------

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "RidgeLogistic":
        """Fit on the rows where both features and target are present.

        Columns are selected per fit, using only the training window: a column
        observed on less than `min_coverage` of those rows is dropped rather
        than imputed. This is not tidiness, it is necessary - the
        `days_since_event_*` columns are NaN before the first event of their
        category (an early-cycle category can have none at all), so a plain dropna
        across every candidate column leaves zero rows.

        Dropping beats imputing here because there is no honest fill value:
        "no such event has happened yet" is not a number of days.
        """
        candidates = usable_features(frame)
        coverage = frame.loc[:, candidates].notna().mean()
        columns = [c for c in candidates if coverage[c] >= self.min_coverage]
        self.dropped_for_coverage = [c for c in candidates if c not in columns]
        if not columns:
            raise ValueError("no feature reaches the required coverage")

        data = frame.loc[:, columns].join(target.rename("__y")).dropna()
        if data.empty:
            raise ValueError("no complete rows to train on")

        y = data["__y"].to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            raise ValueError("the target has only one class in the training window")

        X = data.loc[:, columns].to_numpy(dtype=float)
        mean = X.mean(axis=0)
        variance = X.var(axis=0)
        keep = variance > MIN_VARIANCE
        # Constant columns carry no information and would divide by ~zero.
        columns = [c for c, k in zip(columns, keep) if k]
        X, mean, variance = X[:, keep], mean[keep], variance[keep]

        self.columns = columns
        self.mean = mean
        self.scale = np.sqrt(variance)
        Xs = self._standardise(X)

        start = np.zeros(Xs.shape[1] + 1)
        start[0] = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
        result = minimize(
            self._objective, start, args=(Xs, y, self.alpha),
            jac=True, method="L-BFGS-B",
        )
        self.intercept = float(result.x[0])
        self.coefficients = result.x[1:]
        self.converged = bool(result.success)
        self.n_train = int(len(y))
        self.base_rate = float(y.mean())
        return self

    def predict_proba(self, frame: pd.DataFrame) -> pd.Series:
        """Probability that the target is 1, for every row (NaN where features are)."""
        if self.coefficients is None:
            raise ValueError("the model has not been fitted")
        subset = frame.loc[:, self.columns]
        complete = subset.notna().all(axis=1)
        out = pd.Series(np.nan, index=frame.index, name="probability")
        if not complete.any():
            return out
        Xs = self._standardise(subset.loc[complete].to_numpy(dtype=float))
        z = self.intercept + Xs @ self.coefficients
        out.loc[complete] = 1.0 / (1.0 + np.exp(-z))
        return out

    def weights(self) -> pd.Series:
        """Coefficients on standardised features - comparable across columns.

        Read them as "direction and rough size", not as tested effects. There
        are no p-values here on purpose: with collinear predictors and a
        penalty, individual coefficients are not identified. The question this
        module answers is whether the PREDICTIONS beat the baselines.
        """
        if self.coefficients is None:
            raise ValueError("the model has not been fitted")
        return pd.Series(self.coefficients, index=self.columns).sort_values(
            key=np.abs, ascending=False
        )
