"""Scoring a probabilistic forecast honestly.

Two things here matter more than the model itself.

**Overlapping targets.** The 30-day forward return on consecutive days shares
29 days of the future, so 365 daily predictions in a test window are nothing
like 365 independent observations. Metrics computed over all of them look
reassuringly stable while resting on maybe a dozen genuine cases.
`thin_to_non_overlapping` keeps every `horizon`-th row so the scored
predictions do not overlap. Both versions are reported, and the thinned one is
the one to believe.

**Calibration, not just accuracy.** A model that says 70% should be right about
70% of the time. Accuracy hides this: predicting the base rate every single
day gives decent accuracy and zero information. Hence the Brier score (which
punishes confident errors), the skill score against each baseline, and an
explicit calibration table.

No sklearn: AUC is computed from rank statistics, which is exact and keeps the
dependency list short.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

EPSILON = 1e-12


def thin_to_non_overlapping(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Keep every `horizon`-th day so the forward windows do not overlap."""
    index = pd.DatetimeIndex(index).sort_values()
    if len(index) == 0 or horizon <= 1:
        return index
    return index[::horizon]


def brier_score(target: pd.Series, probability: pd.Series) -> float:
    """Mean squared error of the probability. Lower is better; 0.25 = coin flip."""
    aligned = pd.DataFrame({"y": target, "p": probability}).dropna()
    if aligned.empty:
        return float("nan")
    return float(np.mean((aligned["p"] - aligned["y"]) ** 2))


def log_loss(target: pd.Series, probability: pd.Series) -> float:
    aligned = pd.DataFrame({"y": target, "p": probability}).dropna()
    if aligned.empty:
        return float("nan")
    p = np.clip(aligned["p"].to_numpy(), EPSILON, 1 - EPSILON)
    y = aligned["y"].to_numpy()
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(target: pd.Series, probability: pd.Series, threshold: float = 0.5) -> float:
    aligned = pd.DataFrame({"y": target, "p": probability}).dropna()
    if aligned.empty:
        return float("nan")
    return float(np.mean((aligned["p"] > threshold) == (aligned["y"] > 0.5)))


def auc(target: pd.Series, probability: pd.Series) -> float:
    """Area under the ROC curve via the Mann-Whitney rank statistic."""
    aligned = pd.DataFrame({"y": target, "p": probability}).dropna()
    if aligned.empty:
        return float("nan")
    y = aligned["y"].to_numpy() > 0.5
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(aligned["p"].to_numpy())
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def skill_score(model_brier: float, baseline_brier: float) -> float:
    """1 - model/baseline. Positive means better than the baseline, 0 means equal."""
    if not np.isfinite(model_brier) or not np.isfinite(baseline_brier) or baseline_brier <= 0:
        return float("nan")
    return float(1.0 - model_brier / baseline_brier)


def calibration_table(
    target: pd.Series, probability: pd.Series, *, bins: int = 5
) -> pd.DataFrame:
    """Predicted probability vs how often it actually happened, by bucket.

    A well calibrated model has `mean_predicted` close to `share_positive` in
    every row. Wide gaps mean the numbers cannot be read as probabilities even
    if the ranking is useful.
    """
    aligned = pd.DataFrame({"y": target, "p": probability}).dropna()
    if aligned.empty:
        return pd.DataFrame()
    edges = np.linspace(0.0, 1.0, bins + 1)
    aligned["bucket"] = pd.cut(aligned["p"], edges, include_lowest=True)
    grouped = aligned.groupby("bucket", observed=True)
    table = pd.DataFrame(
        {
            "n": grouped.size(),
            "mean_predicted": grouped["p"].mean(),
            "share_positive": grouped["y"].mean(),
        }
    )
    table["gap"] = table["mean_predicted"] - table["share_positive"]
    return table


def score(target: pd.Series, probability: pd.Series) -> dict:
    """Every metric for one set of predictions."""
    aligned = pd.DataFrame({"y": target, "p": probability}).dropna()
    return {
        "n": int(len(aligned)),
        "brier": brier_score(target, probability),
        "log_loss": log_loss(target, probability),
        "accuracy": accuracy(target, probability),
        "auc": auc(target, probability),
        "mean_probability": float(aligned["p"].mean()) if len(aligned) else float("nan"),
        "base_rate": float(aligned["y"].mean()) if len(aligned) else float("nan"),
    }


def compare_against_baselines(
    target: pd.Series,
    model_probability: pd.Series,
    baseline_probabilities: dict[str, pd.Series],
) -> pd.DataFrame:
    """One row per predictor, with skill scores relative to each baseline."""
    rows = []
    everything = {"model": model_probability, **baseline_probabilities}
    model_brier = brier_score(target, model_probability)
    for name, probability in everything.items():
        row = {"predictor": name}
        row.update(score(target, probability))
        row["skill_vs_coin_flip"] = skill_score(row["brier"], 0.25)
        rows.append(row)

    table = pd.DataFrame(rows).set_index("predictor")
    for name in baseline_probabilities:
        baseline_brier = brier_score(target, baseline_probabilities[name])
        table.loc["model", f"skill_vs_{name}"] = skill_score(model_brier, baseline_brier)
    return table


def verdict(table: pd.DataFrame, *, margin: float = 0.0) -> str:
    """Plain-language answer to "does this forecast beat the baselines"."""
    if table.empty or "model" not in table.index:
        return "no predictions to judge"
    skills = {
        column.removeprefix("skill_vs_"): table.loc["model", column]
        for column in table.columns
        if column.startswith("skill_vs_")
    }
    finite = {k: v for k, v in skills.items() if np.isfinite(v)}
    if not finite:
        return "not enough data to compare against the baselines"

    beaten = [k for k, v in finite.items() if v > margin]
    lost = [k for k, v in finite.items() if v <= margin]
    if not lost:
        return (
            "The model beats every baseline on the Brier score: "
            + ", ".join(f"{k} {finite[k]:+.1%}" for k in beaten)
        )
    return (
        "NO EDGE - the model does not beat: "
        + ", ".join(f"{k} ({finite[k]:+.1%})" for k in lost)
        + (". It does beat: " + ", ".join(beaten) if beaten else "")
    )
