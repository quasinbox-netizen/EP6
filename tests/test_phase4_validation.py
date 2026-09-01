"""Faza 4 - walidacja statystyczna.

Test kluczowy: wzorzec znaleziony przez przeszukanie wielu hipotez na czystym
szumie musi zostac ODRZUCONY po korekcie na wielokrotne testowanie.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.event_study import circular_shift_test, log_returns, window_scan
from validation.multiple_testing import (
    benjamini_hochberg,
    bonferroni,
    correct,
    expected_false_discoveries,
    summarize,
)
from validation.splits import (
    Split,
    assert_no_overlap,
    cycle_of,
    cycle_split,
    replicate_finding,
    split_frame,
    walk_forward_splits,
)
from validation.synthetic import random_walk_prices


# --- korekty --------------------------------------------------------------


def test_bonferroni_scales_by_number_of_tests():
    assert bonferroni([0.01], 10)[0] == pytest.approx(0.10)
    assert bonferroni([0.5], 10)[0] == 1.0  # obciete do 1


def test_bh_is_monotonic_and_less_conservative_than_bonferroni():
    p = np.array([0.001, 0.008, 0.02, 0.04, 0.3, 0.7])
    bh = benjamini_hochberg(p)
    bonf = bonferroni(p)
    assert (bh <= bonf + 1e-12).all()
    assert (np.diff(bh[np.argsort(p)]) >= -1e-12).all(), "q musi rosnac wraz z p"


def test_bh_matches_textbook_example():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    q = benjamini_hochberg(p)
    assert q[0] == pytest.approx(0.05)
    assert q[-1] == pytest.approx(0.05)


def test_untestable_hypotheses_do_not_poison_the_correction():
    """Hipoteza bez danych (NaN) nie moze zepsuc q pozostalym.

    Monotonizacja BH idzie od konca posortowanej listy, wiec pojedynczy NaN
    na koncu potrafi wyzerowac - a raczej "zNaNowac" - cala kolumne.
    """
    p = np.array([0.001, 0.02, np.nan, 0.5, np.nan])
    q = benjamini_hochberg(p)
    assert np.isfinite(q[[0, 1, 3]]).all()
    assert np.isnan(q[[2, 4]]).all()
    assert q[0] == pytest.approx(0.003)  # 3 testowalne hipotezy, nie 5

    scan = pd.DataFrame({"hypothesis": list("abcde"), "p_value": p})
    corrected = correct(scan, method="bh")
    assert corrected.attrs["n_tests"] == 3
    assert corrected.attrs["n_untestable"] == 2
    # q = [0.003, 0.03, 0.5] przy trzech testowalnych hipotezach
    assert corrected["significant_adjusted"].sum() == 2
    assert "pominiete" in summarize(corrected)


def test_bonferroni_ignores_untestable_hypotheses():
    q = bonferroni(np.array([0.01, np.nan, np.nan]))
    assert q[0] == pytest.approx(0.01)  # jeden testowalny wynik, mnoznik 1
    assert np.isnan(q[1:]).all()


def test_expected_false_discoveries_is_reported():
    assert expected_false_discoveries(200, 0.05) == pytest.approx(10.0)


# --- test glowny: falszywe odkrycie znika po korekcie --------------------


def _scan_noise_for_patterns(seed: int, n_hypotheses: int = 60) -> pd.DataFrame:
    """Przeszukuje szum wieloma oknami - fabryka falszywych odkryc."""
    prices = random_walk_prices(2500, start="2014-01-01", seed=seed)
    returns = log_returns(prices).dropna()
    frame = pd.DataFrame({"target": returns})
    rng = np.random.default_rng(seed)
    columns = []
    for i in range(n_hypotheses):
        flag = pd.Series(0, index=returns.index)
        start = int(rng.integers(0, len(returns) - 200))
        width = int(rng.integers(20, 120))
        flag.iloc[start : start + width] = 1
        column = f"okno_{i}"
        frame[column] = flag
        columns.append(column)
    return window_scan(frame, columns, "target", n_permutations=400, seed=seed)


def test_search_over_noise_produces_raw_significant_hits():
    """Kontrola zalozenia: przeszukiwanie szumu MUSI dawac surowe trafienia."""
    scan = _scan_noise_for_patterns(seed=3)
    assert (scan["p_value"] < 0.05).sum() >= 1, "bez trafien nie ma czego korygowac"


def test_false_positive_from_scanning_is_rejected_after_correction():
    """Najlepsza hipoteza z przeszukania szumu nie moze przezyc korekty."""
    scan = _scan_noise_for_patterns(seed=3)
    best = scan.sort_values("p_value").iloc[0]
    assert best["p_value"] < 0.05, "punkt wyjscia: surowo istotny wynik"

    corrected_bh = correct(scan, method="bh", alpha=0.05)
    corrected_bonf = correct(scan, method="bonferroni", alpha=0.05)
    assert not corrected_bh["significant_adjusted"].any()
    assert not corrected_bonf["significant_adjusted"].any()


def test_correction_keeps_a_genuinely_strong_effect():
    """Kontrola drugiej strony: prawdziwy silny efekt ma przezyc korekte."""
    prices = random_walk_prices(2500, start="2014-01-01", seed=9)
    returns = log_returns(prices).dropna()
    frame = pd.DataFrame({"target": returns})
    real = pd.Series(0, index=returns.index)
    for start in (200, 800, 1400, 2000):
        real.iloc[start : start + 120] = 1
    frame["target"] = frame["target"] + real * 0.02
    frame["okno_prawdziwe"] = real
    rng = np.random.default_rng(9)
    columns = ["okno_prawdziwe"]
    for i in range(20):
        flag = pd.Series(0, index=returns.index)
        start = int(rng.integers(0, len(returns) - 200))
        flag.iloc[start : start + 60] = 1
        frame[f"szum_{i}"] = flag
        columns.append(f"szum_{i}")

    scan = window_scan(frame, columns, "target", n_permutations=1000, seed=9)
    corrected = correct(scan, method="bh", alpha=0.05)
    survivors = corrected[corrected["significant_adjusted"]]["hypothesis"].tolist()
    assert "okno_prawdziwe" in survivors


def test_n_tests_override_counts_hypotheses_dropped_along_the_way():
    scan = pd.DataFrame({"hypothesis": ["a"], "p_value": [0.004]})
    honest = correct(scan, method="bonferroni", n_tests_override=100)
    assert honest.iloc[0]["p_adjusted"] == pytest.approx(0.4)
    assert not honest.iloc[0]["significant_adjusted"]


def test_summarize_mentions_chance_level():
    scan = pd.DataFrame({"hypothesis": [f"h{i}" for i in range(40)],
                         "p_value": np.linspace(0.001, 0.9, 40)})
    text = summarize(correct(scan, method="bh"))
    assert "40 hipotez" in text and "przypadek" in text


# --- podzialy -------------------------------------------------------------


def test_cycle_index_matches_halving_schedule():
    dates = pd.to_datetime(["2013-06-01", "2018-01-01", "2021-06-01", "2025-01-01"])
    assert cycle_of(dates).tolist() == [1, 2, 3, 4]


def test_cycle_split_puts_later_cycles_in_test():
    index = pd.date_range("2012-01-01", "2026-01-01", freq="D")
    split = cycle_split(index, [0, 1, 2], [3, 4])
    assert split.train.max() < split.test.min()
    assert split.test.min() == pd.Timestamp("2020-05-11")
    assert_no_overlap(split)


def test_cycle_split_embargo_covers_target_horizon():
    """Bez embarga ostatnie dni treningu widza ceny ze zbioru testowego."""
    index = pd.date_range("2012-01-01", "2026-01-01", freq="D")
    naive = cycle_split(index, [0, 1, 2], [3, 4])
    with pytest.raises(AssertionError, match="luka"):
        assert_no_overlap(naive, horizon_days=90)

    embargoed = cycle_split(index, [0, 1, 2], [3, 4], embargo_days=120)
    assert_no_overlap(embargoed, horizon_days=90)
    assert len(embargoed.embargo) > 0


def test_walk_forward_folds_move_forward_without_overlap():
    index = pd.date_range("2015-01-01", periods=2000, freq="D")
    folds = walk_forward_splits(index, train_days=500, test_days=250, embargo_days=90)
    assert len(folds) >= 3
    for fold in folds:
        assert_no_overlap(fold, horizon_days=30)
        assert fold.train.max() < fold.test.min()
    assert folds[0].test.min() < folds[-1].test.min()


def test_expanding_window_grows_and_sliding_does_not():
    index = pd.date_range("2015-01-01", periods=2000, freq="D")
    expanding = walk_forward_splits(index, train_days=400, test_days=200, expanding=True)
    sliding = walk_forward_splits(index, train_days=400, test_days=200, expanding=False)
    assert len(expanding[-1].train) > len(expanding[0].train)
    assert len(sliding[-1].train) == len(sliding[0].train)


def test_split_frame_returns_disjoint_pieces():
    index = pd.date_range("2016-01-01", periods=800, freq="D")
    frame = pd.DataFrame({"x": np.arange(800)}, index=index)
    split = Split("t", index[:500], index[600:], index[500:600])
    train, test = split_frame(frame, split)
    assert len(train) == 500 and len(test) == 200
    assert train.index.intersection(test.index).empty


# --- replikacja -----------------------------------------------------------


def test_replication_requires_more_than_matching_sign():
    train = {"difference": 0.01, "p_value": 0.001}
    weak = {"difference": 0.001, "p_value": 0.04}   # ten sam znak, efekt zniknal
    strong = {"difference": 0.009, "p_value": 0.01}
    assert not replicate_finding(train, weak)["replicated"]
    assert replicate_finding(train, strong)["replicated"]


def test_replication_fails_on_sign_flip():
    train = {"difference": 0.01, "p_value": 0.001}
    flipped = {"difference": -0.012, "p_value": 0.001}
    result = replicate_finding(train, flipped)
    assert not result["same_sign"]
    assert not result["replicated"]


def test_replication_fails_when_out_of_sample_is_insignificant():
    train = {"difference": 0.01, "p_value": 0.001}
    noisy = {"difference": 0.009, "p_value": 0.4}
    assert not replicate_finding(train, noisy)["replicated"]
