"""Korekta na wielokrotne testowanie.

Problem, ktory ten modul rozwiazuje, jest w tym projekcie glowny. Majac
kilkanascie okien halvingowych, kilka kategorii zdarzen i cztery horyzonty,
testujemy setki hipotez. Przy 100 testach na czystym szumie okolo piec
wyjdzie "istotnych" na poziomie 0.05 - i to wlasnie te piec opisza potem
naglowki o "wzorcu cyklicznym".

Dwie korekty, dwie filozofie:

* Bonferroni kontroluje prawdopodobienstwo POPELNIENIA CHOCBY JEDNEGO bledu
  I rodzaju (FWER). Konserwatywna, wlasciwa gdy pojedynczy falszywy wniosek
  jest kosztowny - np. gdy zamierzasz na nim postawic pieniadze.
* Benjamini-Hochberg kontroluje ODSETEK falszywych odkryc wsrod odrzuconych
  (FDR). Lagodniejsza, wlasciwa na etapie generowania hipotez do dalszego
  badania.

Uczciwosc wymaga zliczania WSZYSTKICH testow, ktore wykonales, a nie tylko
tych, ktore trafily do raportu. Dlatego `correct` przyjmuje cala tabele
skanu, a `n_tests_override` pozwala jawnie doliczyc proby odrzucone po drodze.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("bonferroni", "bh", "none")


def _valid_mask(p: np.ndarray) -> np.ndarray:
    """Hipotezy, ktorych w ogole nie dalo sie przetestowac (np. puste okno).

    Zostaja w tabeli jako NaN, ale nie biora udzialu w korekcie: ani nie
    powiekszaja licznika testow, ani - co wazniejsze - nie moga zepsuc
    monotonizacji BH, ktora idzie od konca posortowanej listy.
    """
    return np.isfinite(p)


def bonferroni(p_values: np.ndarray | pd.Series, n_tests: int | None = None) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    valid = _valid_mask(p)
    n = n_tests or int(valid.sum())
    out = np.full(p.shape, np.nan)
    out[valid] = np.clip(p[valid] * n, 0.0, 1.0)
    return out


def benjamini_hochberg(
    p_values: np.ndarray | pd.Series, n_tests: int | None = None
) -> np.ndarray:
    """Wartosci q metoda BH (monotoniczne, obciete do 1)."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = _valid_mask(p)
    if not valid.any():
        return out

    tested = p[valid]
    m = n_tests or int(valid.sum())
    order = np.argsort(tested)
    ranked = tested[order]
    adjusted = ranked * m / (np.arange(1, len(ranked) + 1))
    # Wymuszenie monotonicznosci od konca - inaczej q moze malec wraz z p.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    out[valid] = restored
    return out


def correct(
    scan: pd.DataFrame,
    *,
    method: str = "bh",
    alpha: float = 0.05,
    p_column: str = "p_value",
    n_tests_override: int | None = None,
) -> pd.DataFrame:
    """Dokleja skorygowane p/q i decyzje do tabeli skanu hipotez."""
    if method not in METHODS:
        raise ValueError(f"nieznana metoda: {method} (dostepne: {METHODS})")
    out = scan.copy()
    if out.empty:
        return out
    testable = int(np.isfinite(np.asarray(out[p_column], dtype=float)).sum())
    n_tests = n_tests_override or testable

    if method == "bonferroni":
        out["p_adjusted"] = bonferroni(out[p_column], n_tests)
    elif method == "bh":
        out["p_adjusted"] = benjamini_hochberg(out[p_column], n_tests)
    else:
        out["p_adjusted"] = out[p_column].to_numpy()

    out["significant_raw"] = (out[p_column] < alpha).fillna(False)
    out["significant_adjusted"] = (out["p_adjusted"] < alpha).fillna(False)
    out.attrs["method"] = method
    out.attrs["alpha"] = alpha
    out.attrs["n_tests"] = n_tests
    out.attrs["n_untestable"] = int(len(out) - testable)
    return out.sort_values(p_column)


def expected_false_discoveries(n_tests: int, alpha: float = 0.05) -> float:
    """Ile "odkryc" da sam przypadek przy tylu testach - liczba do raportu."""
    return n_tests * alpha


def summarize(corrected: pd.DataFrame) -> str:
    if corrected.empty:
        return "brak hipotez do podsumowania"
    n_tests = corrected.attrs.get("n_tests", len(corrected))
    alpha = corrected.attrs.get("alpha", 0.05)
    raw = int(corrected["significant_raw"].sum())
    adjusted = int(corrected["significant_adjusted"].sum())
    untestable = corrected.attrs.get("n_untestable", 0)
    skipped = f" | pominiete (brak danych): {untestable}" if untestable else ""
    return (
        f"{n_tests} hipotez | istotne surowo: {raw} "
        f"(sam przypadek dalby ~{expected_false_discoveries(n_tests, alpha):.1f}) "
        f"| po korekcie {corrected.attrs.get('method', '?')}: {adjusted}{skipped}"
    )
