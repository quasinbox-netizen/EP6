"""Wspolne fixtury testow.

Zaden test nie chodzi do sieci ani do bazy produkcyjnej: dane sa
syntetyczne, baza tymczasowa. Testy sieciowe (oznaczone `network`)
uruchamiasz swiadomie: pytest -m network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from validation.synthetic import random_walk_prices  # noqa: E402


@pytest.fixture
def prices() -> pd.DataFrame:
    """Czysty szum: 3000 dni bladzenia losowego bez zadnego wzorca."""
    return random_walk_prices(3000, start="2013-01-01", seed=7)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.sqlite"
