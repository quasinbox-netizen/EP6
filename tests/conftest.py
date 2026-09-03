"""Shared test fixtures.

No test reaches the network or the production database: the data is synthetic
and the database temporary. Network tests (marked `network`) are run
deliberately: pytest -m network.
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
    """Pure noise: 3000 days of a random walk with no pattern at all."""
    return random_walk_prices(3000, start="2013-01-01", seed=7)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.sqlite"
