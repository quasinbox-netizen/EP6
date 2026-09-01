"""Wczytywanie konfiguracji i sekretow.

Sekrety ida wylacznie przez zmienne srodowiskowe (.env); config.yaml jest
wersjonowany i nie moze zawierac kluczy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Minimalny loader .env - bez zaleznosci, zeby import configu byl tani."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, key: str) -> Path:
        """Sciezka z sekcji `paths`, rozwinieta wzgledem katalogu repo."""
        return self.root / self.raw["paths"][key]

    @property
    def db_path(self) -> Path:
        return self.path("db")


@lru_cache(maxsize=8)
def load_config(path: str | Path | None = None) -> Config:
    root = REPO_ROOT
    cfg_path = Path(path) if path else root / "config.yaml"
    _load_dotenv(root / ".env")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return Config(raw=raw, root=root)


def secret(name: str) -> str | None:
    """Zwraca sekret ze srodowiska albo None. Nigdy nie loguje wartosci."""
    load_config()  # gwarantuje, ze .env zostal wczytany
    value = os.environ.get(name)
    return value or None
