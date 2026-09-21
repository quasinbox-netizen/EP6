"""The cloud decides only when the daily refresh runs, so that is what is tested."""
from __future__ import annotations

import importlib.util
from datetime import datetime

from config import REPO_ROOT

spec = importlib.util.spec_from_file_location("cloud_run", REPO_ROOT / "scripts" / "cloud_run.py")
cloud_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud_run)


def _with_marker(tmp_path, monkeypatch, text):
    marker = tmp_path / "daily_last_run.txt"
    if text is not None:
        marker.write_text(text, encoding="utf-8")
    monkeypatch.setattr(cloud_run, "MARKER", marker)


def test_nothing_runs_before_eight(tmp_path, monkeypatch):
    _with_marker(tmp_path, monkeypatch, None)
    assert not cloud_run.daily_due(datetime(2026, 9, 21, 7, 59), forced=False)


def test_a_late_tick_still_picks_up_the_day(tmp_path, monkeypatch):
    """GitHub drops scheduled runs when busy; the next tick must catch up."""
    _with_marker(tmp_path, monkeypatch, "2026-09-20 1 ok")
    assert cloud_run.daily_due(datetime(2026, 9, 21, 13, 30), forced=False)


def test_a_finished_day_is_not_run_again(tmp_path, monkeypatch):
    _with_marker(tmp_path, monkeypatch, "2026-09-21 1 ok")
    assert not cloud_run.daily_due(datetime(2026, 9, 21, 9, 0), forced=False)


def test_a_failed_day_is_retried_but_not_forever(tmp_path, monkeypatch):
    _with_marker(tmp_path, monkeypatch, "2026-09-21 2 failed")
    assert cloud_run.daily_due(datetime(2026, 9, 21, 9, 0), forced=False)
    _with_marker(tmp_path, monkeypatch, "2026-09-21 3 failed")
    assert not cloud_run.daily_due(datetime(2026, 9, 21, 9, 0), forced=False)


def test_a_manual_run_always_refreshes(tmp_path, monkeypatch):
    _with_marker(tmp_path, monkeypatch, "2026-09-21 1 ok")
    assert cloud_run.daily_due(datetime(2026, 9, 21, 3, 0), forced=True)
