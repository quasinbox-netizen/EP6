"""The live agent: what it may do, and what it may only watch.

The rule it follows was scored on daily closes - by the coverage walk, the
random-timing comparison and the cost model alike. An agent that acted on an
intraday tick would be running an untested strategy while wearing the
credibility of a tested one, so the tests here are mostly about the line
between observing and trading.
"""
from __future__ import annotations

import pandas as pd
import pytest

from agent.live import (
    ARMED_BAND,
    JOURNAL_LIMIT,
    append_observation,
    feed,
    load_journal,
    observe,
    should_publish,
)

NOW = pd.Timestamp("2026-09-07 12:00:00")


def test_a_flat_agent_arms_when_the_price_rises_through_the_level():
    seen = observe(75_000.0, weight=0.0, flip_level=71_804.0, now=NOW)

    assert seen.state == "armed"
    assert "next settled close" in seen.note
    assert "enters" in seen.note


def test_a_holding_agent_arms_when_the_price_falls_through_the_level():
    seen = observe(60_000.0, weight=0.8, flip_level=71_804.0, now=NOW)

    assert seen.state == "armed"
    assert "exits" in seen.note


def test_armed_says_the_close_acts_not_that_the_agent_did():
    """The whole boundary in one assertion: it is a statement about later."""
    seen = observe(75_000.0, weight=0.0, flip_level=71_804.0, now=NOW)

    assert "the next settled close" in seen.note
    # No wording that could be read as a trade having happened.
    for word in ("bought", "sold", "entered", "exited"):
        assert word not in seen.note


def test_a_price_just_short_of_the_level_is_near_not_armed():
    level = 100_000.0
    seen = observe(level * (1 - ARMED_BAND / 2), weight=0.0, flip_level=level, now=NOW)

    assert seen.state == "near"


def test_a_price_far_from_the_level_is_just_the_position():
    assert observe(50_000.0, weight=0.0, flip_level=100_000.0, now=NOW).state == "flat"
    assert observe(50_000.0, weight=0.8, flip_level=10_000.0, now=NOW).state == "holding"


def test_a_rule_without_a_price_trigger_still_records_something():
    seen = observe(50_000.0, weight=0.0, flip_level=None,
                   flip_text="waits for the next halving", now=NOW)

    assert seen.state == "flat"
    assert seen.note == "waits for the next halving"
    assert seen.distance is None


def test_the_journal_keeps_appending(tmp_path):
    path = tmp_path / "journal.csv"
    for minute in range(3):
        append_observation(path, observe(
            70_000.0 + minute, weight=0.0, flip_level=71_804.0,
            now=NOW + pd.Timedelta(minutes=minute)))

    assert len(load_journal(path)) == 3


def test_the_journal_is_trimmed_rather_than_left_to_grow(tmp_path):
    path = tmp_path / "journal.csv"
    rows = pd.DataFrame([
        observe(70_000.0, weight=0.0, flip_level=71_804.0,
                now=NOW + pd.Timedelta(minutes=i)).as_row()
        for i in range(JOURNAL_LIMIT + 5)
    ])
    rows.to_csv(path, index=False)

    trimmed = append_observation(path, observe(
        70_001.0, weight=0.0, flip_level=71_804.0, now=NOW))

    assert len(trimmed) == JOURNAL_LIMIT
    # The newest row survived the trim; the oldest did not.
    assert trimmed.iloc[-1]["price"] == 70001.0


def test_publishing_waits_unless_something_changed():
    journal = pd.DataFrame({"state": ["flat", "flat"]})
    publish, reason = should_publish(
        journal, last_published=NOW - pd.Timedelta(minutes=20), now=NOW)

    assert not publish
    assert "unchanged" in reason


def test_a_change_of_state_publishes_immediately():
    journal = pd.DataFrame({"state": ["flat", "armed"]})
    publish, reason = should_publish(
        journal, last_published=NOW - pd.Timedelta(minutes=1), now=NOW)

    assert publish
    assert "flat to armed" in reason


def test_a_quiet_agent_still_refreshes_on_the_heartbeat():
    journal = pd.DataFrame({"state": ["flat", "flat"]})
    publish, _ = should_publish(
        journal, last_published=NOW - pd.Timedelta(hours=4), now=NOW)

    assert publish


def test_the_feed_is_newest_first(tmp_path):
    path = tmp_path / "journal.csv"
    for minute in range(4):
        append_observation(path, observe(
            70_000.0 + minute, weight=0.0, flip_level=71_804.0,
            now=NOW + pd.Timedelta(minutes=minute)))

    payload = feed(load_journal(path))

    assert payload["observations"][0]["price"] == 70003.0
    assert payload["updated"].endswith("12:03:00")
