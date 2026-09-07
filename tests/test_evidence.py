"""The evidence meter: a summary number built so it cannot be talked up.

A single score is the easiest thing in a research tool to abuse, and these
tests pin the three defences: a result worse than chance earns nothing, a
missing input earns nothing and says so, and the weights stay equal.
"""
from __future__ import annotations

import pytest

from analysis.evidence import WEIGHTS, build
from config import REPO_ROOT

METER = REPO_ROOT / "dashboard" / "evidence_meter.html"
APP = REPO_ROOT / "dashboard" / "app.py"


def test_a_project_that_found_nothing_scores_near_zero():
    """The real numbers of this project, and the reading they deserve."""
    score = build(
        survivors=0, hypotheses=26,
        specifications_significant=2, specifications_null_mean=15.1,
        halving_p_value=0.30,
        replicated=1, replication_attempts=23,
    )

    assert score.score < 5
    assert score.label == "NOISE"


def test_random_dates_beating_the_real_ones_earns_zero_not_a_fraction():
    """Below the null is evidence against, and must not be scored as partial credit."""
    score = build(specifications_significant=2, specifications_null_mean=15.1)
    robustness = next(c for c in score.components if c.name == "robustness")

    assert robustness.value == 0.0


def test_a_p_value_above_the_threshold_earns_nothing():
    weak = build(halving_p_value=0.30)
    strong = build(halving_p_value=0.001)

    assert next(c for c in weak.components if c.name == "event_strength").value == 0.0
    assert next(c for c in strong.components if c.name == "event_strength").value > 0.9


def test_an_empty_track_record_is_marked_as_missing_not_perfect():
    """No settled claim is not a clean sheet."""
    score = build(settled_claims=0, kept_claims=0)
    record = next(c for c in score.components if c.name == "track_record")

    assert record.value == 0.0
    assert not record.has_data
    assert "no claim has matured" in record.detail


def test_everything_passing_reaches_the_top_band():
    score = build(
        survivors=26, hypotheses=26,
        specifications_significant=40, specifications_null_mean=15.1,
        halving_p_value=0.0,
        replicated=23, replication_attempts=23,
        settled_claims=30, kept_claims=30,
    )

    assert score.score == 100.0
    assert score.label == "TELL SOMEONE"


def test_weights_are_equal():
    """Unequal weights would let a tuned component carry the total.

    Changing them is allowed; doing it silently is not, so the test states the
    expectation rather than reading it back from the dict it is testing.
    """
    assert set(WEIGHTS) == {"survivors", "robustness", "event_strength",
                            "replication", "track_record"}
    assert len(set(WEIGHTS.values())) == 1


def test_components_are_reported_with_the_score():
    score = build(survivors=0, hypotheses=26).as_dict()

    assert score["components"]
    assert all("detail" in component for component in score["components"])
    assert set(score) == {"score", "label", "blurb", "components"}


# --- the panel -----------------------------------------------------------------


def test_meter_markers_match_the_dashboard():
    markup = METER.read_text(encoding="utf-8")
    source = APP.read_text(encoding="utf-8")
    for marker in ("<!--METER:START-->", "<!--METER:END-->",
                   "<!--DATA:START-->", "<!--DATA:END-->"):
        assert markup.count(marker) == 1
        assert marker in source


def test_meter_says_what_it_is_not():
    """A single score with a needle reads as a signal unless it says otherwise."""
    markup = METER.read_text(encoding="utf-8").lower()
    assert "never which way a price goes" in markup
