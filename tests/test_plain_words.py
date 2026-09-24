"""The site is read by people who do not have the vocabulary it is written in.

Every check here is about that: machine names turned into sentences, a page
saying what it is for, and a claim that cannot flip tomorrow saying so rather
than printing a dash.
"""
from __future__ import annotations

from publish.pages import TERMS, glossary, plain, plain_hypothesis
from publish.site import SUBTITLES


def test_a_hypothesis_reads_as_a_sentence():
    assert plain_hypothesis("event_credit_event_30d") == "The 30 days after a credit shock"
    assert plain_hypothesis("halving_after_365d") == "The year after a halving"
    assert plain_hypothesis("halving_after_90d") == "The 90 days after a halving"
    assert plain_hypothesis("phase_expanding_rising").startswith("Days with money supply")


def test_an_unknown_identifier_still_loses_its_underscores():
    """A name the map has not met must not reach the page as code."""
    assert "_" not in plain_hypothesis("something_nobody_mapped_yet")


def test_every_page_says_what_it_is_for():
    """The header used to carry the same sentence about method on all of them."""
    for label in ("Today", "The cycle", "The agent", "Evidence", "Signals", "Receipts"):
        assert len(SUBTITLES[label]) > 40


def test_the_glossary_defines_the_words_the_tables_use():
    words = [word.lower() for word, _ in TERMS]
    text = glossary()

    assert "p-value" in words
    assert "sharpe ratio" in words
    for word, meaning in TERMS:
        assert word in text and meaning[:30] in text


def test_a_plain_translation_is_marked_as_one():
    assert plain("Anything.").startswith("<p class='plain'><b>In plain words.</b>")
