"""The intro overlay is decoration, but the seam between two files is not.

intro.html has to work in two places at once: opened on its own in a browser,
and cut in half by dashboard/app.py and dropped over the running app. The
markers are the contract between those two uses, and everything here is a way
that contract broke while it was being built.
"""
from __future__ import annotations

import pytest

from config import load_config

from .test_language import DIACRITICS

ROOT = load_config().root
INTRO = ROOT / "dashboard" / "intro.html"
APP = ROOT / "dashboard" / "app.py"
START, END = "<!--INTRO:START-->", "<!--INTRO:END-->"


@pytest.fixture(scope="module")
def markup() -> str:
    return INTRO.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def fragment(markup: str) -> str:
    """Exactly what app.py hands to Streamlit."""
    return markup.split(START, 1)[1].split(END, 1)[0]


def test_markers_exist_in_order(markup: str):
    assert markup.count(START) == 1 and markup.count(END) == 1
    assert markup.index(START) < markup.index(END)


def test_fragment_is_self_contained(fragment: str):
    """The overlay carries its own styling, canvas and code."""
    assert 'id="btclab-intro"' in fragment
    assert "<canvas>" in fragment
    assert "<style>" in fragment and "<script>" in fragment


def test_standalone_flag_stays_outside_the_fragment(markup: str, fragment: str):
    """The flag decides whether the intro loops or gets out of the way.

    Standalone it replays, because otherwise the page goes black and stays
    black. Inside the app it removes itself, because otherwise the dashboard
    is never reachable. The flag is set above the fragment on purpose: if it
    ever drifted between the markers, the app would inherit the loop and the
    dashboard would be unusable.
    """
    setter = "window.__BTCLAB_STANDALONE__ = true"
    assert markup.count(setter) == 1
    assert markup.index(setter) < markup.index(START)
    assert setter not in fragment
    # The fragment still reads the flag - it just never sets it.
    assert "__BTCLAB_STANDALONE__" in fragment


def test_app_extracts_the_fragment_with_the_same_markers():
    source = APP.read_text(encoding="utf-8")
    assert f'INTRO_START, INTRO_END = "{START}", "{END}"' in source
    assert "play_intro" in source


def test_intro_plays_once_per_session():
    """A splash screen on every rerun would make the app unusable."""
    source = APP.read_text(encoding="utf-8")
    assert "intro_pending" in source
    assert "st.session_state" in source


def test_no_polish_diacritics(markup: str):
    """test_language.py scans code and prose suffixes, not .html.

    The detector is imported rather than restated: a second copy of that
    character class in this file is itself a Polish string, and the language
    guard rightly flags it.
    """
    assert not DIACRITICS.search(markup)
