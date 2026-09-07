"""The published site: what it must contain, and what it must never contain.

The second half is the reason this file exists. The dashboard runs on the
machine that owns the data, so the licence question never comes up there. The
site is served to strangers, and DATA_SOURCES.md is explicit: Yahoo Finance
permits personal use and forbids redistribution, so hosting the control-group
series publicly is serving that data onward. A test is the only thing that
keeps that promise once someone adds a panel in a hurry.
"""
from __future__ import annotations

import pandas as pd
import pytest

from analysis.evidence import build as build_evidence
from config import REPO_ROOT
from pipeline import LabData, cycle_outlook, halving_event_study, strategy_signals
from publish.site import PAGES, SiteInputs, build, panel
from validation.synthetic import random_walk_prices

DASHBOARD = REPO_ROOT / "dashboard"

# Things that would identify a control series in the output. Bare "GOLD" is
# deliberately not here: it is also a colour constant in the panels, and a
# marker that fires on the intro's palette is a test nobody trusts.
CONTROL_MARKERS = ("NASDAQ", "^GSPC", "^IXIC", "GC=F", "control_GOLD", "control_SP500")


@pytest.fixture(scope="module")
def data() -> LabData:
    prices = random_walk_prices(2600, start="2018-01-01", seed=4)
    features = prices.set_index("date")[["close"]]
    features["ma_200_ratio"] = features["close"] / features["close"].rolling(200).mean()
    from features.halving import halving_features
    features = features.join(halving_features(features.index, strict=True))
    empty = pd.DataFrame()
    return LabData(prices=prices, macro=empty, events=empty, features=features)


@pytest.fixture(scope="module")
def site(tmp_path_factory, data) -> dict:
    destination = tmp_path_factory.mktemp("site")
    signals = strategy_signals(data)
    inputs = SiteInputs(
        outlook=cycle_outlook(data),
        signals=signals,
        evidence=build_evidence(survivors=0, hypotheses=26, halving_p_value=0.3),
        study=halving_event_study(data, post=365),
        scan=pd.DataFrame({"significant_adjusted": [False] * 26}),
        curve_summary=pd.DataFrame(),
        control_note="The control group is described here, not published.",
    )
    written = build(destination, DASHBOARD, inputs)
    return {
        "paths": written,
        "text": {path.name: path.read_text(encoding="utf-8") for path in written},
    }


def test_every_page_is_written(site):
    assert {path.name for path in site["paths"]} == {name for name, _ in PAGES}


def test_no_control_series_reaches_the_output(site):
    """The finding may be published; the licensed data may not."""
    for name, text in site["text"].items():
        for marker in CONTROL_MARKERS:
            assert marker not in text, f"{marker} leaked into {name}"


def test_the_site_has_nowhere_to_put_control_data(site):
    """Structural, not textual: the builder cannot be handed the series at all.

    A leak would have to add a field first, and adding one is a visible change
    in a diff rather than a line of formatting nobody reads.
    """
    control_fields = [
        name for name in SiteInputs.__dataclass_fields__ if "control" in name
    ]
    assert control_fields == ["control_note"]


def test_the_licence_reason_is_stated_on_every_page(site):
    for text in site["text"].values():
        assert "forbid redistribution" in text


def test_the_pages_say_they_are_not_advice(site):
    for text in site["text"].values():
        assert "not advice" in text


def test_the_panels_are_embedded_rather_than_linked(site):
    """A page that fetched its panel would break the moment it moved."""
    index = site["text"]["index.html"]
    assert "btclab-intro" in index      # the intro plays on the front page
    assert "btclab-meter" in index      # the evidence gauge is inline
    assert "btclab-desk" in site["text"]["signals.html"]


def test_the_desk_keeps_its_live_quote(site):
    """The one thing the static site still fetches at run time."""
    assert "api.binance.com" in site["text"]["signals.html"]


def test_a_panel_can_be_lifted_without_data(site):
    fragment = panel(DASHBOARD, "intro")

    assert "<canvas>" in fragment
    assert "INTRO:START" not in fragment  # markers stay behind


def test_building_twice_replaces_rather_than_accumulates(tmp_path, data):
    """A stale page from a previous build is worse than a missing one."""
    signals = strategy_signals(data)
    inputs = SiteInputs(
        outlook=cycle_outlook(data), signals=signals,
        evidence=build_evidence(hypotheses=1), study=halving_event_study(data, post=365),
        scan=pd.DataFrame({"significant_adjusted": [False]}),
        curve_summary=pd.DataFrame(), control_note="note",
    )
    destination = tmp_path / "site"
    build(destination, DASHBOARD, inputs)
    (destination / "stale.html").write_text("old", encoding="utf-8")
    build(destination, DASHBOARD, inputs)

    assert not (destination / "stale.html").exists()
