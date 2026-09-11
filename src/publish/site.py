"""Turn the lab into a folder of files a web server can hand out.

Why static rather than a hosted Streamlit: this app is a reader, not a
service. Everything it shows is computed by a scheduled job that already runs
on the machine holding the database, so a public server would add a Python
process, an open port and a copy of the data without adding an answer. What
it would add is a way to get the data wrong in public.

WHAT IS DELIBERATELY LEFT OUT. The control group - NASDAQ, S&P 500, gold -
comes from Yahoo Finance, whose terms forbid redistribution, and
DATA_SOURCES.md says so in as many words: hosting the dashboard publicly is
serving that data onward. The published site therefore carries the *finding*
from that comparison as a sentence, and none of the series behind it. Nothing
here writes a control price to disk.

The pages are self-contained: the panels are the same HTML the dashboard
embeds, fed the same JSON, so the site cannot drift from the app it was built
from. Plotly is loaded from its CDN because bundling it would quadruple the
upload for a file every visitor already has cached.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from analysis.evidence import build as build_evidence
from backtest.rank_history import MIN_OBSERVATIONS as MIN_RANK_OBSERVATIONS
from backtest.rank_history import placement as rank_placement
from forecast.ledger import load as load_ledger
from forecast.ledger import score as score_ledger
from forecast.ledger import scoreboard as ledger_scoreboard
from forecast.ledger import verdict as ledger_verdict
from publish.charts import (band_sweep_chart, cycle_clock, range_chart,
                            rank_history_chart, target_sweep_chart)
from publish.disclosure import (corrections_html, provenance_html,
                                rank_history_html, sweep_html,
                                target_sweep_html)
from publish.pages import cards as _cards
from publish.pages import (agent_in_plain_words, evidence_page, figure_html,
                           hero, how_far, lede, method, which_way, workings)
from publish.pages import table as _table
from publish.payloads import desk_payload

PANELS = {
    "intro": ("intro.html", "<!--INTRO:START-->", "<!--INTRO:END-->"),
    "desk": ("signal_desk.html", "<!--DESK:START-->", "<!--DESK:END-->"),
    "meter": ("evidence_meter.html", "<!--METER:START-->", "<!--METER:END-->"),
    "paper": ("paper_desk.html", "<!--PAPER:START-->", "<!--PAPER:END-->"),
    "agent": ("agent_desk.html", "<!--AGENT:START-->", "<!--AGENT:END-->"),
}
CHART_FITTER = """<script>
  // Plotly measures its container at the moment the figure is inserted, and
  // in a two-column row that moment is before the grid has settled: the first
  // chart is drawn at the full width of the page, keeps it, and spills over
  // the chart beside it. `responsive: true` only listens for window resizes,
  // and there is no window resize on the way in - so watch the containers
  // instead. The width check is what stops a resize from feeding itself.
  (function () {
    var widths = new WeakMap();
    function fit(node) {
      if (!window.Plotly || !node || !node.parentElement) { return; }
      var width = Math.round(node.parentElement.clientWidth);
      if (!width || widths.get(node) === width) { return; }
      widths.set(node, width);
      try { Plotly.Plots.resize(node); } catch (err) { /* not a plot yet */ }
    }
    function fitAll() {
      document.querySelectorAll(".js-plotly-plot").forEach(fit);
    }
    window.addEventListener("load", function () {
      fitAll();
      if (!window.ResizeObserver) { return; }
      var observer = new ResizeObserver(function (entries) {
        entries.forEach(function (entry) {
          var node = entry.target.querySelector(".js-plotly-plot");
          if (node) { fit(node); }
        });
      });
      document.querySelectorAll(".js-plotly-plot").forEach(function (node) {
        if (node.parentElement) { observer.observe(node.parentElement); }
      });
    });
  })();
</script>"""
DATA_START, DATA_END = "<!--DATA:START-->", "<!--DATA:END-->"
PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.min.js"

PAGES = (
    ("index.html", "Today"),
    ("now.html", "The cycle"),
    ("trader.html", "The agent"),
    ("evidence.html", "Evidence"),
    ("signals.html", "Signals"),
    ("receipts.html", "Receipts"),
)

HEIGHT_REPORTER = """<script>
  // Embedded in a page elsewhere - tell the host how tall this is, so the
  // frame can size itself instead of guessing and showing two scrollbars.
  (function () {
    function report() {
      try {
        parent.postMessage({ btclabHeight: document.documentElement.scrollHeight }, "*");
      } catch (err) { /* not embedded, or a host that will not listen */ }
    }
    window.addEventListener("load", report);
    window.addEventListener("resize", report);
    window.setInterval(report, 1500);
  })();
</script>"""




@dataclass
class SiteInputs:
    """Everything the pages need, computed once by the caller."""

    outlook: object
    signals: dict
    evidence: object
    study: object
    scan: pd.DataFrame
    curve_summary: pd.DataFrame
    control_note: str
    range_forecast: pd.DataFrame | None = None
    range_calibration: pd.DataFrame | None = None
    range_days: int = 30
    ledger: pd.DataFrame | None = None
    paper: dict | None = None
    # Saved results, read rather than recomputed: the site must not be able to
    # publish a number the terminal never printed.
    saved: dict = None


def panel(dashboard_dir: Path, name: str, payload: dict | None = None) -> str:
    """Lift a panel out of its dashboard file, optionally with data in it."""
    filename, start, end = PANELS[name]
    markup = (dashboard_dir / filename).read_text(encoding="utf-8")
    fragment = markup.split(start, 1)[1].split(end, 1)[0]
    if payload is None:
        return fragment
    head, rest = fragment.split(DATA_START, 1)
    _, tail = rest.split(DATA_END, 1)
    # A literal </script> inside the JSON would close the data block early.
    return head + json.dumps(payload).replace("</", r"<\/") + tail



def _nav(current: str) -> str:
    links = []
    for href, label in PAGES:
        active = ' class="on"' if label == current else ""
        links.append(f'<a href="{href}"{active}>{label}</a>')
    return "<nav>" + "".join(links) + "</nav>"


def _shell(title: str, current: str, body: str, *, as_of: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="{PLOTLY}"></script>
<style>
  /* One typeface, the reader's own. `-apple-system` resolves to SF Pro on a
     Mac or an iPhone and to Segoe UI Variable on Windows 11: the look is the
     system's, which is why it never looks downloaded, and it costs no request
     and sends nobody's address to a font host. Monospace survives only inside
     <code>. Everything else - headings, tables, the big figures - is the sans
     with tabular numerals, so columns still line up on the decimal point.

     Size and weight do the sorting the old page asked colour to do: a finding
     is 21px, a caveat 15px and folded away, and the space between sections is
     wide enough that a reader can tell where one answer ends. */
  :root{{--bg:#000308;--panel:#101219;--raise:#171a22;--line:#252935;
    --ink:#f4f6f9;--soft:#c3cad6;--dim:#8f99a8;--gold:#f7931a;--up:#2fcf83;--down:#ff5a5f;
    --sans:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display",
      "Segoe UI Variable Text","Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --r:16px}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);
    font:17px/1.6 var(--sans);letter-spacing:-.011em;
    -webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;
    -webkit-text-size-adjust:100%}}
  code{{font-family:var(--mono);font-size:.88em;color:var(--soft)}}
  a{{color:var(--gold);text-decoration:none}}
  a:hover{{text-decoration:underline}}
  header{{padding:44px 24px 0;max-width:1100px;margin:0 auto}}
  header h1{{margin:0;font-size:40px;font-weight:700;letter-spacing:-.024em;line-height:1.08}}
  header p{{margin:14px 0 0;color:var(--dim);font-size:17px;max-width:60ch;
    letter-spacing:-.01em}}
  nav{{margin:26px 0 0;display:flex;gap:7px;flex-wrap:wrap}}
  nav a{{padding:8px 16px;border-radius:999px;background:var(--panel);
    color:var(--soft);font-size:15px;font-weight:500;letter-spacing:-.01em;
    border:1px solid transparent;text-decoration:none}}
  nav a:hover{{background:var(--raise);color:var(--ink);text-decoration:none}}
  nav a.on{{background:var(--gold);color:#0a0600;font-weight:600}}
  main{{max-width:1100px;margin:0 auto;padding:0 24px 40px}}
  section{{margin:0;padding:56px 0 8px;border-top:1px solid var(--line)}}
  main>section:first-child{{border-top:0;padding-top:34px}}
  h2{{font-size:28px;font-weight:700;letter-spacing:-.021em;line-height:1.2;
    margin:0 0 18px}}
  main p{{max-width:64ch}}
  .lede{{font-size:21px;line-height:1.45;letter-spacing:-.015em;color:var(--soft);
    margin:18px 0 0;max-width:60ch}}
  .lede b{{color:var(--ink);font-weight:600}}
  /* min-width:0 is load-bearing. A grid track is minmax(auto,1fr), and `auto`
     resolves to min-content - so a Plotly SVG that renders wide pushes its own
     column wider than the page and takes the neighbouring chart with it. */
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:26px}}
  .grid>div{{min-width:0}}
  .grid>div>h2{{margin-top:0}}
  .js-plotly-plot,.plot-container{{max-width:100%}}
  .js-plotly-plot{{border-radius:var(--r);overflow:hidden;margin-top:22px}}

  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
  .card{{border-radius:var(--r);background:var(--panel);padding:16px 18px 18px}}
  .card u{{display:block;text-decoration:none;color:var(--dim);font-size:12px;
    font-weight:600;letter-spacing:.055em;text-transform:uppercase}}
  .card b{{display:block;margin-top:7px;font-size:32px;font-weight:600;
    letter-spacing:-.026em;line-height:1.12;font-variant-numeric:tabular-nums}}
  .hero .cards{{margin-bottom:22px}}
  .hero .card b{{font-size:34px}}

  /* Where the price can be, as a bar rather than a sentence with four numbers
     in it. The marker is today; the bar is the calibrated interval. */
  .ranges{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
    gap:16px;margin-top:26px}}
  .range{{background:var(--panel);border-radius:var(--r);padding:18px 20px 20px}}
  .range-top{{display:flex;justify-content:space-between;align-items:baseline;gap:12px}}
  .range-top b{{font-size:19px;font-weight:600;letter-spacing:-.016em}}
  .range-top span{{color:var(--dim);font-size:13px}}
  .range-bar{{position:relative;height:10px;border-radius:999px;margin:18px 0 12px;
    background:linear-gradient(90deg,rgba(76,139,245,.30),rgba(76,139,245,.62),
      rgba(76,139,245,.30))}}
  .range-bar i{{position:absolute;top:-5px;width:3px;height:20px;border-radius:2px;
    background:var(--gold);transform:translateX(-1.5px);
    box-shadow:0 0 0 3px rgba(0,3,8,.85)}}
  .range-ends{{display:flex;justify-content:space-between;
    font-variant-numeric:tabular-nums;font-size:17px;font-weight:600;
    letter-spacing:-.018em}}
  .range-ends u{{display:block;text-decoration:none;color:var(--dim);font-size:12.5px;
    font-weight:500;letter-spacing:0}}
  .range-ends span:last-child{{text-align:right}}

  /* The rest of the site, as doors rather than as a row of tabs nobody read. */
  .doors{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
    gap:12px;margin-top:24px}}
  .door{{display:block;background:var(--panel);border-radius:var(--r);padding:18px 20px;
    color:var(--ink);text-decoration:none;border:1px solid transparent}}
  .door:hover{{background:var(--raise);border-color:var(--line);text-decoration:none}}
  .door b{{display:block;font-size:19px;font-weight:600;letter-spacing:-.016em}}
  .door span{{display:block;margin-top:6px;color:var(--dim);font-size:14.5px;
    line-height:1.45}}

  table{{width:100%;border-collapse:collapse;font-size:15px;
    font-variant-numeric:tabular-nums;letter-spacing:-.008em}}
  th,td{{text-align:left;padding:12px 16px;border-bottom:1px solid var(--line);
    white-space:nowrap;vertical-align:baseline}}
  tbody tr:last-child td{{border-bottom:0}}
  tbody tr:hover td{{background:var(--raise)}}
  th{{color:var(--dim);font-weight:600;font-size:12px;letter-spacing:.05em;
    text-transform:uppercase;white-space:normal}}
  th abbr{{text-decoration:none;border-bottom:1px dotted #5b6675;cursor:help}}
  td.num,th.num,td.flag,th.flag{{text-align:right}}
  td.key{{color:var(--ink);font-weight:600;white-space:normal}}
  td.txt{{white-space:normal;font-size:15px;color:var(--dim);min-width:26ch}}
  /* Deliberately not green and red. "Rests on one trade: yes" and "survives
     correction: yes" want opposite colours, so a palette that means good and
     bad would lie in one column to be readable in the other. Weight carries
     the presence instead. */
  td .yes{{font-style:normal;font-weight:700;color:var(--ink)}}
  td .no{{font-style:normal;color:#71808f}}
  th.mark{{color:var(--gold)}}
  th.mark,td.mark{{background:rgba(247,147,26,.055)}}
  /* A table wider than its box gets a shadow at the edge it can scroll to,
     and loses it when it cannot. Pure CSS: the covering gradients scroll with
     the content, the shadows do not, so the shadow shows only once there is
     something past the edge. */
  .wrap{{overflow-x:auto;border-radius:var(--r);margin-top:22px;
    background:
      linear-gradient(to right,var(--panel) 30%,rgba(16,18,25,0)) left center,
      linear-gradient(to left,var(--panel) 30%,rgba(16,18,25,0)) right center,
      radial-gradient(farthest-side at 0 50%,rgba(0,0,0,.7),rgba(0,0,0,0)) left center,
      radial-gradient(farthest-side at 100% 50%,rgba(0,0,0,.7),rgba(0,0,0,0)) right center,
      var(--panel);
    background-repeat:no-repeat;
    background-size:44px 100%,44px 100%,16px 100%,16px 100%,100% 100%;
    background-attachment:local,local,scroll,scroll,local}}

  .note{{color:var(--dim);font-size:15px;line-height:1.6;margin-top:14px;max-width:66ch}}
  details.method{{margin-top:20px;border-radius:var(--r);background:var(--panel)}}
  details.method>summary{{cursor:pointer;padding:14px 18px;color:var(--soft);
    font-size:14px;font-weight:600;letter-spacing:-.006em;list-style:none}}
  details.method>summary::-webkit-details-marker{{display:none}}
  details.method>summary::before{{content:"+";color:var(--gold);margin-right:10px;
    font-weight:700}}
  details.method[open]>summary::before{{content:"\\2013"}}
  details.method>summary:hover{{color:var(--ink)}}
  details.method[open]>summary{{border-bottom:1px solid var(--line)}}
  details.method .note{{margin:0;padding:16px 18px;max-width:70ch}}

  .verdict{{border-radius:var(--r);background:#1b0f13;padding:20px 22px;
    color:#f8dade;font-size:18px;line-height:1.5;letter-spacing:-.012em;max-width:68ch;
    border:1px solid #4a1f28}}
  .verdict b{{color:#fff;font-weight:600}}

  /* Four sentences, one per line, each one a whole thought. Nothing about the
     robot's position needs a table or a chart to be said. */
  .steps{{list-style:none;margin:24px 0 0;padding:0;max-width:66ch}}
  .steps li{{position:relative;padding:14px 0 14px 26px;font-size:17px;line-height:1.55;
    border-bottom:1px solid var(--line);color:var(--soft)}}
  .steps li:last-child{{border-bottom:0}}
  .steps li::before{{content:"";position:absolute;left:2px;top:23px;width:7px;height:7px;
    border-radius:50%;background:var(--gold)}}
  .steps b{{color:var(--ink);font-weight:600}}

  footer{{max-width:1100px;margin:0 auto;padding:28px 24px 48px;color:var(--dim);
    font-size:13.5px;line-height:1.6;border-top:1px solid var(--line)}}
  footer a{{color:var(--dim)}}

  /* On a phone every one of these tables ran off the side of the screen with
     nothing to say so. A row becomes a card, and each cell carries the column
     name it was read under. */
  @media (max-width:640px){{
    header{{padding:30px 18px 0}}
    header h1{{font-size:32px}}
    header p{{font-size:16px}}
    main,footer{{padding-left:18px;padding-right:18px}}
    section{{padding-top:40px}}
    h2{{font-size:23px}}
    .lede{{font-size:19px}}
    /* Two across rather than four stacked: a phone screen of one number per
       scroll is a worse summary than a 2x2 a reader takes in at once. */
    .cards{{grid-template-columns:1fr 1fr;gap:10px}}
    .card{{padding:14px 15px 15px}}
    .card u{{font-size:11px;letter-spacing:.045em}}
    .card b,.hero .card b{{font-size:23px;letter-spacing:-.02em}}
    .verdict{{font-size:17px;padding:18px}}
    .wrap{{border-radius:0;background:none;overflow:visible}}
    table,tbody,tr,td{{display:block;width:auto}}
    thead{{display:none}}
    tr{{border-radius:var(--r);background:var(--panel);padding:4px 16px;margin:0 0 12px}}
    td{{display:flex;justify-content:space-between;align-items:baseline;gap:16px;
      padding:9px 0;border-bottom:1px solid var(--line);white-space:normal;
      text-align:right;line-height:1.45}}
    td:last-child{{border-bottom:0}}
    td::before{{content:attr(data-label);color:var(--dim);font-size:12px;
      font-weight:500;letter-spacing:.02em;text-transform:uppercase;text-align:left;
      flex:0 0 auto;max-width:50%;line-height:1.45}}
    td.key{{color:var(--gold);font-weight:700;font-size:17px;text-align:left;
      padding:14px 0 11px}}
    td.key::before{{display:none}}
    td.mark{{background:none}}
    td.txt{{min-width:0}}
  }}
</style>
</head>
<body>
<header>
  <h1>BTC Cycle Lab</h1>
  <p>Whether the halving cycle and macro events explain anything in the price of
  BTC. Every result carries a confidence interval and a count of observations;
  without those it is not a result. Sample to {as_of}.</p>
  {_nav(current)}
</header>
<main>
{body}
</main>
{CHART_FITTER}
{HEIGHT_REPORTER}
<footer>
  Research output, not advice, and not a signal to act on. Built from the local
  sample by <code>run.py publish</code>. The control-group comparison is
  computed but not published: its series come from Yahoo Finance, whose terms
  forbid redistribution.
</footer>
</body>
</html>
"""




def _band_sweep_section(inputs: SiteInputs) -> str:
    """The band sweep with its chart, or nothing if the sweep was never run."""
    saved = inputs.saved or {}
    sweep = saved.get("sizing_sweep", pd.DataFrame())
    sizing = saved.get("sizing_today", pd.DataFrame())
    if sweep.empty or sizing.empty or "band" not in sweep.columns:
        return ""
    adopted = float(sizing.iloc[0]["band"])
    # Frozen: the weight never changes again after the entry. See
    # backtest.sweep.FROZEN_CHANGES for why this counts rather than thresholds.
    quiet = (
        sweep.loc[sweep["n_position_changes"] <= 1, "band"]
        if "n_position_changes" in sweep.columns else pd.Series(dtype=float)
    )
    chart = figure_html(
        band_sweep_chart(sweep, adopted,
                         float(quiet.min()) if len(quiet) else None),
        "bandsweep",
    )
    return sweep_html(sweep, adopted, chart)


def _target_sweep_section(inputs: SiteInputs) -> str:
    """The volatility-target sweep with its chart, or nothing if never run."""
    saved = inputs.saved or {}
    sweep = saved.get("sizing_target_sweep", pd.DataFrame())
    sizing = saved.get("sizing_today", pd.DataFrame())
    if sweep.empty or sizing.empty or "target" not in sweep.columns:
        return ""
    adopted = sizing.iloc[0].get("target_annual_volatility")
    if adopted is None or pd.isna(adopted):
        return ""
    pinned = (
        sweep.loc[sweep["at_the_cap"] >= 0.99, "target"]
        if "at_the_cap" in sweep.columns else pd.Series(dtype=float)
    )
    chart = figure_html(
        target_sweep_chart(sweep, float(adopted),
                           float(pinned.min()) if len(pinned) else None),
        "targetsweep",
    )
    return target_sweep_html(sweep, float(adopted), chart)


def _rank_history_section(inputs: SiteInputs) -> str:
    """How the constants' placings have moved across runs, once there are any."""
    history = (inputs.saved or {}).get("rank_history", pd.DataFrame())
    if history.empty or "parameter" not in history.columns:
        return ""
    chart = ""
    if history["recorded"].nunique() >= MIN_RANK_OBSERVATIONS:
        chart = figure_html(rank_history_chart(rank_placement(history)), "rankhistory")
    return rank_history_html(history, chart, minimum=MIN_RANK_OBSERVATIONS)


def _track_record(scored: pd.DataFrame, *, base_rate: float | None) -> str:
    """The claims written before the outcome, counted on the front page.

    Everything else on this site is a backtest, written by someone who had
    already seen the answer, and every page says so. The ledger is the one
    exception, and it was the last tab: a reader could leave without ever
    learning that the site writes its forecasts down in advance and scores
    them when the window closes. That is the only part of it that can ever
    become evidence, so it belongs where it can be seen.

    The verdict line comes from `forecast.ledger`, not from here. It is phrased
    so a thin record cannot be read as a good one - under twenty settled claims
    it refuses to give a score at all - and rewriting that sentence for a
    summary card is exactly how a summary starts flattering its own data.
    """
    if scored.empty:
        return ""
    matured = scored["matured"].astype(bool)
    settled, open_claims = int(matured.sum()), int((~matured).sum())
    board = ledger_scoreboard(scored, base_rate=base_rate)

    pending = pd.to_datetime(scored.loc[~matured, "target_date"])
    when = f"{pending.min():%d %b %Y}" if len(pending) else "-"
    kept = "-"
    if not board.empty:
        intervals = board[board["claim"] == "interval"]
        if len(intervals):
            within = int((intervals["delivered"] >= intervals["promised"] - 0.1).sum())
            kept = f"{within} of {len(intervals)}"

    return (
        "<section><h2>Written down before the outcome</h2>"
        + lede("Every other page here is a backtest, scored by someone who had "
               "already seen the answer. <b>This one is not.</b> The claims are "
               "recorded while the window is still open, and settled when it "
               "closes.")
        + _cards([
            ("Claims recorded", f"{len(scored):,}"),
            ("Settled so far", f"{settled:,}"),
            ("Next one settles" if open_claims else "All settled", when),
            ("Levels that kept their promise", kept),
        ])
        + f"<div class='verdict' style='margin-top:20px'>"
          f"{ledger_verdict(board, open_claims)}</div>"
        + "<p class='note'>Every claim, with the day it was written and the day "
          "it settles, is on <a href='receipts.html'>the receipts page</a>.</p>"
        + "</section>"
    )


def build(destination: Path, dashboard_dir: Path, inputs: SiteInputs) -> list:
    """Write the site. Returns the paths written."""
    destination = Path(destination)
    # Delete only what this builder writes. It used to rmtree the whole folder,
    # which also ate the agent's feed: the agent writes agent_feed.json into
    # docs/ every quarter of an hour, so every morning's publish removed it and
    # the live page lost its journal until the next tick pushed it back. A
    # builder may clear its own output; it may not clear a directory it shares.
    destination.mkdir(parents=True, exist_ok=True)
    for stale in list(destination.glob("*.html")) + list(destination.glob(".nojekyll")):
        stale.unlink()
    # GitHub Pages runs Jekyll unless told not to, and Jekyll drops files it
    # does not recognise. Nothing here is a Jekyll site.
    (destination / ".nojekyll").write_text("", encoding="utf-8")

    outlook = inputs.outlook
    as_of = f"{outlook.as_of:%Y-%m-%d}"
    written = []
    conditional, unconditional = outlook.rates[0]

    # Scored once and read twice: the front page carries the count, the
    # receipts page carries the claims. Two calls would be two chances for the
    # summary and the detail to disagree about how many have settled.
    ledger = inputs.ledger if inputs.ledger is not None else pd.DataFrame()
    scored = (score_ledger(ledger, inputs.signals["price"])
              if not ledger.empty else pd.DataFrame())

    # --- the practical page ----------------------------------------------
    # First, because it answers the question a visitor arrives with. The intro
    # rides on this page only; a reader who came back for the tables should not
    # have to sit through it again.
    written.append(_write(
        destination / "index.html",
        _shell("BTC Cycle Lab", "Today",
               panel(dashboard_dir, "intro")
               + hero(inputs)
               + which_way(inputs)
               + how_far(inputs)
               + _track_record(scored, base_rate=unconditional.share_positive)
               + agent_in_plain_words(inputs)
               + "<section><h2>How much has actually survived the tests</h2>"
               + panel(dashboard_dir, "meter", inputs.evidence.as_dict())
               + "</section>"
               + workings(), as_of=as_of),
    ))

    # --- the board -------------------------------------------------------
    body = [
        "<section>",
        f"<h2>{outlook.days_since_halving} days after the last halving</h2>",
        _cards([
            ("Last close", f"${outlook.last_price:,.0f}"),
            ("Days since halving", f"{outlook.days_since_halving}"),
            ("Cycle phase", outlook.cycle_label.split()[0].capitalize()),
            ("200-day trend", "Above" if "above" in outlook.trend_label else "Below"),
        ]),
        "</section>",
        "<section><h2>How much has actually survived the tests</h2>",
        panel(dashboard_dir, "meter", inputs.evidence.as_dict()),
        "</section>",
        "<section class='grid'>",
        "<div><h2>The cycle, four laps on one dial</h2>"
        + figure_html(cycle_clock(outlook.paths, outlook.days_since_halving), "clock")
        + lede("The shape the laps share is why people believe in the cycle; the "
               "distance between them is why <b>four of them cannot prove it</b>.")
        + method("How the dial is drawn",
                 "Each lap starts at its own halving, drawn as a multiple of the "
                 "price on that day.")
        + "</div>",
    ]

    if inputs.range_forecast is not None and not inputs.range_forecast.empty:
        quoted = ", ".join(f"{level:.0%}" for level in inputs.range_forecast["level"])
        withheld = ""
        if inputs.range_calibration is not None:
            failed = inputs.range_calibration.loc[
                ~inputs.range_calibration["within_tolerance"], "level"]
            if len(failed):
                withheld = ("; " + ", ".join(f"{level:.0%}" for level in failed)
                            + " failed its coverage test and is withheld")
        body.append(
            "<div><h2>Where the price may be</h2>"
            + figure_html(range_chart(inputs.range_forecast, inputs.signals["price"],
                                      inputs.range_days), "range")
            + lede("It says <b>how far</b>, never which way.")
            + method(
                "Which levels are drawn, and which are not",
                f"{inputs.range_days}-day interval from the volatility model, drawn "
                f"only at the horizon it was scored at. Levels shown ({quoted}) kept "
                f"their promise in a walk-forward coverage test{withheld}.",
            )
            + "</div>"
        )
    body.append("</section>")

    rates = pd.DataFrame([
        {
            "horizon": f"{matched.horizon}d",
            "median_matched": round(100 * matched.median, 1),
            "positive_matched": round(100 * matched.share_positive),
            "independent_windows": matched.effective_n,
            "median_all": round(100 * whole.median, 1),
            "positive_all": round(100 * whole.share_positive),
            "quotable": "yes" if matched.usable else "no - too few windows",
        }
        for matched, whole in outlook.rates
    ])
    body += [
        "<section><h2>What usually happened from days like today</h2>",
        lede("The long-horizon rows are the ones that look most impressive and "
             "<b>rest on the fewest windows</b>."),
        _table(rates, highlight="independent_windows"),
        method(
            "Why the window count is the column to read",
            "Matched days are not observations: consecutive days share almost all "
            "of their forward window, so only non-overlapping ones are counted.",
        ),
        "</section>",
    ]

    rules = pd.DataFrame([
        {
            "rule": item["report"].name,
            "position": "in" if item["report"].side == "long" else "out",
            "since": "-" if item["report"].since is None else f"{item['report'].since:%Y-%m-%d}",
            "what flips it": item["report"].trigger.text,
        }
        for item in inputs.signals["strategies"]
    ])
    verdict_lines = [
        f"Halving effect after a year: {inputs.study.car_summary['car']:+.0%} "
        f"[{inputs.study.car_summary['ci_low']:+.0%}, "
        f"{inputs.study.car_summary['ci_high']:+.0%}], "
        f"p={inputs.study.car_summary['p_value']:.2f}, n={inputs.study.n_events}.",
        f"{len(inputs.scan)} hypotheses scanned, "
        f"{int(inputs.scan['significant_adjusted'].sum())} significant after correction.",
    ]
    if not inputs.curve_summary.empty:
        row = inputs.curve_summary.iloc[0]
        verdict_lines.append(
            f"Specification curve: {int(row['n_significant'])} of {int(row['n_specs'])} "
            f"specifications significant, against {row['null_significant_mean']:.1f} "
            "for randomly placed dates."
        )
    verdict_lines.append(inputs.control_note)
    body += [
        "<section><h2>What the rules say, and what the tests say</h2>",
        lede("Where each rule stands today, and underneath it what the tests did "
             "to the rules themselves."),
        _table(rules, highlight="position"),
        "<div class='verdict' style='margin-top:16px'>"
        + "<br>".join(verdict_lines)
        + "<br>Read all of it as description, not as a plan.</div></section>",
    ]

    written.append(_write(destination / "now.html",
                          _shell("BTC Cycle Lab - the cycle", "The cycle", "".join(body), as_of=as_of)))

    # --- the paper portfolio ---------------------------------------------
    if inputs.paper:
        equity = inputs.paper.get("curve")
        chart = ""
        if equity is not None and not equity.empty:
            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=equity.index, y=equity["equity"], mode="lines", name="the portfolio",
                line={"color": "#20c97e", "width": 2},
            ))
            figure.add_trace(go.Scatter(
                x=equity.index, y=equity["buy_and_hold"], mode="lines",
                name="just holding BTC", line={"color": "#8b98a9", "width": 1.4, "dash": "dot"},
            ))

            # Every trade marked where it happened. The table below says when the
            # agent acted; without these a reader cannot see what the action did
            # to the curve, which is the only thing that matters about it.
            for action, colour, symbol in (
                ("buy", "#20c97e", "triangle-up"), ("sell", "#ef4056", "triangle-down"),
            ):
                marks = [t for t in inputs.paper["payload"]["trades"]
                         if t["action"] == action]
                if not marks:
                    continue
                days = pd.to_datetime([t["date"] for t in marks])
                figure.add_trace(go.Scatter(
                    x=days,
                    y=[equity["equity"].asof(day) for day in days],
                    mode="markers", name=action,
                    marker={"symbol": symbol, "size": 11, "color": colour,
                            "line": {"color": "#05070a", "width": 1.5}},
                    customdata=[[t["price"], t["weight_to"] * 100] for t in marks],
                    hovertemplate=(action + " at $%{customdata[0]:,.0f}<br>"
                                   "position after: %{customdata[1]:.0f}%<br>"
                                   "portfolio: $%{y:,.0f}<extra></extra>"),
                ))

            figure.update_layout(
                height=340, margin={"l": 56, "r": 16, "t": 20, "b": 40},
                hovermode="x unified", yaxis_title="USD",
                legend={"orientation": "h", "y": -0.18},
            )
            chart = figure_html(figure, "paper")

        trader_body = (
            "<section><h2>What the agent is doing right now</h2>"
            + panel(dashboard_dir, "agent")
            + method(
                "Why it watches often and acts rarely",
                "It looks at the price every fifteen minutes and writes down what "
                "it sees, whether or not anything happens. The position changes "
                "only when a daily bar settles: the rule was tested on daily "
                "closes, and acting on an intraday tick would be an untested rule "
                "borrowing a tested one's credibility.",
            )
            + "</section>"
            + "<section><h2>A virtual portfolio, in public</h2>"
            + panel(dashboard_dir, "paper", inputs.paper["payload"])
            + "</section><section><h2>Against simply holding</h2>"
            + chart
            + lede("The dotted line is what doing nothing would have earned, and "
                   "it is <b>the only benchmark that matters</b>.")
            + method(
                "What the portfolio is made of",
                "Virtual money, started on the last halving - a date fixed by the "
                "subject rather than chosen after seeing the curve. Direction comes "
                "from the 50/200 crossover, which this project shows has no "
                "demonstrated edge; size comes from the volatility forecast, which "
                "does. Costs are charged on every change.",
            )
            + "</section>"
            + provenance_html((inputs.saved or {}).get("sizing_today", pd.DataFrame()),
                              (inputs.saved or {}).get("sizing_comparison", pd.DataFrame()))
            + _band_sweep_section(inputs)
            + _target_sweep_section(inputs)
            + _rank_history_section(inputs)
            + corrections_html("trader.html")
        )
        written.append(_write(destination / "trader.html",
                              _shell("BTC Cycle Lab - the agent", "The agent",
                                     trader_body, as_of=as_of)))

    written.append(_write(destination / "evidence.html",
                          _shell("BTC Cycle Lab - evidence", "Evidence",
                                 evidence_page(inputs), as_of=as_of)))

    # --- the desk --------------------------------------------------------
    desk_body = (
        "<section><h2>What each rule says right now</h2>"
        + "<div style='height:640px'>"
        + panel(dashboard_dir, "desk", desk_payload(inputs.signals))
        + "</div>"
        + method(
            "Where the numbers on the panel come from",
            "The live quote comes from Binance and is used only for the distance "
            "to a trigger; every statistic on the panel is from the local sample. "
            "Turn sound on to hear entries and exits during a replay.",
        )
        + "</section>"
    )
    written.append(_write(destination / "signals.html",
                          _shell("BTC Cycle Lab - signals", "Signals", desk_body, as_of=as_of)))

    # --- the receipts ----------------------------------------------------
    # `scored` was built once at the top: the count on the front page and the
    # rows here have to come from the same scoring pass.
    if scored.empty:
        receipts = ("<section><h2>Receipts</h2><p class='note'>No claim has been "
                    "recorded yet.</p></section>")
    else:
        open_claims = int((~scored["matured"].astype(bool)).sum())
        board = ledger_scoreboard(scored, base_rate=unconditional.share_positive)
        shown = scored.assign(
            as_of=scored["as_of"].dt.strftime("%Y-%m-%d"),
            settles=scored["target_date"].dt.strftime("%Y-%m-%d"),
        ).loc[:, ["as_of", "settles", "horizon", "claim", "level", "low", "high",
                  "p_up", "matured", "realised"]].round(4)
        receipts = (
            "<section><h2>Receipts</h2>"
            + lede("Every other page is a backtest, written by someone who had "
                   "already seen the outcome. <b>This one is the exception</b>: "
                   "claims recorded before their window closed, scored when it closes.")
            + f"<div class='verdict' style='margin:16px 0'>"
              f"{ledger_verdict(board, open_claims)}</div>"
            + _table(shown, highlight="realised")
            + ("" if board.empty else "<h2 style='margin-top:26px'>Settled</h2>"
               + _table(board.round(3)))
            + "</section>"
        )
    written.append(_write(destination / "receipts.html",
                          _shell("BTC Cycle Lab - receipts", "Receipts", receipts, as_of=as_of)))
    return written


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path
