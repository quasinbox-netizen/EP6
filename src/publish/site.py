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

from analysis.evidence import build as build_evidence
from forecast.ledger import load as load_ledger
from forecast.ledger import score as score_ledger
from forecast.ledger import scoreboard as ledger_scoreboard
from forecast.ledger import verdict as ledger_verdict
from publish.charts import cycle_clock, range_chart
from publish.payloads import desk_payload

PANELS = {
    "intro": ("intro.html", "<!--INTRO:START-->", "<!--INTRO:END-->"),
    "desk": ("signal_desk.html", "<!--DESK:START-->", "<!--DESK:END-->"),
    "meter": ("evidence_meter.html", "<!--METER:START-->", "<!--METER:END-->"),
}
DATA_START, DATA_END = "<!--DATA:START-->", "<!--DATA:END-->"
PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.min.js"

PAGES = (("index.html", "Now"), ("signals.html", "Signals"), ("receipts.html", "Receipts"))

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


def figure_html(figure, element_id: str) -> str:
    """A Plotly figure as a div, with the library loaded once per page."""
    figure.update_layout(
        paper_bgcolor="#080b11", plot_bgcolor="#080b11",
        font={"color": "#c3ccd8", "family": "ui-monospace,Consolas,monospace"},
    )
    return figure.to_html(
        full_html=False, include_plotlyjs=False, div_id=element_id,
        config={"displayModeBar": False, "responsive": True},
    )


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
  :root{{--bg:#05070a;--panel:#0d1219;--line:#1c2530;--ink:#e7ecf3;--dim:#8b98a9;--gold:#f7931a}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);
    font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
  header{{padding:22px 24px 0;max-width:1180px;margin:0 auto}}
  header h1{{margin:0;font-size:26px;letter-spacing:.02em}}
  header p{{margin:6px 0 0;color:var(--dim);font-size:12.5px;max-width:70ch}}
  nav{{margin:16px 0 0;display:flex;gap:6px;flex-wrap:wrap}}
  nav a{{padding:6px 12px;border:1px solid var(--line);border-radius:999px;
    color:var(--dim);text-decoration:none;font-size:12px}}
  nav a.on{{background:var(--gold);border-color:var(--gold);color:#06090d;font-weight:700}}
  nav a:hover{{color:var(--ink)}}
  main{{max-width:1180px;margin:0 auto;padding:18px 24px 40px}}
  section{{margin:26px 0}}
  h2{{font-size:16px;letter-spacing:.04em;margin:0 0 10px;color:var(--ink)}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
  .card{{border:1px solid var(--line);border-radius:8px;background:var(--panel);padding:11px 13px}}
  .card u{{display:block;text-decoration:none;color:var(--dim);font-size:10px;
    letter-spacing:.14em;text-transform:uppercase}}
  .card b{{font-size:21px;font-weight:600}}
  table{{width:100%;border-collapse:collapse;font-size:12.5px}}
  th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}}
  th{{color:var(--dim);font-weight:600;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase}}
  .wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel)}}
  .note{{color:var(--dim);font-size:11.5px;margin-top:8px;max-width:80ch}}
  .verdict{{border:1px solid #5c2230;border-left-width:3px;border-radius:8px;
    background:#180d11;padding:12px 14px;color:#f3c9cf;font-size:12.5px}}
  footer{{max-width:1180px;margin:0 auto;padding:0 24px 40px;color:var(--dim);font-size:11px}}
  footer a{{color:var(--dim)}}
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


def _cards(pairs) -> str:
    cells = "".join(f"<div class='card'><u>{label}</u><b>{value}</b></div>"
                    for label, value in pairs)
    return f"<div class='cards'>{cells}</div>"


def _table(frame: pd.DataFrame) -> str:
    head = "".join(f"<th>{column}</th>" for column in frame.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{'' if pd.isna(value) else value}</td>" for value in row) + "</tr>"
        for row in frame.itertuples(index=False)
    )
    return f"<div class='wrap'><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>"


def build(destination: Path, dashboard_dir: Path, inputs: SiteInputs) -> list:
    """Write the site. Returns the paths written."""
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    # GitHub Pages runs Jekyll unless told not to, and Jekyll drops files it
    # does not recognise. Nothing here is a Jekyll site.
    (destination / ".nojekyll").write_text("", encoding="utf-8")

    outlook = inputs.outlook
    as_of = f"{outlook.as_of:%Y-%m-%d}"
    written = []

    # --- the board -------------------------------------------------------
    conditional, unconditional = outlook.rates[0]
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
        + "<p class='note'>Each lap starts at its own halving, drawn as a multiple of "
          "the price on that day. The shape they share is why people believe in the "
          "cycle; the distance between them is why four of them cannot prove it.</p></div>",
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
            + f"<p class='note'>{inputs.range_days}-day interval from the volatility "
              f"model, drawn only at the horizon it was scored at. Levels shown "
              f"({quoted}) kept their promise in a walk-forward coverage test{withheld}. "
              "It says how far, never which way.</p></div>"
        )
    body.append("</section>")

    rates = pd.DataFrame([
        {
            "horizon": f"{matched.horizon}d",
            "median %, days like today": round(100 * matched.median, 1),
            "positive %, days like today": round(100 * matched.share_positive),
            "independent windows": matched.effective_n,
            "median %, every day": round(100 * whole.median, 1),
            "positive %, every day": round(100 * whole.share_positive),
            "quotable": "yes" if matched.usable else "no - too few windows",
        }
        for matched, whole in outlook.rates
    ])
    body += [
        "<section><h2>What usually happened from days like today</h2>",
        _table(rates),
        "<p class='note'>Matched days are not observations: consecutive days share "
        "almost all of their forward window, so only non-overlapping ones are counted. "
        "The long-horizon rows are the ones that look most impressive and rest on the "
        "fewest.</p></section>",
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
        _table(rules),
        "<div class='verdict' style='margin-top:14px'>"
        + "<br>".join(verdict_lines)
        + "<br>Read all of it as description, not as a plan.</div></section>",
    ]

    written.append(_write(destination / "index.html",
                          _shell("BTC Cycle Lab", "Now",
                                 panel(dashboard_dir, "intro") + "".join(body), as_of=as_of)))

    # --- the desk --------------------------------------------------------
    desk_body = (
        "<section><h2>What each rule says right now</h2>"
        + "<div style='height:640px'>"
        + panel(dashboard_dir, "desk", desk_payload(inputs.signals))
        + "</div>"
        "<p class='note'>The live quote comes from Binance and is used only for the "
        "distance to a trigger; every statistic on the panel is from the local "
        "sample. Turn sound on to hear entries and exits during a replay.</p></section>"
    )
    written.append(_write(destination / "signals.html",
                          _shell("BTC Cycle Lab - signals", "Signals", desk_body, as_of=as_of)))

    # --- the receipts ----------------------------------------------------
    ledger = inputs.ledger if inputs.ledger is not None else pd.DataFrame()
    if ledger.empty:
        receipts = ("<section><h2>Receipts</h2><p class='note'>No claim has been "
                    "recorded yet.</p></section>")
    else:
        scored = score_ledger(ledger, inputs.signals["price"])
        open_claims = int((~scored["matured"].astype(bool)).sum())
        board = ledger_scoreboard(scored, base_rate=unconditional.share_positive)
        shown = scored.assign(
            as_of=scored["as_of"].dt.strftime("%Y-%m-%d"),
            settles=scored["target_date"].dt.strftime("%Y-%m-%d"),
        ).loc[:, ["as_of", "settles", "horizon", "claim", "level", "low", "high",
                  "p_up", "matured", "realised"]].round(4)
        receipts = (
            "<section><h2>Receipts</h2>"
            "<p class='note'>Every other page is a backtest, written by someone who had "
            "already seen the outcome. This one is the exception: claims recorded before "
            "their window closed, scored when it closes.</p>"
            f"<div class='verdict' style='margin:14px 0'>{ledger_verdict(board, open_claims)}</div>"
            + _table(shown)
            + ("" if board.empty else "<h2 style='margin-top:22px'>Settled</h2>"
               + _table(board.round(3)))
            + "</section>"
        )
    written.append(_write(destination / "receipts.html",
                          _shell("BTC Cycle Lab - receipts", "Receipts", receipts, as_of=as_of)))
    return written


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path
