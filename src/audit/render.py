"""Turning a report into something a person reads: terminal text and one HTML file.

The HTML is deliberately a single self-contained file with no CDN, no fonts to
fetch and no build step. It is the artefact the customer keeps, mails to a
partner or prints, and every external reference is one more way for it to look
broken in three years.

Colours come from the project's validated categorical palette (slots 1 and 2)
and its fixed status palette. Status is never carried by colour alone: every
verdict badge pairs its colour with a glyph and the word.
"""
from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .checks import FAIL, NA, PASS, WARN
from .report import REJECTED, SURVIVES, UNPROVEN

_GLYPH = {PASS: "+", FAIL: "x", WARN: "!", NA: "-"}
_WORD = {PASS: "PASSED", FAIL: "FAILED", WARN: "WEAK", NA: "NOT RUN"}

# Percent-formatted keys get a %, ratio-formatted ones do not. Anything not
# listed falls through to a plain number, which is the safe default: a wrong
# unit on a report like this is worse than an ugly one.
_PERCENT_KEYS = {
    "total_return", "cagr", "volatility", "max_drawdown", "win_rate",
    "time_in_market", "exposure", "share_of_resamples_negative",
    "first_half_return", "second_half_return", "strategy_return",
    "benchmark_return", "strategy_max_drawdown", "benchmark_max_drawdown",
    "total_return_without_top_days", "best_single_day", "worst_single_day",
    "share_of_days_removed", "net_at_25bps", "net_at_50bps", "net_at_100bps",
}
_BPS_KEYS = {"applied_bps", "breakeven_bps"}


def _fmt(key: str, value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (bool, str)):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "-"
        if key in _PERCENT_KEYS:
            return f"{number:+.2%}" if key not in {"win_rate", "time_in_market",
                                                   "exposure", "volatility",
                                                   "share_of_resamples_negative",
                                                   "share_of_days_removed"} else f"{number:.1%}"
        if key in _BPS_KEYS:
            return f"{number:,.0f} bps"
        if key.startswith("p_value") or key.endswith("_p_value"):
            return f"{number:.4f}"
        return f"{number:,.3f}"
    return str(value)


def _label(key: str) -> str:
    return key.replace("_", " ")


# --------------------------------------------------------------------------
# Terminal
# --------------------------------------------------------------------------
def to_text(report, *, width: int = 78) -> str:
    lines: list[str] = []
    meta = report.meta
    rule = "=" * width

    lines.append(rule)
    lines.append(f"STRATEGY REALITY CHECK - {meta.get('label', 'input')}")
    lines.append(rule)
    lines.append(
        f"{meta['observations']:,} observations, {meta['first_day']} to {meta['last_day']} "
        f"| read as {meta['input_form']} | {meta['periods_per_year']} periods/year"
    )
    if meta.get("licence_mode") != "licensed":
        lines.append("EVALUATION MODE - permutation counts are reduced; see the banner above.")
    lines.append("")

    counts = meta["counts"]
    lines.append(f"VERDICT: {report.verdict}")
    lines.extend(_wrap(report.summary, width))
    lines.append("")
    lines.append(
        f"  passed {counts['passed']} | failed {counts['failed']} | "
        f"weak {counts['weak']} | not run {counts['not_run']} "
        f"(of {counts['blocking_total']} checks that count)"
    )
    lines.append("")

    for check in report.checks:
        marker = _GLYPH.get(check.verdict, "?")
        head = f"[{marker}] {check.title.upper()}"
        if check.blocking:
            head += f"  -  {_WORD.get(check.verdict, check.verdict)}"
        lines.append(head)
        lines.append("    " + check.question)
        lines.extend(_wrap(check.headline, width - 4, indent="    "))
        if check.detail:
            lines.extend(_wrap(check.detail, width - 4, indent="    "))
        if check.numbers:
            for key, value in check.numbers.items():
                lines.append(f"      {_label(key):<34} {_fmt(key, value)}")
        lines.append("")

    if meta.get("notes"):
        lines.append("HOW THE FILE WAS READ")
        for note in meta["notes"]:
            lines.extend(_wrap("- " + note, width - 2, indent="  "))
        lines.append("")

    lines.append(rule)
    lines.append(
        "This is an audit of a track record, not advice. It says how much of a past "
        "result could be chance. It does not say what will happen next, and no "
        "result here is a recommendation to buy, sell or hold anything."
    )
    lines.append(rule)
    return "\n".join(lines)


def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width, initial_indent=indent,
                         subsequent_indent=indent) or [indent.rstrip()]


# --------------------------------------------------------------------------
# Equity chart: one line per series, log scale, direct-labelled
# --------------------------------------------------------------------------
def _equity_svg(data) -> str:
    """Compounded growth of 1 unit, log scale, with the benchmark if there is one.

    Log scale is not a stylistic choice here: on a linear axis a curve that
    compounds makes its last year look like the whole story and its first
    three look flat, which is the opposite of what this report is arguing.
    """
    equity = data.equity
    if len(equity) < 2:
        return ""

    series = [("Strategy", equity, "var(--series-1)")]
    if data.benchmark_returns is not None:
        bench = (1 + data.benchmark_returns.reindex(equity.index).fillna(0.0)).cumprod()
        series.append(("Benchmark", bench, "var(--series-2)"))
    elif data.asset_returns is not None:
        bench = (1 + data.asset_returns.reindex(equity.index).fillna(0.0)).cumprod()
        series.append(("Buy and hold", bench, "var(--series-2)"))

    width, height = 720.0, 300.0
    # The right-hand gutter holds the direct labels ("Buy and hold 3.2x"), so it
    # is sized from the longest one rather than guessed; a clipped label is
    # exactly the kind of small broken thing that makes a paid report look cheap.
    pad_left, pad_top, pad_bottom = 52.0, 18.0, 30.0
    labels = [f"{name} {float(values.iloc[-1]):,.1f}x" for name, values, _ in series]
    pad_right = max(96.0, 14.0 + 7.6 * max(len(text) for text in labels))
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    all_values = np.concatenate([np.asarray(s[1], dtype=float) for s in series])
    all_values = all_values[np.isfinite(all_values) & (all_values > 0)]
    if all_values.size == 0:
        return ""
    low, high = float(all_values.min()), float(all_values.max())
    if high <= low:
        high = low * 1.01
    log_low, log_high = math.log(low), math.log(high)

    index = equity.index
    span = (index[-1] - index[0]).days or 1

    def x_of(stamp) -> float:
        return pad_left + plot_w * ((stamp - index[0]).days / span)

    def y_of(value: float) -> float:
        if not np.isfinite(value) or value <= 0:
            return pad_top + plot_h
        return pad_top + plot_h * (1 - (math.log(value) - log_low) / (log_high - log_low))

    # Recessive gridlines at readable multiples of the starting capital.
    ticks: list[float] = []
    exponent = math.floor(math.log10(low))
    while 10.0 ** exponent <= high * 1.001:
        for step in (1.0, 2.0, 5.0):
            value = step * 10.0 ** exponent
            if low * 0.999 <= value <= high * 1.001:
                ticks.append(value)
        exponent += 1
    if len(ticks) < 2:
        ticks = [low, high]

    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" class="equity" '
        f'aria-label="Growth of one unit, logarithmic scale" preserveAspectRatio="xMidYMid meet">'
    ]
    for value in ticks:
        y = y_of(value)
        parts.append(
            f'<line x1="{pad_left:.1f}" y1="{y:.1f}" x2="{pad_left + plot_w:.1f}" '
            f'y2="{y:.1f}" class="grid"/>'
        )
        text = f"{value:,.0f}x" if value >= 1 else f"{value:,.2f}x"
        parts.append(
            f'<text x="{pad_left - 8:.1f}" y="{y + 4:.1f}" class="tick" '
            f'text-anchor="end">{html.escape(text)}</text>'
        )

    for _name, values, colour in series:
        points = " ".join(
            f"{x_of(stamp):.1f},{y_of(float(value)):.1f}"
            for stamp, value in values.items()
            if np.isfinite(value)
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colour}" '
                     f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')

    # Label positions, nudged apart so two lines that finish close together do
    # not print one label on top of the other.
    placed: list[float] = []
    for (_name, values, colour), text in zip(series, labels):
        y = y_of(float(values.iloc[-1])) + 4.0
        while any(abs(y - taken) < 15.0 for taken in placed):
            y += 15.0
        y = min(max(y, pad_top + 10.0), pad_top + plot_h)
        placed.append(y)
        parts.append(
            f'<text x="{pad_left + plot_w + 8:.1f}" y="{y:.1f}" '
            f'class="direct" fill="{colour}">{html.escape(text)}</text>'
        )

    for stamp, anchor in ((index[0], "start"), (index[-1], "end")):
        parts.append(
            f'<text x="{x_of(stamp):.1f}" y="{height - 8:.1f}" class="tick" '
            f'text-anchor="{anchor}">{stamp.date().isoformat()}</text>'
        )
    parts.append("</svg>")

    # A legend as well as the direct labels: identity must never rest on colour
    # alone, and the right-hand labels disappear when the chart is cropped or
    # read by someone who cannot separate the two hues.
    legend = "".join(
        f'<span class="key"><i style="background:{colour}"></i>{html.escape(name)}</span>'
        for name, _, colour in series
    )
    return f'<div class="legend">{legend}</div>' + "".join(parts)


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
_CSS = """
:root{color-scheme:light dark;
--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--ink-3:#77766f;
--line:#e4e3de;--series-1:#2a78d6;--series-2:#eb6834;
--good:#0ca30c;--warn:#fab219;--serious:#ec835a;--critical:#d03b3b;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink-2:#c3c2b7;--ink-3:#8f8e85;
--line:#313130;--series-1:#3987e5;--series-2:#d95926;}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:860px;margin:0 auto;padding:32px 20px 64px}
header{border-bottom:2px solid var(--ink);padding-bottom:16px;margin-bottom:24px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--ink-2);font-size:13px;margin:0}
.verdict{background:var(--surface);border:1px solid var(--line);border-radius:10px;
padding:20px 22px;margin:0 0 28px}
.verdict h2{margin:0 0 8px;font-size:19px;letter-spacing:.02em}
.verdict p{margin:0;color:var(--ink-2)}
.tally{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;font-size:13px}
.tally span{border:1px solid var(--line);border-radius:999px;padding:3px 11px;color:var(--ink-2)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;
padding:18px 20px;margin:0 0 14px}
.card h3{margin:0;font-size:16px;display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.q{color:var(--ink-3);font-size:13px;font-style:italic;margin:6px 0 10px}
.head{margin:0 0 8px}
.detail{color:var(--ink-2);font-size:13.5px;margin:0}
.badge{font-size:11px;font-weight:700;letter-spacing:.08em;border-radius:5px;
padding:3px 8px;border:1.5px solid currentColor;white-space:nowrap}
.b-pass{color:var(--good)}.b-fail{color:var(--critical)}
.b-warn{color:var(--serious)}.b-na{color:var(--ink-3)}
table{border-collapse:collapse;width:100%;margin-top:12px;font-size:13px;
font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}
td:last-child{text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
th{color:var(--ink-3);font-weight:600;font-size:11px;letter-spacing:.06em;text-transform:uppercase}
.chart{background:var(--surface);border:1px solid var(--line);border-radius:10px;
padding:18px 20px;margin:0 0 24px;overflow-x:auto}
.chart h3{margin:0 0 2px;font-size:16px}
.equity{width:100%;height:auto;min-width:560px;display:block;margin-top:8px}
.grid{stroke:var(--line);stroke-width:1}
.tick{fill:var(--ink-3);font-size:11px;font-family:ui-monospace,Menlo,monospace}
.direct{font-size:12px;font-weight:600}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;font-size:12.5px;color:var(--ink-2)}
.key{display:inline-flex;align-items:center;gap:6px}
.key i{width:14px;height:3px;border-radius:2px;display:inline-block}
.notes{font-size:13px;color:var(--ink-2)}
.notes li{margin-bottom:6px}
footer{margin-top:32px;padding-top:16px;border-top:1px solid var(--line);
font-size:12.5px;color:var(--ink-3)}
.stamp{background:var(--surface);border:1px dashed var(--serious);color:var(--ink-2);
border-radius:8px;padding:12px 16px;margin:0 0 20px;font-size:13px}
@media print{body{background:#fff}.card,.verdict,.chart{break-inside:avoid}}
"""

_BADGE_CLASS = {PASS: "b-pass", FAIL: "b-fail", WARN: "b-warn", NA: "b-na"}


def _provenance(meta: dict, esc) -> str:
    """Who this report was prepared for, when it is a licensed one.

    Traceability cuts both ways and both are wanted: a report an investment
    committee reads should say whose copy produced it, and a licensee whose
    name is on every page they hand out is a licensee who thinks before handing
    out the licence. An evaluation report names nobody - there is nobody to
    name, and the watermark at the top has already said so.
    """
    licensee = str(meta.get("licensee", "")).strip()
    if meta.get("licence_mode") != "licensed" or not licensee:
        return ""
    return f"<p>Prepared with a licensed copy registered to {esc(licensee)}.</p>"


def to_html(report, data) -> str:
    meta = report.meta
    esc = html.escape

    verdict_colour = {SURVIVES: "var(--good)", UNPROVEN: "var(--serious)",
                      REJECTED: "var(--critical)"}.get(report.verdict, "var(--ink)")

    body = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>Strategy Reality Check - {esc(str(meta.get('label', '')))}</title>",
        f"<style>{_CSS}</style></head><body><div class=\"wrap\">",
        "<header><h1>Strategy Reality Check</h1>",
        f"<p class=\"sub\">{esc(str(meta.get('label','')))} &middot; "
        f"{meta['observations']:,} observations &middot; {esc(meta['first_day'])} to "
        f"{esc(meta['last_day'])} &middot; generated {esc(meta['generated_utc'])}</p></header>",
    ]

    if meta.get("licence_mode") != "licensed":
        body.append(
            '<p class="stamp"><strong>EVALUATION COPY.</strong> Permutation counts are '
            "reduced, so the p-values below are coarser than a licensed run would "
            "produce. Every verdict is computed the same way; nothing is withheld or "
            "altered.</p>"
        )

    counts = meta["counts"]
    body.append(
        f'<section class="verdict"><h2 style="color:{verdict_colour}">'
        f"{esc(report.verdict)}</h2><p>{esc(report.summary)}</p>"
        f'<div class="tally"><span>{counts["passed"]} passed</span>'
        f'<span>{counts["failed"]} failed</span><span>{counts["weak"]} weak</span>'
        f'<span>{counts["not_run"]} not run</span></div></section>'
    )

    chart = _equity_svg(data)
    if chart:
        body.append(
            '<section class="chart"><h3>Growth of 1 unit</h3>'
            '<p class="q">Logarithmic scale, so equal vertical distances are equal '
            "percentage moves. Each line is labelled at its right-hand end.</p>"
            + chart + "</section>"
        )

    for check in report.checks:
        badge = ""
        if check.blocking:
            badge = (
                f'<span class="badge {_BADGE_CLASS.get(check.verdict, "b-na")}">'
                f'{esc(_GLYPH.get(check.verdict, "?"))} '
                f'{esc(_WORD.get(check.verdict, check.verdict))}</span>'
            )
        rows = "".join(
            f"<tr><td>{esc(_label(key))}</td><td>{esc(_fmt(key, value))}</td></tr>"
            for key, value in check.numbers.items()
        )
        table = (
            f"<table><thead><tr><th>measure</th><th>value</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>" if rows else ""
        )
        detail = f'<p class="detail">{esc(check.detail)}</p>' if check.detail else ""
        body.append(
            f'<article class="card"><h3>{esc(check.title)}{badge}</h3>'
            f'<p class="q">{esc(check.question)}</p>'
            f'<p class="head">{esc(check.headline)}</p>{detail}{table}</article>'
        )

    if meta.get("notes"):
        notes = "".join(f"<li>{esc(note)}</li>" for note in meta["notes"])
        body.append(
            f'<section class="card"><h3>How the file was read</h3>'
            f'<ul class="notes">{notes}</ul></section>'
        )

    body.append(
        "<footer><p><strong>This is an audit, not advice.</strong> It measures how "
        "much of a past track record could be chance, cost or a handful of days. It "
        "makes no forecast, and nothing here is a recommendation to buy, sell or hold "
        "any instrument. Past performance does not predict future results.</p>"
        f"<p>Read as <em>{esc(meta['input_form'])}</em>, position convention "
        f"<em>{esc(meta['position_convention'])}</em>, "
        f"{meta['periods_per_year']} periods per year, "
        f"{meta['cost_bps_applied']:.0f} bps cost applied, "
        f"{meta['permutations']:,} permutations. "
        f"Variants declared: {meta.get('variants_declared') or 'not declared'}.</p>"
        + _provenance(meta, esc)
        + "</footer></div></body></html>"
    )
    return "".join(body)


def write_html(report, data, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_html(report, data), encoding="utf-8")
    return path
