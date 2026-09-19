"""The front door: three questions a normal visitor arrives with, in two languages.

Everything else on the site was written for someone who already knows what a
p-value is. This page is written for someone who does not, and who will give
it thirty seconds. It answers the three questions people actually bring - can
anyone time Bitcoin, what is the robot doing, and what is this for - each in
one sentence, with one picture, and sends the curious onward.

Two languages, one build. Both versions are written into the same document
and the reader's choice only decides which one is shown, so a number is
formatted once and cannot say one thing in Polish and another in English. The
Polish lines carry the `non-english-ok` marker, which is how this project
tells its language guard that a line is a translation on purpose rather than
a leak.

The sentences are chosen by the data, not typed in. If a hypothesis ever
survives, or the robot ever pulls ahead of holding, the page says so - and
says, in the second case, that the tests call a lead luck rather than skill,
because a front page that congratulates a rule the evidence rejects is the
exact failure the rest of the site exists to prevent.
"""
from __future__ import annotations

from html import escape

import pandas as pd

# How a confidence level is said to someone who has never met one. A 68%
# interval is "2 times out of 3", which a reader can picture; "68%" is a
# number they have to take on trust.
IN_WORDS = {
    0.50: ("half of the time", "w połowie przypadków"),  # non-english-ok
    0.68: ("2 times out of 3", "2 razy na 3"),
    0.90: ("9 times out of 10", "9 razy na 10"),
    0.95: ("19 times out of 20", "19 razy na 20"),
}

COPY = {
    "en": {
        "title": "BTC Cycle Lab",
        "switch": "Polski",
        "lede": "An experiment that checks, in public, whether anyone can "
                "predict Bitcoin. Updated every day.",
        "q1": "Can anyone predict when to buy Bitcoin?",
        "a1_none": "No. We checked {n} popular ideas - for example, that the "
                   "price rises after the \"halving\". None of them worked "
                   "better than guessing.",
        "a1_some": "Mostly no. We checked {n} popular ideas; {s} passed our "
                   "tests. The details show how weak that is.",
        "q2": "What does the robot do?",
        "a2": "It plays with pretend money, following one simple rule. It "
              "started with {capital} on {start}. It has {equity} now. If it "
              "had just bought and held, it would have {hold}.",
        "a2_behind": "So far, doing nothing would have been better.",
        "a2_ahead": "So far it is ahead - but our tests say that is luck, "
                    "not skill.",
        "q3": "So what is this page for?",
        "a3": "It shows how far the price might move in a month - not which "
              "way, because nobody knows that.",
        "a3_range": "{often}, a month from now the price is between {low} "
                    "and {high}.",
        "robot": "Robot", "hold": "Just holding",
        "details": "Want the details?",
        "details_link": "See how it was tested",
        "advice": "This is not advice. It is an experiment with pretend money.",
        "licence": "Price comparisons with shares and gold are computed but not "
                   "published: that data comes from Yahoo Finance, whose terms "
                   "forbid redistribution.",
        "updated": "Data up to {as_of}.",
    },
    "pl": {
        "title": "BTC Cycle Lab",
        "switch": "English",
        "lede": "Eksperyment, który publicznie sprawdza, czy ktokolwiek "  # non-english-ok
                "potrafi przewidzieć Bitcoina. Aktualizowany codziennie.",  # non-english-ok
        "q1": "Czy ktoś potrafi przewidzieć, kiedy kupić Bitcoina?",  # non-english-ok
        "a1_none": "Nie. Sprawdziliśmy {n} popularnych teorii - na przykład, "  # non-english-ok
                   "że cena rośnie po „halvingu”. Żadna nie "  # non-english-ok
                   "zadziałała lepiej niż zgadywanie.",  # non-english-ok
        "a1_some": "Raczej nie. Sprawdziliśmy {n} popularnych teorii; {s} "  # non-english-ok
                   "przeszło nasze testy. Szczegóły pokazują, jak słabo.",  # non-english-ok
        "q2": "Co robi robot?",
        "a2": "Gra udawanymi pieniędzmi według jednej prostej reguły. "  # non-english-ok
              "Zaczął z {capital} dnia {start}. Ma teraz {equity}. Gdyby po "  # non-english-ok
              "prostu kupił i trzymał, miałby {hold}.",  # non-english-ok
        "a2_behind": "Na razie lepiej wyszłoby nic nie robić.",  # non-english-ok
        "a2_ahead": "Na razie jest do przodu - ale nasze testy mówią, że "  # non-english-ok
                    "to szczęście, nie umiejętność.",  # non-english-ok
        "q3": "To po co ta strona?",  # non-english-ok
        "a3": "Pokazuje, jak mocno cena może się ruszyć w ciągu miesiąca - "  # non-english-ok
              "nie w którą stronę, bo tego nikt nie wie.",  # non-english-ok
        "a3_range": "{often} cena za miesiąc jest między {low} a {high}.",  # non-english-ok
        "robot": "Robot", "hold": "Samo trzymanie",  # non-english-ok
        "details": "Chcesz szczegółów?",  # non-english-ok
        "details_link": "Zobacz, jak to sprawdzaliśmy",  # non-english-ok
        "advice": "To nie jest porada inwestycyjna. To eksperyment na "  # non-english-ok
                  "udawanych pieniądzach.",  # non-english-ok
        "licence": "Porównania z akcjami i złotem są liczone, ale nie "  # non-english-ok
                   "publikowane: te dane pochodzą z Yahoo Finance, którego "  # non-english-ok
                   "regulamin zabrania ich rozpowszechniania.",  # non-english-ok
        "updated": "Dane do {as_of}.",
    },
}


def _dollars(value: float) -> str:
    return f"${value:,.0f}"


def _both(key: str, **values) -> str:
    """One sentence in both languages, the numbers formatted once."""
    return "".join(
        f'<span lang="{lang}">{escape(COPY[lang][key].format(**values))}</span>'
        for lang in ("pl", "en")
    )


def _both_raw(pl: str, en: str) -> str:
    return f'<span lang="pl">{pl}</span><span lang="en">{en}</span>'


def _often(level: float) -> tuple[str, str]:
    """A confidence level as a frequency a reader can picture."""
    for known, words in IN_WORDS.items():
        if abs(level - known) < 0.005:
            return words
    percent = f"{level:.0%}"
    return (f"{percent} of the time", f"w {percent} przypadków")  # non-english-ok


def race_svg(curve: pd.DataFrame) -> str:
    """The robot against simply holding, as two lines and nothing else.

    Inline SVG rather than a charting library: this page has one picture, and
    a visitor who came for thirty seconds should not wait for three megabytes
    of Plotly to draw it. No axes, no grid - the start and the end are the
    only two numbers that matter, and they are printed at the ends.
    """
    if curve is None or curve.empty:
        return ""
    step = max(1, len(curve) // 160)
    thin = curve.iloc[::step]
    if thin.index[-1] != curve.index[-1]:
        thin = pd.concat([thin, curve.iloc[[-1]]])

    width, height, pad_x, pad_y = 640.0, 230.0, 8.0, 18.0
    low = float(min(thin["equity"].min(), thin["buy_and_hold"].min()))
    high = float(max(thin["equity"].max(), thin["buy_and_hold"].max()))
    span = (high - low) or 1.0
    plot_w = width - 2 * pad_x - 150  # room for the end labels

    def points(series: pd.Series) -> str:
        coords = []
        for i, value in enumerate(series.to_numpy()):
            x = pad_x + plot_w * i / max(1, len(series) - 1)
            y = pad_y + (height - 2 * pad_y) * (1 - (float(value) - low) / span)
            coords.append(f"{x:.1f},{y:.1f}")
        return " ".join(coords)

    def end_y(series: pd.Series) -> float:
        return pad_y + (height - 2 * pad_y) * (1 - (float(series.iloc[-1]) - low) / span)

    robot_y, hold_y = end_y(thin["equity"]), end_y(thin["buy_and_hold"])
    # Keep the two end labels from printing on top of each other.
    if abs(robot_y - hold_y) < 30:
        nudge = (30 - abs(robot_y - hold_y)) / 2
        if robot_y <= hold_y:
            robot_y, hold_y = robot_y - nudge, hold_y + nudge
        else:
            robot_y, hold_y = robot_y + nudge, hold_y - nudge

    label_x = pad_x + plot_w + 12
    robot_value = _dollars(float(curve["equity"].iloc[-1]))
    hold_value = _dollars(float(curve["buy_and_hold"].iloc[-1]))

    def label(y: float, key: str, value: str, colour: str) -> str:
        return "".join(
            f'<text lang="{lang}" x="{label_x:.1f}" y="{y:.1f}" fill="{colour}" '
            f'font-size="15" dominant-baseline="middle">'
            f'<tspan font-weight="700">{escape(value)}</tspan>'
            f'<tspan x="{label_x:.1f}" dy="18" fill="#8f99a8" font-size="13">'
            f'{escape(COPY[lang][key])}</tspan></text>'
            for lang in ("pl", "en")
        )

    return (
        f'<svg class="race" viewBox="0 0 {width:.0f} {height + 20:.0f}" '
        f'role="img" aria-label="robot against simply holding">'
        f'<polyline points="{points(thin["buy_and_hold"])}" fill="none" '
        f'stroke="#8f99a8" stroke-width="2" stroke-dasharray="5 5"/>'
        f'<polyline points="{points(thin["equity"])}" fill="none" '
        f'stroke="#f7931a" stroke-width="3"/>'
        + label(robot_y, "robot", robot_value, "#f7931a")
        + label(hold_y, "hold", hold_value, "#c3cad6")
        + "</svg>"
    )


def simple_body(inputs, *, as_of: str) -> str:
    """The three answers. Every sentence here is picked by the data."""
    parts = [
        '<header class="simple-head">',
        f'<h1>{escape(COPY["en"]["title"])}</h1>',
        '<button id="btclab-lang" type="button">'
        + _both("switch") + "</button>",
        f'<p class="lede">{_both("lede")}</p>',
        "</header>",
    ]

    # 1 - can anyone time it
    scan = inputs.scan if inputs.scan is not None else pd.DataFrame()
    tested = int(len(scan))
    survived = int(scan["significant_adjusted"].sum()) if "significant_adjusted" in scan else 0
    if tested:
        answer = _both("a1_none" if survived == 0 else "a1_some", n=tested, s=survived)
        parts.append(
            f'<section class="qa"><h2>{_both("q1")}</h2><p>{answer}</p></section>'
        )

    # 2 - what the robot does
    paper = inputs.paper or {}
    state = (paper.get("payload") or {}).get("state") or {}
    if state.get("started"):
        equity, hold = float(state["equity"]), float(state["hold"])
        verdict = "a2_behind" if equity < hold else "a2_ahead"
        parts.append(
            f'<section class="qa"><h2>{_both("q2")}</h2>'
            f'<p>{_both("a2", capital=_dollars(state["capital"]), start=state["started"], equity=_dollars(equity), hold=_dollars(hold))}</p>'
            f'<p class="verdict-line">{_both(verdict)}</p>'
            + race_svg(paper.get("curve"))
            + "</section>"
        )

    # 3 - what this is for
    ranges = (inputs.saved or {}).get("range_30")
    if ranges is None or getattr(ranges, "empty", True):
        ranges = inputs.range_forecast
    range_line = ""
    if ranges is not None and not ranges.empty:
        levels = ranges.sort_values("level")
        near = levels[levels["level"] <= 0.70]
        row = (near if not near.empty else levels).iloc[-1]
        often_en, often_pl = _often(float(row["level"]))
        low, high = _dollars(float(row["low"])), _dollars(float(row["high"]))
        range_line = (
            '<p class="range-line">'
            + _both_raw(
                escape(COPY["pl"]["a3_range"].format(often=often_pl.capitalize(), low=low, high=high)),
                escape(COPY["en"]["a3_range"].format(often=often_en.capitalize(), low=low, high=high)),
            )
            + "</p>"
        )
    parts.append(
        f'<section class="qa"><h2>{_both("q3")}</h2><p>{_both("a3")}</p>'
        + range_line + "</section>"
    )

    parts.append(
        '<section class="more">'
        f'<p>{_both("details")}</p>'
        f'<a class="button" href="today.html">{_both("details_link")} &rarr;</a>'
        "</section>"
    )
    parts.append(
        '<footer class="simple-foot">'
        f'<p><b>{_both("advice")}</b></p>'
        f'<p>{_both("updated", as_of=as_of)}</p>'
        f'<p>{_both("licence")}</p>'
        "</footer>"
    )
    return "".join(parts)


# The language is chosen before the page paints, so a Polish reader never sees
# a flash of English. A saved choice wins; otherwise the browser's own language
# decides, and anything that is not Polish falls back to English.
LANGUAGE_BOOT = """<script>
  (function () {
    var lang = null;
    try { lang = window.localStorage.getItem("btclab.lang"); } catch (err) {}
    if (lang !== "pl" && lang !== "en") {
      var said = (navigator.language || navigator.userLanguage || "").toLowerCase();
      lang = said.indexOf("pl") === 0 ? "pl" : "en";
    }
    document.documentElement.setAttribute("data-lang", lang);
    document.documentElement.setAttribute("lang", lang);
  })();
</script>"""

LANGUAGE_SWITCH = """<script>
  (function () {
    var button = document.getElementById("btclab-lang");
    if (!button) { return; }
    button.addEventListener("click", function () {
      var root = document.documentElement;
      var next = root.getAttribute("data-lang") === "pl" ? "en" : "pl";
      root.setAttribute("data-lang", next);
      root.setAttribute("lang", next);
      try { window.localStorage.setItem("btclab.lang", next); } catch (err) {}
    });
  })();
</script>"""

SIMPLE_CSS = """
  :root{--bg:#000308;--panel:#101219;--line:#252935;--ink:#f4f6f9;--soft:#c3cad6;
    --dim:#8f99a8;--gold:#f7931a;
    --sans:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display",
      "Segoe UI Variable Text","Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:19px/1.6 var(--sans);
    letter-spacing:-.011em;-webkit-font-smoothing:antialiased;-webkit-text-size-adjust:100%}
  /* Only the chosen language is shown. Both are in the document. */
  [data-lang="pl"] [lang="en"],[data-lang="en"] [lang="pl"]{display:none !important}
  main{max-width:720px;margin:0 auto;padding:40px 22px 56px}
  .simple-head{position:relative}
  .simple-head h1{margin:0;font-size:40px;font-weight:700;letter-spacing:-.024em}
  .simple-head .lede{margin:12px 0 0;color:var(--dim);font-size:19px}
  #btclab-lang{position:absolute;top:6px;right:0;padding:7px 14px;border-radius:999px;
    border:1px solid var(--line);background:var(--panel);color:var(--soft);
    font:600 14px var(--sans);cursor:pointer}
  #btclab-lang:hover{color:var(--ink);border-color:var(--gold)}
  .qa{margin:44px 0 0;padding:26px 26px 22px;border-radius:18px;background:var(--panel)}
  .qa h2{margin:0 0 10px;font-size:25px;line-height:1.25;letter-spacing:-.02em}
  .qa p{margin:0}
  .verdict-line{margin-top:12px !important;color:var(--gold);font-weight:600}
  .range-line{margin-top:12px !important;font-weight:600}
  .race{display:block;width:100%;height:auto;margin-top:18px}
  .more{margin:48px 0 0;text-align:center}
  .more p{margin:0 0 14px;color:var(--dim)}
  .button{display:inline-block;padding:13px 26px;border-radius:999px;
    background:var(--gold);color:#000308;font-weight:700;text-decoration:none}
  .button:hover{filter:brightness(1.08)}
  .simple-foot{margin:52px 0 0;padding-top:20px;border-top:1px solid var(--line);
    color:var(--dim);font-size:14px}
  .simple-foot p{margin:0 0 8px}
  .simple-foot b{color:var(--soft)}
  @media (max-width:560px){
    body{font-size:17px}
    .simple-head h1{font-size:32px;padding-right:96px}
    .qa{padding:20px 18px 18px}
    .qa h2{font-size:21px}
  }
"""


def simple_document(inputs, *, as_of: str, intro: str, height_reporter: str) -> str:
    """The whole front page, ready to write to index.html."""
    return (
        '<!doctype html>\n<html lang="en" data-lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>BTC Cycle Lab</title>\n"
        + LANGUAGE_BOOT
        + "\n<style>" + SIMPLE_CSS + "</style>\n</head>\n<body>\n"
        + intro
        + "<main>" + simple_body(inputs, as_of=as_of) + "</main>\n"
        + LANGUAGE_SWITCH + "\n" + height_reporter
        + "\n</body>\n</html>\n"
    )
