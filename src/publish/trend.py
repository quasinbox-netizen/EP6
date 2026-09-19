"""Two moving pictures: the trend as a line that changes colour, and the cycle.

Both are hand-drawn SVG rather than Plotly, for one reason: the animation has
to be the drawing itself. The line is laid down left to right, green where the
trend was up and red where it was down, so the reader watches the colour flip
at the dates it actually flipped - which is the whole point of the picture.
Plotly can animate frames; it cannot draw a line the way a pen does.

What the colours are is stated on the chart and kept to what is measurable:
green means the price (or the 50-day average, for the agent) sat above the
200-day average that day. It describes the day. The lab's tests show it does
not predict the next one, and every caption here says so.

The live part is small and deliberately limited. The page fetches the current
quote and moves one dot. The history is never recoloured by a live tick: a
colour means "the day closed like this", and a day that has not closed has not
earned one. If the live price has crossed the average, the page says so in
words and leaves the line alone.
"""
from __future__ import annotations

from html import escape

import pandas as pd

from analysis.cycle_timing import FOLK_DOWN_DAYS, FOLK_UP_DAYS, CycleTiming

W, H = 1000, 320
PAD_L, PAD_R, PAD_T, PAD_B = 12, 78, 18, 30
DRAW_SECONDS = 2.8


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _short(value: float) -> str:
    if value >= 1000:
        return f"${value / 1000:,.0f}k"
    return f"${value:,.0f}"


def _path(xs, ys) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def _runs(state: pd.Series) -> list:
    """Stretches of consecutive days in the same state: (start, end, value)."""
    runs, start = [], 0
    values = state.to_numpy()
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            runs.append((start, i - 1, bool(values[start])))
            start = i
    return runs


def trend_ribbon(close: pd.Series, *, element_id: str, days: int = 730,
                 fast: int | None = None, slow: int = 200,
                 trades: list | None = None, since: str | None = None) -> str:
    """The trend as one line, coloured day by day, drawn in as it is watched.

    With `fast` unset the colour is the price against its `slow`-day average,
    which is the trend people quote. With `fast` set it is the fast average
    against the slow one - the rule the agent trades - so the colour on its
    page is the colour of its own position.
    """
    close = close.dropna().astype(float)
    slow_avg = close.rolling(slow).mean()
    fast_avg = close.rolling(fast).mean() if fast else None
    frame = pd.DataFrame({"close": close, "slow": slow_avg})
    if fast_avg is not None:
        frame["fast"] = fast_avg
    frame = frame.dropna()
    if since is not None:
        frame = frame[frame.index >= pd.Timestamp(since)]
    else:
        frame = frame.iloc[-days:]
    if len(frame) < 10:
        return ""

    up = (frame["fast"] > frame["slow"]) if fast else (frame["close"] > frame["slow"])
    runs = _runs(up)
    flips = len(runs) - 1

    low = float(frame[[c for c in frame.columns]].min().min())
    high = float(frame[[c for c in frame.columns]].max().max())
    margin = (high - low) * 0.07
    low, high = low - margin, high + margin
    inner_w, inner_h = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def x_of(i):
        return PAD_L + inner_w * i / (len(frame) - 1)

    def y_of(v):
        return PAD_T + inner_h * (1 - (v - low) / (high - low))

    xs = [x_of(i) for i in range(len(frame))]
    ys = [y_of(v) for v in frame["close"]]

    parts = [f'<svg class="tr-svg" viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Price over time, green when the trend was up and red when it was down">']

    # Gridlines with prices at the right edge: four levels, rounded to what a
    # person would say out loud.
    for k in range(4):
        value = low + (high - low) * (0.12 + 0.76 * k / 3)
        y = y_of(value)
        parts.append(f'<line class="tr-grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text class="tr-axis" x="{W - PAD_R + 8}" y="{y + 4:.1f}">{_short(value)}</text>')
    # Year and half-year marks along the bottom.
    for day in pd.date_range(frame.index[0], frame.index[-1], freq="MS"):
        if day.month not in (1, 7):
            continue
        i = frame.index.searchsorted(day)
        if i >= len(frame):
            continue
        label = f"{day:%Y}" if day.month == 1 else f"{day:%b}"
        parts.append(f'<text class="tr-axis" x="{xs[i]:.1f}" y="{H - 8}" text-anchor="middle">{label}</text>')

    # The averages first, underneath, fading in: the line is what moves.
    parts.append(f'<path class="tr-avg" d="{_path(xs, [y_of(v) for v in frame["slow"]])}"/>')
    if fast:
        parts.append(f'<path class="tr-avg tr-fast" d="{_path(xs, [y_of(v) for v in frame["fast"]])}"/>')

    # One path per stretch, each starting where the last ended, each given its
    # share of the drawing time so the pen moves at one speed across the page.
    for start, end, is_up in runs:
        stop = min(end + 1, len(frame) - 1)
        segment_x, segment_y = xs[start:stop + 1], ys[start:stop + 1]
        if len(segment_x) < 2:
            segment_x, segment_y = xs[max(start - 1, 0):stop + 1], ys[max(start - 1, 0):stop + 1]
        delay = DRAW_SECONDS * (segment_x[0] - PAD_L) / inner_w
        duration = max(DRAW_SECONDS * (segment_x[-1] - segment_x[0]) / inner_w, 0.05)
        colour = "up" if is_up else "down"
        area = (_path(segment_x, segment_y)
                + f" L{segment_x[-1]:.1f},{H - PAD_B} L{segment_x[0]:.1f},{H - PAD_B} Z")
        parts.append(f'<path class="tr-area {colour}" d="{area}" style="--d:{delay:.2f}s"/>')
        parts.append(f'<path class="tr-line {colour}" pathLength="1" d="{_path(segment_x, segment_y)}" '
                     f'style="--d:{delay:.2f}s;--t:{duration:.2f}s"/>')

    # The comet: a spark that runs the whole line once it is drawn, again and
    # again, so the chart keeps moving without the data pretending to.
    comet_path = f"{element_id}-track"
    parts.append(f'<path id="{comet_path}" d="{_path(xs, ys)}" fill="none" stroke="none"/>')
    parts.append(f'<circle class="tr-comet" r="3.2"><animateMotion dur="5.5s" repeatCount="indefinite" '
                 f'begin="indefinite"><mpath href="#{comet_path}"/></animateMotion></circle>')

    for trade in trades or []:
        day = pd.Timestamp(trade["date"])
        if day < frame.index[0] or day > frame.index[-1]:
            continue
        i = min(frame.index.searchsorted(day), len(frame) - 1)
        x, y = xs[i], ys[i]
        delay = DRAW_SECONDS * (x - PAD_L) / inner_w
        if trade["action"] == "buy":
            shape = f"{x:.1f},{y + 7:.1f} {x - 7:.1f},{y + 19:.1f} {x + 7:.1f},{y + 19:.1f}"
            colour = "up"
        else:
            shape = f"{x:.1f},{y - 7:.1f} {x - 7:.1f},{y - 19:.1f} {x + 7:.1f},{y - 19:.1f}"
            colour = "down"
        parts.append(f'<polygon class="tr-trade {colour}" points="{shape}" style="--d:{delay:.2f}s">'
                     f'<title>{escape(trade["action"])} at {_money(float(trade["price"]))} on {day:%Y-%m-%d}</title></polygon>')

    now_up = bool(up.iloc[-1])
    colour = "up" if now_up else "down"
    parts.append(f'<circle class="tr-ring {colour}" cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="6"/>'
                 f'<circle class="tr-dot {colour}" cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="5"/>')
    # The live dot starts hidden on the last close and is moved by the script.
    parts.append(f'<line class="tr-live-link" x1="{xs[-1]:.1f}" y1="{ys[-1]:.1f}" x2="{xs[-1]:.1f}" y2="{ys[-1]:.1f}"/>'
                 f'<circle class="tr-live" cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="4.5"/>')
    parts.append("</svg>")

    start_of_run = frame.index[runs[-1][0]]
    run_days = int((frame.index[-1] - start_of_run).days) + 1
    last = frame.iloc[-1]
    word = "UP" if now_up else "DOWN"
    if fast:
        what = f"the {fast}-day average is {'above' if now_up else 'below'} the {slow}-day"
    else:
        gap = last["close"] / last["slow"] - 1
        what = f"price is {abs(gap):.0%} {'above' if now_up else 'below'} its {slow}-day average ({_money(last['slow'])})"
    period = f"since {frame.index[0]:%b %Y}" if since else f"in the last {round(len(frame) / 365)} years"

    head = (
        f'<div class="tr-head">'
        f'<span class="tr-badge {colour}">Trend {word}</span>'
        f'<span class="tr-since">for {run_days} days, since {start_of_run:%d %b %Y}</span>'
        f'<span class="tr-sp"></span>'
        f'<span class="tr-quote"><i class="tr-pip"></i><span class="tr-price">last close {_money(last["close"])}</span></span>'
        f"</div>"
    )
    foot = (
        f'<p class="tr-note">Today {escape(what)}. '
        f'<span class="tr-live-note"></span> '
        f"The colour flipped <b>{flips} times</b> {period} - every flip is a moment "
        f"someone called the new trend, and most of them did not last.</p>"
    )
    data = (f'data-low="{low:.2f}" data-high="{high:.2f}" data-top="{PAD_T}" data-inner="{inner_h}" '
            f'data-x="{xs[-1]:.1f}" data-y="{ys[-1]:.1f}" data-slow="{float(last["slow"]):.2f}" '
            f'data-up="{1 if now_up else 0}" data-mode="{"cross" if fast else "price"}"')
    return f'<div class="tr-box tr-anim" id="{element_id}" {data}>{head}{"".join(parts)}{foot}</div>'


def cycle_chart(close: pd.Series, timing: CycleTiming, *, element_id: str = "cycle-lap") -> str:
    """Every lap on one line: red from each top to its bottom, green back up.

    Log scale, because the price went from a thousand to a hundred thousand
    and on a straight axis the first two laps would be a flat line. Under the
    line runs a lane of bars, one per leg, labelled with its length in days -
    the numbers the folk calendar is built from, drawn where they happened.
    """
    import math

    close = close.dropna().astype(float)
    turning = [close.index[0]]
    for lap in timing.laps:
        turning.append(lap.top)
        if lap.bottom is not None:
            turning.append(lap.bottom)
    turning.append(close.index[-1])
    weekly = close.resample("W").last().dropna()
    series = pd.concat([weekly, close.loc[turning]]).sort_index()
    series = series[~series.index.duplicated(keep="last")]

    end = max(timing.as_of, timing.folk_bottom_date) + pd.Timedelta(days=120)
    start = series.index[0]
    span = (end - start).days
    lane_h = 46
    height = H + lane_h
    inner_w, inner_h = W - PAD_L - PAD_R, H - PAD_T - PAD_B
    lo, hi = math.log10(series.min() * 0.8), math.log10(series.max() * 1.25)

    def x_of(day):
        return PAD_L + inner_w * (day - start).days / span

    def y_of(v):
        return PAD_T + inner_h * (1 - (math.log10(v) - lo) / (hi - lo))

    parts = [f'<svg class="tr-svg" viewBox="0 0 {W} {height}" role="img" '
             f'aria-label="Bitcoin price since 2011 on a log scale, red from each cycle top to its bottom and green back up">']
    for value in (100, 1_000, 10_000, 100_000):
        if lo <= math.log10(value) <= hi:
            y = y_of(value)
            parts.append(f'<line class="tr-grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>'
                         f'<text class="tr-axis" x="{W - PAD_R + 8}" y="{y + 4:.1f}">{_short(value)}</text>')
    for year in range(start.year + 1, end.year + 1):
        day = pd.Timestamp(f"{year}-01-01")
        if year % 2 == 0:
            parts.append(f'<text class="tr-axis" x="{x_of(day):.1f}" y="{H - 8}" text-anchor="middle">{year}</text>')
    for lap in timing.laps:
        x = x_of(lap.halving)
        parts.append(f'<line class="tr-halving" x1="{x:.1f}" x2="{x:.1f}" y1="{PAD_T}" y2="{H - PAD_B}"/>'
                     f'<text class="tr-axis tr-halving-label" x="{x + 4:.1f}" y="{PAD_T + 10}">halving</text>')

    # Legs between turning points. The stretch after this lap's lowest close so
    # far is gold: it is neither a finished fall nor a confirmed rise yet.
    legs = []
    points = [close.index[0]]
    kinds = []
    for lap in timing.laps:
        points.append(lap.top)
        kinds.append("up")
        if lap.bottom is not None:
            points.append(lap.bottom)
            kinds.append("down")
    points.append(close.index[-1])
    kinds.append("open" if not timing.current.bottom_final else "up")
    for (a, b), kind in zip(zip(points[:-1], points[1:]), kinds):
        if b <= a:
            continue
        legs.append((a, b, kind))

    for a, b, kind in legs:
        piece = series[(series.index >= a) & (series.index <= b)]
        if len(piece) < 2:
            continue
        xs = [x_of(d) for d in piece.index]
        ys = [y_of(v) for v in piece]
        delay = DRAW_SECONDS * (xs[0] - PAD_L) / inner_w
        duration = max(DRAW_SECONDS * (xs[-1] - xs[0]) / inner_w, 0.05)
        parts.append(f'<path class="tr-line {kind}" pathLength="1" d="{_path(xs, ys)}" '
                     f'style="--d:{delay:.2f}s;--t:{duration:.2f}s"/>')

        # The lane: one bar per finished leg, its length in days written on it.
        if kind in ("up", "down") and a != close.index[0]:
            days = (b - a).days
            x0, x1 = x_of(a), x_of(b)
            y = H + 6
            arrow = "&#8593;" if kind == "up" else "&#8595;"
            # This lap's fall is not over until the next halving says so: its
            # bar is gold and its count carries a plus, not a final number.
            unfinished = (kind == "down" and b == timing.current.bottom
                          and not timing.current.bottom_final)
            bar = "open" if unfinished else kind
            days_text = f"{days:,}+" if unfinished else f"{days:,}"
            parts.append(f'<rect class="tr-leg {bar}" x="{x0:.1f}" y="{y}" width="{x1 - x0:.1f}" height="16" rx="4" '
                         f'style="--d:{delay + duration:.2f}s"/>'
                         f'<text class="tr-leg-label" x="{(x0 + x1) / 2:.1f}" y="{y + 12}" text-anchor="middle" '
                         f'style="--d:{delay + duration:.2f}s">{arrow} {days_text} d</text>')

    for lap in timing.laps:
        x, y = x_of(lap.top), y_of(lap.top_price)
        parts.append(f'<circle class="tr-turn down" cx="{x:.1f}" cy="{y:.1f}" r="4"/>'
                     f'<text class="tr-turn-label" x="{x:.1f}" y="{y - 9:.1f}" text-anchor="middle">top {_short(lap.top_price)}</text>')
        if lap.bottom is not None:
            x, y = x_of(lap.bottom), y_of(lap.bottom_price)
            label = "low so far" if not lap.bottom_final else "bottom"
            parts.append(f'<circle class="tr-turn up" cx="{x:.1f}" cy="{y:.1f}" r="4"/>'
                         f'<text class="tr-turn-label" x="{x:.1f}" y="{y + 18:.1f}" text-anchor="middle">{label} {_short(lap.bottom_price)}</text>')

    # Where the folk calendar puts this lap's bottom, as a dashed line into
    # the future. It is the only thing on the site drawn past today, and it is
    # labelled as the calendar's date, not the lab's.
    x = x_of(timing.folk_bottom_date)
    parts.append(f'<line class="tr-folk" x1="{x:.1f}" x2="{x:.1f}" y1="{PAD_T}" y2="{H - PAD_B}"/>'
                 f'<text class="tr-axis tr-folk-label" x="{x - 4:.1f}" y="{H - PAD_B - 8}" text-anchor="end">'
                 f'folk "365 days": {timing.folk_bottom_date:%d %b %Y}</text>')

    x_now, y_now = x_of(timing.as_of), y_of(timing.last_price)
    parts.append(f'<circle class="tr-ring open" cx="{x_now:.1f}" cy="{y_now:.1f}" r="6"/>'
                 f'<circle class="tr-dot open" cx="{x_now:.1f}" cy="{y_now:.1f}" r="5"/>')
    parts.append("</svg>")
    return f'<div class="tr-box tr-anim" id="{element_id}">{"".join(parts)}</div>'


def cycle_meters(timing: CycleTiming) -> str:
    """How far into the calendar this lap is, as two bars that fill."""
    downs = [d for *_, d in timing.down_days()]
    ups = [d for *_, d in timing.up_days()]
    since = timing.days_since_top
    scale_down = max(downs + [FOLK_DOWN_DAYS, since]) * 1.08
    scale_up = max(ups + [FOLK_UP_DAYS]) * 1.08

    def ticks(values, scale, kind):
        return "".join(
            f'<i class="tr-tick {kind}" style="left:{100 * v / scale:.1f}%" title="{v} days"></i>'
            for v in values
        )

    left = FOLK_DOWN_DAYS - since
    if left > 0:
        down_text = (f"<b>Day {since}</b> since the top on {timing.current.top:%d %b %Y}. "
                     f"The folk calendar says the fall lasts {FOLK_DOWN_DAYS} days - "
                     f"{left} more, to about {timing.folk_bottom_date:%d %b %Y}.")
    else:
        down_text = (f"<b>Day {since}</b> since the top on {timing.current.top:%d %b %Y} - "
                     f"past the calendar's {FOLK_DOWN_DAYS} days by {-left}.")
    low = timing.current
    if low.bottom is not None:
        down_text += (f" Lowest close so far: {_money(low.bottom_price)} on {low.bottom:%d %b %Y}, "
                      f"{low.down_days} days after the top - final only when the next halving arrives.")

    up_text = (f"The rise from the bottom to the next top took {', '.join(f'{u:,}' for u in ups)} days. "
               f"If this lap kept the calendar, the next top would come around "
               f"<b>{timing.folk_next_top_date:%b %Y}</b>.")
    fill = min(since / scale_down, 1.0)
    return (
        '<div class="tr-meters tr-anim">'
        '<div class="tr-meter">'
        f'<div class="tr-meter-head"><span class="tr-badge down">The fall</span>'
        f'<span>past laps: {", ".join(str(d) for d in downs)} days &middot; folk: {FOLK_DOWN_DAYS}</span></div>'
        f'<div class="tr-bar"><span class="tr-fill down" style="--w:{100 * fill:.1f}%"></span>'
        f'{ticks(downs, scale_down, "down")}'
        f'<i class="tr-tick folk" style="left:{100 * FOLK_DOWN_DAYS / scale_down:.1f}%"></i></div>'
        f'<p>{down_text}</p></div>'
        '<div class="tr-meter">'
        f'<div class="tr-meter-head"><span class="tr-badge up">The rise</span>'
        f'<span>past laps: {", ".join(f"{u:,}" for u in ups)} days &middot; folk: {FOLK_UP_DAYS:,}</span></div>'
        f'<div class="tr-bar"><span class="tr-fill up" style="--w:0%"></span>'
        f'{ticks(ups, scale_up, "up")}'
        f'<i class="tr-tick folk" style="left:{100 * FOLK_UP_DAYS / scale_up:.1f}%"></i></div>'
        f'<p>{up_text}</p></div>'
        "</div>"
    )


TREND_ASSETS = """<style>
  .tr-box{background:var(--panel);border-radius:var(--r);padding:16px 16px 12px;margin-top:18px}
  .tr-svg{display:block;width:100%;height:auto;overflow:visible}
  .tr-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:8px}
  .tr-sp{flex:1 1 auto}
  .tr-badge{padding:5px 13px;border-radius:999px;font-size:13px;font-weight:700;
    letter-spacing:.08em;text-transform:uppercase;transition:background .6s,color .6s}
  .tr-badge.up{background:rgba(47,207,131,.16);color:var(--up);box-shadow:0 0 18px rgba(47,207,131,.25)}
  .tr-badge.down{background:rgba(255,90,95,.16);color:var(--down);box-shadow:0 0 18px rgba(255,90,95,.25)}
  .tr-since{color:var(--dim);font-size:15px}
  .tr-quote{display:flex;align-items:center;gap:8px;font-size:15px;color:var(--soft);
    font-variant-numeric:tabular-nums}
  .tr-pip{width:8px;height:8px;border-radius:50%;background:#4a5666;display:inline-block}
  .tr-pip.on{background:var(--gold);box-shadow:0 0 10px var(--gold)}
  .tr-note{color:var(--soft);font-size:15px;margin:10px 2px 2px;max-width:none}
  .tr-live-note b{color:var(--gold)}
  .tr-grid{stroke:rgba(255,255,255,.06);stroke-width:1}
  .tr-axis{fill:var(--dim);font:12px var(--sans)}
  .tr-avg{fill:none;stroke:#6b7686;stroke-width:1.3;stroke-dasharray:5 5;opacity:0}
  .tr-fast{stroke:#a9b3c2;stroke-dasharray:2 4}
  .tr-line{fill:none;stroke-width:2.4;stroke-linejoin:round;stroke-linecap:round;
    stroke-dasharray:1;stroke-dashoffset:1}
  .tr-line.up{stroke:var(--up);filter:drop-shadow(0 0 4px rgba(47,207,131,.55))}
  .tr-line.down{stroke:var(--down);filter:drop-shadow(0 0 4px rgba(255,90,95,.55))}
  .tr-line.open{stroke:var(--gold);filter:drop-shadow(0 0 4px rgba(247,147,26,.55))}
  .tr-area{opacity:0}
  .tr-area.up{fill:rgba(47,207,131,.07)}
  .tr-area.down{fill:rgba(255,90,95,.07)}
  .tr-comet{fill:#fff;opacity:0;filter:drop-shadow(0 0 6px #fff)}
  .tr-trade{opacity:0;transform-box:fill-box;transform-origin:center;stroke:#05070a;stroke-width:1.2}
  .tr-trade.up{fill:var(--up)} .tr-trade.down{fill:var(--down)}
  .tr-dot.up,.tr-turn.up{fill:var(--up)} .tr-dot.down,.tr-turn.down{fill:var(--down)}
  .tr-dot.open{fill:var(--gold)}
  .tr-dot,.tr-ring{opacity:0}
  .tr-ring{fill:none;stroke-width:2;transform-box:fill-box;transform-origin:center}
  .tr-ring.up{stroke:var(--up)} .tr-ring.down{stroke:var(--down)} .tr-ring.open{stroke:var(--gold)}
  .tr-live{fill:var(--gold);opacity:0;transition:cy .8s ease,opacity .4s}
  .tr-live-link{stroke:var(--gold);stroke-width:1.2;stroke-dasharray:3 3;opacity:0;transition:y2 .8s ease,opacity .4s}
  .tr-halving{stroke:rgba(247,147,26,.28);stroke-width:1;stroke-dasharray:2 4}
  .tr-halving-label{fill:rgba(247,147,26,.7);font-size:11px}
  .tr-folk{stroke:var(--gold);stroke-width:1.4;stroke-dasharray:6 5;opacity:.8}
  .tr-folk-label{fill:var(--gold)}
  .tr-turn{opacity:.95;stroke:#05070a;stroke-width:1.2}
  .tr-turn-label{fill:var(--soft);font:11.5px var(--sans)}
  .tr-leg{opacity:0;transform-box:fill-box;transform-origin:left center}
  .tr-leg.up{fill:rgba(47,207,131,.28)} .tr-leg.down{fill:rgba(255,90,95,.3)} .tr-leg.open{fill:rgba(247,147,26,.3)}
  .tr-leg-label{fill:var(--ink);font:600 11.5px var(--sans);opacity:0}

  .tr-meters{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px;margin-top:18px}
  .tr-meter{background:var(--panel);border-radius:var(--r);padding:16px 18px}
  .tr-meter-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap;color:var(--dim);font-size:14px}
  .tr-meter p{color:var(--soft);font-size:15px;margin:12px 0 0}
  .tr-meter p b{color:var(--ink)}
  .tr-bar{position:relative;height:14px;border-radius:999px;background:var(--raise);margin-top:14px;overflow:visible}
  .tr-fill{position:absolute;left:0;top:0;bottom:0;width:0;border-radius:999px}
  .tr-fill.down{background:linear-gradient(90deg,rgba(255,90,95,.35),var(--down))}
  .tr-fill.up{background:linear-gradient(90deg,rgba(47,207,131,.35),var(--up))}
  .tr-tick{position:absolute;top:-4px;width:2px;height:22px;border-radius:2px;background:var(--ink);opacity:.55}
  .tr-tick.folk{background:var(--gold);opacity:1;width:3px}

  /* Everything waits until the chart is on screen: an animation that plays
     below the fold has played for nobody. */
  .play .tr-line{animation:tr-draw var(--t) linear var(--d) forwards}
  .play .tr-area{animation:tr-fade .8s ease calc(var(--d) + .2s) forwards}
  .play .tr-avg{animation:tr-fade 1.2s ease .3s forwards}
  .play .tr-trade{animation:tr-pop .5s cubic-bezier(.3,1.6,.5,1) var(--d) forwards}
  .play .tr-dot{animation:tr-fade .3s ease 2.8s forwards}
  .play .tr-ring{animation:tr-pulse 1.8s ease-out 2.9s infinite}
  .play .tr-comet{animation:tr-fade .4s ease 3s forwards}
  .play .tr-leg{animation:tr-grow .6s ease var(--d) forwards}
  .play .tr-leg-label{animation:tr-fade .5s ease calc(var(--d) + .4s) forwards}
  .play .tr-fill{animation:tr-fill 1.8s cubic-bezier(.2,.8,.2,1) .2s forwards}
  @keyframes tr-draw{to{stroke-dashoffset:0}}
  @keyframes tr-fade{to{opacity:1}}
  @keyframes tr-pop{from{opacity:0;transform:scale(.2)}to{opacity:1;transform:scale(1)}}
  @keyframes tr-grow{from{opacity:0;transform:scaleX(0)}to{opacity:1;transform:scaleX(1)}}
  @keyframes tr-pulse{0%{opacity:.9;transform:scale(1)}100%{opacity:0;transform:scale(3)}}
  @keyframes tr-fill{to{width:var(--w)}}
  /* The drawing is 1000 units wide and a phone shows it at about a third of
     that, so the lettering is set in drawing units large enough to survive. */
  @media (max-width:640px){
    .tr-axis,.tr-turn-label{font-size:24px} .tr-leg-label{font-size:20px}
    .tr-halving-label{display:none} .tr-box{padding:12px 10px 10px}
  }
  @media (prefers-reduced-motion:reduce){
    .tr-line{stroke-dashoffset:0} .tr-area,.tr-avg,.tr-trade,.tr-dot,.tr-leg,.tr-leg-label{opacity:1}
    .tr-fill{width:var(--w)} .play *{animation:none !important} .tr-comet{display:none}
  }
</style>
<script>
  (function () {
    var boxes = document.querySelectorAll('.tr-anim');
    function play(box) {
      if (box.classList.contains('play')) { return; }
      box.classList.add('play');
      window.setTimeout(function () {
        box.querySelectorAll('animateMotion').forEach(function (m) {
          try { m.beginElement(); } catch (err) { /* no SMIL here */ }
        });
      }, 3000);
    }
    if (!window.IntersectionObserver) { boxes.forEach(play); }
    else {
      var seen = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) { play(e.target); seen.unobserve(e.target); }
        });
      }, { threshold: 0.25 });
      boxes.forEach(function (box) { seen.observe(box); });
    }

    /* The live dot. It moves; the history does not. A day's colour is earned
       by the close, so a tick that crosses the average is reported in words
       and the line stays as the days left it. */
    var ribbons = document.querySelectorAll('.tr-box[data-slow]');
    if (!ribbons.length) { return; }
    function money(v) { return '$' + Math.round(v).toLocaleString('en-US'); }
    function show(price) {
      ribbons.forEach(function (box) {
        var d = box.dataset, low = +d.low, high = +d.high;
        var y = +d.top + +d.inner * (1 - (price - low) / (high - low));
        y = Math.max(4, Math.min(+d.top + +d.inner + 10, y));
        var dot = box.querySelector('.tr-live'), link = box.querySelector('.tr-live-link');
        dot.setAttribute('cy', y.toFixed(1)); dot.style.opacity = 1;
        link.setAttribute('y2', y.toFixed(1)); link.style.opacity = .8;
        box.querySelector('.tr-price').textContent = 'live ' + money(price);
        box.querySelector('.tr-pip').className = 'tr-pip on';
        if (d.mode !== 'price') { return; }
        var slow = +d.slow, liveUp = price > slow, dayUp = d.up === '1';
        var note = box.querySelector('.tr-live-note');
        note.innerHTML = liveUp === dayUp ? ''
          : '<b>Right now the live price is ' + (liveUp ? 'above' : 'below')
            + ' the average (' + money(slow) + ').</b> The colour changes only if a day closes there.';
      });
    }
    function poll() {
      fetch('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT', { cache: 'no-store' })
        .then(function (r) { return r.json(); })
        .then(function (j) { var p = parseFloat(j.price); if (isFinite(p)) { show(p); } })
        .catch(function () { /* offline: the last close stays on the chart */ });
    }
    poll();
    window.setInterval(poll, 15000);
  })();
</script>"""
