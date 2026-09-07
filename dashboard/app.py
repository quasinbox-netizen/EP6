"""Streamlit dashboard - presentation layer only.

The rule: no research logic in this file. Every number comes from a module in
src/ and passes through the same functions as the CLI. If you want to change
how something is computed, do it in src/ - otherwise the chart and the
terminal will start showing different things and you will believe the
prettier one.

Run it with:
    python run.py dashboard
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from analysis.evidence import build as build_evidence  # noqa: E402
from analysis.outlook import MIN_EFFECTIVE_N  # noqa: E402
from backtest.sizing import apply_rebalance_band  # noqa: E402
from config import load_config  # noqa: E402
from forecast.ledger import load as load_ledger  # noqa: E402
from forecast.ledger import score as score_ledger  # noqa: E402
from forecast.ledger import scoreboard as ledger_scoreboard  # noqa: E402
from forecast.ledger import verdict as ledger_verdict  # noqa: E402
from features.halving import CONFIRMED_HALVINGS  # noqa: E402
from pipeline import (  # noqa: E402
    category_event_studies,
    control_comparison,
    forecast_report,
    halving_event_study,
    load_lab_data,
    out_of_sample_check,
    cycle_outlook,
    run_strategies,
    scan_hypotheses,
    strategy_signals,
)
from validation.multiple_testing import summarize  # noqa: E402

st.set_page_config(page_title="BTC Cycle Lab", layout="wide")

INTRO = Path(__file__).with_name("intro.html")
INTRO_START, INTRO_END = "<!--INTRO:START-->", "<!--INTRO:END-->"

METER = Path(__file__).with_name("evidence_meter.html")
METER_START, METER_END = "<!--METER:START-->", "<!--METER:END-->"
METER_DATA_START, METER_DATA_END = "<!--DATA:START-->", "<!--DATA:END-->"

DESK = Path(__file__).with_name("signal_desk.html")
DESK_START, DESK_END = "<!--DESK:START-->", "<!--DESK:END-->"
DESK_DATA_START, DESK_DATA_END = "<!--DATA:START-->", "<!--DATA:END-->"
# Five years of daily closes. Two fitted the panel more comfortably, but the
# rules trade rarely - the trend crossover produced four signals in two years -
# and a replay of four markers is over before it reads as a sequence. The line
# is thinned to about 900 points before it reaches the canvas, so the extra
# history costs drawing time rather than legibility.
DESK_WINDOW_DAYS = 1825


def play_intro() -> None:
    """Drop the intro animation over the whole page, once per session.

    The animation lives in intro.html, which also opens on its own in a
    browser; the markers carve out the part that is portable between the two.

    Why an iframe that writes into its parent, rather than the simpler
    `st.html`: an overlay has to escape whatever element Streamlit puts it in,
    and `st.html` content belongs to Streamlit's element tree. It was tried
    first and the animation kept dying halfway through - the element is
    re-rendered on a rerun or a socket reconnect, and a <script> that arrives
    through innerHTML the second time never executes, leaving a black
    rectangle over the app. An iframe's scripts run on every mount, and what
    they append to the parent's <body> is nobody else's to remove.

    The iframe itself carries no visible content and is zero-sized.
    """
    if not INTRO.exists():
        return
    markup = INTRO.read_text(encoding="utf-8")
    fragment = markup.split(INTRO_START, 1)[1].split(INTRO_END, 1)[0]
    # A literal </script> inside the string would close the loader early.
    payload = json.dumps(fragment).replace("</", r"<\/")
    # A remount of the same iframe must not restart the animation, but the
    # replay button must. The token tells the two apart.
    token = json.dumps(str(st.session_state.get("intro_plays", 0)))
    st.iframe(
        """
<script>
(function () {
  var host;
  try { host = window.parent; } catch (err) { return; }
  if (!host || !host.document || !host.document.body) return;
  if (host.__btclabIntroToken === __TOKEN__) return;
  host.__btclabIntroToken = __TOKEN__;

  var doc = host.document;
  var holder = doc.createElement('div');
  holder.innerHTML = __FRAGMENT__;
  Array.prototype.slice.call(holder.childNodes).forEach(function (node) {
    if (node.tagName === 'SCRIPT') {
      // innerHTML never executes a script; rebuilding the element does.
      var script = doc.createElement('script');
      script.textContent = node.textContent;
      doc.body.appendChild(script);
    } else {
      doc.body.appendChild(node);
    }
  });
})();
</script>
""".replace("__FRAGMENT__", payload).replace("__TOKEN__", token),
        # 1x1 rather than 0: st.iframe rejects a zero width outright, and the
        # loader has nothing to show anyway.
        height=1,
        width=1,
    )


# The ledger stores timestamps; nobody needs to read 00:00:00 seven times.
LEDGER_DATE_COLUMNS = {
    "as_of": st.column_config.DateColumn("as of", format="YYYY-MM-DD"),
    "target_date": st.column_config.DateColumn("settles on", format="YYYY-MM-DD"),
}

COLORS = {
    "price": "#e8a33d",
    "car": "#4c8bf5",
    "band": "rgba(76, 139, 245, 0.18)",
    "halving": "rgba(232, 163, 61, 0.14)",
    "zero": "#8a8a8a",
}


@st.cache_data(show_spinner="Loading data from the local database...")
def cached_data():
    return load_lab_data()


@st.cache_data(show_spinner="Computing the event study...")
def cached_halving_study(post: int):
    return halving_event_study(cached_data(), post=post)


@st.cache_data(show_spinner="Computing event studies per category...")
def cached_category_studies(post: int):
    return category_event_studies(cached_data(), post=post)


@st.cache_data(show_spinner="Comparing against the control group...")
def cached_control(post: int):
    return control_comparison(cached_data(), post=post)


@st.cache_data(show_spinner="Fitting and scoring the forecast...")
def cached_forecast():
    return forecast_report(cached_data())


def processed_dir(config) -> Path:
    """Where the CLI leaves its output. Relative paths are repo-relative."""
    directory = Path(config["paths"]["processed"])
    return directory if directory.is_absolute() else config.root / directory


@st.cache_data(show_spinner="Reading the specification curve...")
def load_specification_results(_config):
    """Read what `run.py speccurve` saved. Never compute it here.

    One curve is 160 event studies and the permutation test is 201 of those -
    about ten minutes. Streamlit renders every tab body on every run, so
    computing it here would freeze the whole dashboard for anyone who never
    opens this tab.
    """
    directory = processed_dir(_config)

    def read(name):
        path = directory / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    return read("specification_curve.csv"), read("specification_summary.csv")


@st.cache_data(show_spinner="Reading the sizing results...")
def load_sizing_results(_config):
    """Read what `run.py sizing` saved. Never compute it here.

    Building the volatility series is a GARCH refit every 30 days over 4000
    days - about six minutes. Streamlit renders every tab body on every run, so
    computing it here would freeze the dashboard for everyone who never opens
    this tab, the same reason the specification curve is read rather than run.
    """
    directory = Path(_config["paths"]["processed"])
    if not directory.is_absolute():
        directory = _config.root / directory

    def read(name, **kwargs):
        path = directory / name
        return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()

    volatility = read("conditional_volatility.csv", index_col=0, parse_dates=True)
    series = volatility.iloc[:, 0] if not volatility.empty else pd.Series(dtype=float)
    return read("sizing_comparison.csv", index_col=0), read("sizing_today.csv"), series


def sizing_chart(volatility: pd.Series, position: pd.Series) -> go.Figure:
    """Forecast volatility and the position it implies, on one time axis.

    Two axes on purpose. The point of the picture is that the position is the
    mirror of the volatility - it should be visible at a glance that the line
    goes down exactly where the other goes up, because that is the entire
    mechanism and any other shape would mean something is wrong.
    """
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=volatility.index, y=volatility * (365 ** 0.5),
        name="forecast volatility (annualised)",
        line={"color": "#d1495b", "width": 1},
    ))
    figure.add_trace(go.Scatter(
        x=position.index, y=position, name="position held",
        yaxis="y2", line={"color": "#4a6fa5", "width": 1.5},
    ))
    figure.update_layout(
        # Log scale on volatility. The March 2020 spike reaches 340% against a
        # median of 60%, and on a linear axis it flattens eleven years of the
        # series into the bottom third - the spike stays visible either way, but
        # only this way is the rest of it readable.
        yaxis={
            "title": "annualised volatility (log)", "type": "log",
            "tickformat": ".0%",
        },
        yaxis2={
            "title": "position", "overlaying": "y", "side": "right",
            "range": [0, 1.05], "tickformat": ".0%", "showgrid": False,
        },
        height=420,
        margin={"l": 60, "r": 60, "t": 30, "b": 40},
        legend={"orientation": "h", "y": 1.12},
    )
    return figure


def specification_chart(curve: pd.DataFrame) -> go.Figure:
    """Specifications sorted by effect, with their intervals.

    The convention of the specification curve: order by estimate, not by name.
    What matters is the shape of the whole distribution and where zero sits in
    it - a reader should be able to see at a glance whether the answer depends
    on which analytical choices were made.
    """
    usable = curve[curve["car"].notna()].sort_values("car").reset_index(drop=True)
    position = list(range(len(usable)))
    significant = usable["significant"].astype(bool)

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=position + position[::-1],
        y=list(usable["ci_high"]) + list(usable["ci_low"])[::-1],
        fill="toself", fillcolor="rgba(120,140,180,0.18)",
        line={"width": 0}, hoverinfo="skip", name="95% confidence interval",
    ))
    figure.add_trace(go.Scatter(
        x=position, y=usable["car"], mode="markers",
        marker={
            "size": 6,
            "color": ["#d1495b" if s else "#4a6fa5" for s in significant],
        },
        text=usable["label"],
        customdata=usable[["n_events", "p_value"]],
        hovertemplate=(
            "%{text}<br>CAR %{y:+.1%}<br>"
            "events %{customdata[0]}<br>p=%{customdata[1]:.3f}<extra></extra>"
        ),
        name="CAR at the horizon",
    ))
    figure.add_hline(y=0, line_dash="dash", line_color="gray")
    # Scale to the estimates, not to the intervals. Two-event specifications
    # carry intervals of +-1000%, and letting those set the range flattens
    # every estimate into a line at zero. The intervals are still drawn and
    # still run off the top and bottom of the chart, which is the honest
    # picture: the band is genuinely that wide, and the caption says so rather
    # than the axis quietly hiding it.
    span = max(abs(usable["car"].min()), abs(usable["car"].max()))
    figure.update_yaxes(range=[-1.4 * span, 1.4 * span])
    figure.update_layout(
        xaxis_title="specifications, ordered by effect size",
        yaxis_title="cumulative abnormal return",
        yaxis_tickformat=".0%",
        height=460,
        margin={"l": 60, "r": 20, "t": 30, "b": 50},
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


@st.cache_data(show_spinner="Placing the cycle...")
def cached_outlook():
    return cycle_outlook(cached_data())


@st.cache_data(show_spinner="Reading the calibrated range forecast...")
def load_range_forecast(_config, days: int):
    """Read what `run.py range --days N --calibrate` left behind.

    Never computed here: one calibration is a walk over 5,000 days refitting
    GARCH every 25, which is minutes. More to the point, an interval that has
    not been through that walk has not earned a place on the front page - so
    a missing file has to stay missing rather than be filled in live.
    """
    directory = processed_dir(_config)

    forecast = directory / f"range_forecast_{days}d.csv"
    calibration = directory / f"range_calibration_{days}d.csv"
    if not forecast.exists() or not calibration.exists():
        return None, None
    return pd.read_csv(forecast), pd.read_csv(calibration)


def cycle_clock(paths: dict, days_now: int) -> go.Figure:
    """Every halving cycle on one dial, normalised to its halving day.

    Polar because the thing being read is a position in a repeating cycle, and
    a reader should see at a glance that the four laps do not retrace each
    other. The radius is log-scaled by hand - Plotly has no log radial axis -
    so a x2 and a x10 are not drawn as if they were a step apart.
    """
    figure = go.Figure()
    ghosts = ["#5c6b80", "#6f7f96", "#8496ad"]
    ordered = sorted(paths.items())

    for position, (label, series) in enumerate(ordered):
        current = position == len(ordered) - 1
        theta = [360.0 * day / 1461.0 for day in series.index]
        radius = [max(0.0, math.log10(value) + 1.0) if value > 0 else 0.0
                  for value in series.to_numpy()]
        figure.add_trace(go.Scatterpolar(
            r=radius, theta=theta, mode="lines",
            name=("this cycle" if current else label[:4]),
            line={"color": COLORS["price"] if current else ghosts[position % len(ghosts)],
                  "width": 2.6 if current else 1.1},
            opacity=1.0 if current else 0.55,
            customdata=[[day, value] for day, value in zip(series.index, series.to_numpy())],
            hovertemplate="%{customdata[0]:.0f} days after the halving<br>"
                          "x%{customdata[1]:.2f} of the halving price<extra></extra>",
        ))
        if current:
            figure.add_trace(go.Scatterpolar(
                r=[radius[-1]], theta=[theta[-1]], mode="markers",
                marker={"size": 11, "color": COLORS["price"],
                        "line": {"color": "#0b0f15", "width": 2}},
                name="today", hoverinfo="skip", showlegend=False,
            ))

    ticks = [0.5, 1, 2, 5, 10, 20]
    figure.update_layout(
        polar={
            "radialaxis": {
                "tickvals": [math.log10(t) + 1.0 for t in ticks],
                "ticktext": [f"x{t:g}" for t in ticks],
                "angle": 90, "tickfont": {"size": 10}, "gridcolor": "rgba(140,155,175,.18)",
            },
            "angularaxis": {
                "direction": "clockwise", "rotation": 90,
                "tickvals": [0, 90, 180, 270],
                "ticktext": ["halving", "1 year", "2 years", "3 years"],
                "gridcolor": "rgba(140,155,175,.18)",
            },
        },
        height=430, margin={"l": 30, "r": 30, "t": 30, "b": 30},
        legend={"orientation": "h", "y": -0.08},
    )
    return figure


def range_chart(forecast: pd.DataFrame, close: pd.Series, days: int) -> go.Figure:
    """The last months of price, then the calibrated bands at the horizon.

    The bands are drawn only at the horizon, not as a widening cone. The
    coverage test scored the endpoint and nothing else, so a cone would be
    drawing three months of confidence that was never checked.
    """
    recent = close.iloc[-120:]
    last_date, last_price = recent.index[-1], float(recent.iloc[-1])
    target = last_date + pd.Timedelta(days=days)

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=recent.index, y=recent, mode="lines", name="BTC/USD",
        line={"color": COLORS["price"], "width": 1.4},
    ))
    shades = ["rgba(76,139,245,.45)", "rgba(76,139,245,.28)", "rgba(76,139,245,.16)"]
    for position, row in enumerate(forecast.sort_values("level").itertuples()):
        figure.add_trace(go.Scatter(
            x=[target, target], y=[row.low, row.high], mode="lines",
            line={"color": shades[position % len(shades)], "width": 12},
            name=f"{row.level:.0%} interval",
            hovertemplate=(f"{row.level:.0%} of the time between "
                           f"{row.low:,.0f} and {row.high:,.0f}<extra></extra>"),
        ))
    figure.add_trace(go.Scatter(
        x=[last_date, target], y=[last_price, last_price], mode="lines",
        line={"color": COLORS["zero"], "width": 1, "dash": "dot"},
        name="today's price", hoverinfo="skip",
    ))
    figure.update_layout(
        height=350, margin={"l": 10, "r": 10, "t": 20, "b": 40},
        hovermode="x unified", yaxis_title="USD",
        legend={"orientation": "h", "y": -0.28},
        # The bands are drawn as thick lines, so half their width sits beyond
        # the horizon date and needs room. Scaled to the horizon rather than
        # fixed, or a 10-day chart gets the padding of a 365-day one.
        xaxis={"range": [recent.index[0], target + pd.Timedelta(days=max(4, days // 3))]},
    )
    return figure


@st.cache_data(show_spinner="Reading the strategy signals...")
def cached_signals():
    return strategy_signals(cached_data())


def desk_payload(bundle: dict) -> dict:
    """Reshape the signal bundle for the panel. No arithmetic on results.

    Everything here is pixels-and-labels work: cut the window, thin it out so
    the canvas is not asked to draw 5,000 segments, and translate dates into
    array positions. Every number passed through is one the pipeline already
    computed.
    """
    close = bundle["price"]
    window = close.iloc[-DESK_WINDOW_DAYS:]
    step = max(1, len(window) // 700)
    shown = window.iloc[::step]
    if shown.index[-1] != window.index[-1]:  # never drop the latest close
        shown = pd.concat([shown, window.iloc[[-1]]])

    def slot(timestamp):
        """Nearest drawn point, or None when the date is off the left edge."""
        stamp = pd.Timestamp(timestamp)
        if stamp < shown.index[0]:
            return None
        position = int(shown.index.searchsorted(stamp))
        return min(position, len(shown) - 1)

    strategies = []
    for item in bundle["strategies"]:
        signal = item["report"]
        held = item["positions"].reindex(shown.index).fillna(0.0).abs() > 0
        holds, start = [], None
        for i, flag in enumerate(held.tolist()):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                holds.append([start, i])
                start = None
        if start is not None:
            holds.append([start, len(held) - 1])

        marks = []
        for row in signal.trades.itertuples():
            entry = slot(row.entry_date)
            if entry is not None:
                marks.append({"i": entry, "side": "buy"})
            if not row.open:
                exit_slot = slot(row.exit_date)
                if exit_slot is not None:
                    marks.append({"i": exit_slot, "side": "sell"})

        metrics = signal.metrics
        strategies.append({
            "name": signal.name,
            "side": signal.side,
            "since": None if signal.since is None else signal.since.strftime("%Y-%m-%d"),
            "days": signal.days,
            "unrealized": _clean(signal.unrealized),
            "trigger": signal.trigger.as_dict(),
            "metrics": {
                "sharpe": _clean(metrics.get("sharpe")),
                "cagr": _clean(metrics.get("cagr")),
                "maxDrawdown": _clean(metrics.get("max_drawdown")),
                "totalReturn": _clean(metrics.get("total_return")),
            },
            "excessSharpe": _clean(item["excess_sharpe"]),
            "tradeCount": int(len(signal.trades)),
            "closedCount": signal.closed_trades,
            "winRateNet": _clean(signal.win_rate_net),
            "trades": [
                {
                    "in": row.entry_date.strftime("%Y-%m-%d"),
                    "out": row.exit_date.strftime("%Y-%m-%d"),
                    "net": _clean(row.return_net),
                    "open": bool(row.open),
                }
                for row in signal.trades.itertuples()
            ],
            "marks": marks,
            "holds": holds,
        })

    return {
        "asOf": bundle["as_of"].strftime("%Y-%m-%d"),
        "sampleDays": int(len(close)),
        "lastClose": float(close.iloc[-1]),
        "costRate": float(bundle["cost_rate"]),
        "price": {
            "d": [stamp.strftime("%Y-%m-%d") for stamp in shown.index],
            "c": [round(float(value), 2) for value in shown.tolist()],
        },
        "strategies": strategies,
    }


def _clean(value):
    """JSON has no NaN. Anything unmeasurable becomes null."""
    if value is None:
        return None
    number = float(value)
    return None if number != number or number in (float("inf"), float("-inf")) else number


def render_evidence_meter(payload: dict) -> None:
    """The two mascots as a gauge, driven by the validation results.

    Same iframe treatment as the desk, for the same reason: a rerun must not
    leave a dead panel behind. The numbers come from `analysis.evidence`,
    which is handed values the other tabs already computed - the meter can be
    wrong about taste but it cannot disagree with the tests it summarises.
    """
    if not METER.exists():
        return
    markup = METER.read_text(encoding="utf-8")
    fragment = markup.split(METER_START, 1)[1].split(METER_END, 1)[0]
    head, rest = fragment.split(METER_DATA_START, 1)
    _, tail = rest.split(METER_DATA_END, 1)
    encoded = json.dumps(payload).replace("</", r"<\/")
    st.iframe(head + encoded + tail, height=360)


def render_signal_desk(payload: dict) -> None:
    """The desk runs in an iframe: it owns a canvas, audio and a live poll.

    An iframe rather than `st.html` for the same reason as the intro - a
    rerun re-renders the element, and a component whose script does not run
    again is a dead panel. It also keeps the browser's audio permission,
    which is granted to the frame the user clicked in.
    """
    if not DESK.exists():
        st.info("signal_desk.html is missing.")
        return
    markup = DESK.read_text(encoding="utf-8")
    fragment = markup.split(DESK_START, 1)[1].split(DESK_END, 1)[0]
    head, rest = fragment.split(DESK_DATA_START, 1)
    _, tail = rest.split(DESK_DATA_END, 1)
    # An unescaped </script> inside the JSON would close the data block early.
    encoded = json.dumps(payload).replace("</", r"<\/")
    # Tall enough for the status column without a nested scrollbar on a
    # normal window; the panel itself stretches to fill it.
    st.iframe(head + encoded + tail, height=640)


@st.cache_data(show_spinner="Scanning hypotheses...")
def cached_scan():
    return scan_hypotheses(cached_data())


@st.cache_data(show_spinner="Checking out-of-sample replication...")
def cached_out_of_sample():
    return out_of_sample_check(cached_data())


@st.cache_data(show_spinner="Running backtests...")
def cached_backtests():
    table, results = run_strategies(cached_data())
    curves = pd.DataFrame({r.name: r.equity for r in results})
    return table, curves


def price_chart(data, halving_window_days: int, show_events: bool) -> go.Figure:
    frame = data.features
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["close"],
            name="BTC/USD",
            line=dict(color=COLORS["price"], width=1.2),
        )
    )
    for halving in CONFIRMED_HALVINGS:
        end = halving + pd.Timedelta(days=halving_window_days)
        figure.add_vrect(x0=halving, x1=end, fillcolor=COLORS["halving"], line_width=0)
        figure.add_vline(x=halving, line=dict(color=COLORS["price"], width=1, dash="dot"))

    if show_events and not data.events.empty:
        non_halving = data.events[data.events["category"] != "halving"]
        prices_by_day = frame["close"]
        for _, event in non_halving.iterrows():
            day = pd.Timestamp(event["available_from"]).normalize()
            if day not in prices_by_day.index:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[day],
                    y=[prices_by_day.loc[day]],
                    mode="markers",
                    marker=dict(size=8, symbol="diamond"),
                    name=event["name"],
                    hovertext=f"{event['category']}: {event['description']}",
                    showlegend=False,
                )
            )

    figure.update_layout(
        yaxis_type="log",
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="price (log scale)",
        hovermode="x unified",
    )
    return figure


def car_chart(result, title: str) -> go.Figure:
    table = result.table.dropna(subset=["car"])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=table.index, y=table["car_ci_high"], line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=table.index, y=table["car_ci_low"], fill="tonexty",
            fillcolor=COLORS["band"], line=dict(width=0),
            name="95% confidence interval",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=table.index, y=table["car"], name="mean CAR",
            line=dict(color=COLORS["car"], width=2),
        )
    )
    figure.add_hline(y=0, line=dict(color=COLORS["zero"], dash="dash", width=1))
    figure.update_layout(
        title=title,
        height=420,
        xaxis_title="days from the event",
        yaxis_title="cumulative abnormal return",
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis_tickformat=".0%",
    )
    return figure


def equity_chart(curves: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for column in curves.columns:
        figure.add_trace(
            go.Scatter(
                x=curves.index, y=curves[column], name=column,
                line=dict(width=2.5 if column == "buy and hold" else 1.4),
            )
        )
    figure.update_layout(
        yaxis_type="log", height=440, yaxis_title="equity (log scale)",
        margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
    )
    return figure


def main() -> None:
    # First thing in the run, so it lands before the data load blocks.
    if st.session_state.get("intro_pending", True):
        st.session_state["intro_pending"] = False
        st.session_state["intro_plays"] = st.session_state.get("intro_plays", 0) + 1
        play_intro()

    config = load_config()
    st.title("BTC Cycle Lab")
    st.caption(
        "A tool for testing whether the halving cycle and macro events explain "
        "anything in the price of BTC. Every result comes with a confidence "
        "interval and a count of observations - without those it is not a result."
    )

    data = cached_data()
    if data.is_empty:
        st.error(
            "The database is empty. Download the data first:\n\n"
            "`python run.py ingest --what all`"
        )
        st.stop()

    with st.sidebar:
        st.header("Settings")
        halving_window_days = st.slider("Halving window on the chart (days)", 30, 730, 365, 5)
        study_post = st.slider("Event study horizon (days after the event)", 30, 730, 365, 5)
        show_events = st.checkbox("Show events on the chart", value=True)
        st.divider()

        # Everything on this page is cached, so a fresh `ingest` is invisible
        # until the cache is dropped. The toolbar's "Clear cache" is hidden by
        # config.toml along with the Deploy button, so the button lives here
        # instead - where someone looking for it would actually look.
        if st.button("Reload data", icon=":material/refresh:", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.caption("Press this after running `ingest` to pick up new data.")

        if st.button("Replay intro", icon=":material/skull:", width="stretch"):
            st.session_state["intro_pending"] = True
            st.rerun()

        st.divider()
        st.caption(
            f"Data: {data.features.index.min().date()} - {data.features.index.max().date()} "
            f"({len(data.features)} days)\n\n"
            f"Stitching: {', '.join(config['price']['stitch_priority'])}\n\n"
            f"Costs: {config['backtest']['fee_bps']} bps fees + "
            f"{config['backtest']['slippage_bps']} bps slippage"
        )

    (
        tab_now, tab_price, tab_study, tab_control, tab_validation, tab_specs,
        tab_backtest, tab_sizing, tab_signals, tab_forecast, tab_receipts
    ) = st.tabs(
        ["Now", "Price and cycles", "Event study", "Control group", "Validation",
         "Specification curve", "Backtest", "Sizing", "Signal desk", "Forecast",
         "Receipts"]
    )

    with tab_now:
        outlook = cached_outlook()
        st.markdown(
            f"## {outlook.days_since_halving} days after the last halving"
        )
        st.caption(
            f"{outlook.cycle_label.capitalize()}, price {outlook.trend_label}. "
            f"Everything below is measured on the sample to "
            f"{outlook.as_of:%Y-%m-%d}; nothing here is a forecast of direction."
        )

        headline = st.columns(4)
        headline[0].metric("Last close", f"${outlook.last_price:,.0f}")
        headline[1].metric("Days since halving", f"{outlook.days_since_halving}")
        headline[2].metric("Cycle phase", outlook.cycle_label.split()[0].capitalize())
        headline[3].metric(
            "200-day trend", "Above" if "above" in outlook.trend_label else "Below"
        )

        scan_now = cached_scan()
        oos_now = cached_out_of_sample()
        study_now = cached_halving_study(365)
        _, curve_now = load_specification_results(config)
        ledger_now = load_ledger(processed_dir(config) / "predictions.csv")
        settled_now = score_ledger(ledger_now, data.features["close"]) if not ledger_now.empty             else ledger_now
        matured_now = (
            settled_now[settled_now["matured"].astype(bool)]
            if "matured" in getattr(settled_now, "columns", []) else pd.DataFrame()
        )
        evidence = build_evidence(
            survivors=int(scan_now["significant_adjusted"].sum()) if not scan_now.empty else 0,
            hypotheses=int(len(scan_now)),
            specifications_significant=(
                float(curve_now.iloc[0]["n_significant"]) if not curve_now.empty else 0.0
            ),
            specifications_null_mean=(
                float(curve_now.iloc[0]["null_significant_mean"]) if not curve_now.empty else 0.0
            ),
            halving_p_value=float(study_now.car_summary["p_value"]),
            replicated=int(oos_now["replicated"].sum()) if not oos_now.empty else 0,
            replication_attempts=int(len(oos_now)),
            settled_claims=int(len(matured_now)),
            kept_claims=int(matured_now["hit"].fillna(False).astype(bool).sum())
            if not matured_now.empty else 0,
        )
        st.subheader("How much has actually survived the tests")
        render_evidence_meter(evidence.as_dict())

        clock_column, range_column = st.columns([1, 1])
        with clock_column:
            st.subheader("The cycle, four laps on one dial")
            st.plotly_chart(
                cycle_clock(outlook.paths, outlook.days_since_halving),
                width="stretch",
            )
            st.caption(
                "Each lap starts at its own halving and is drawn as a multiple "
                "of the price on that day, on a log radius. The shape they "
                "share is the reason people believe in the cycle; the distance "
                "between them is the reason four of them cannot prove it."
            )

        with range_column:
            st.subheader("Where the price may be")
            horizon = 30 if load_range_forecast(config, 30)[0] is not None else 10
            forecast, calibration = load_range_forecast(config, horizon)
            if forecast is None:
                st.info(
                    "No calibrated interval yet. An interval that has never "
                    "been checked is a decoration, so none is drawn. Build one "
                    "with:\n\n`python run.py range --days 30 --calibrate`"
                )
            else:
                st.plotly_chart(
                    range_chart(forecast, data.features["close"], horizon),
                    width="stretch",
                )
                withheld = calibration.loc[~calibration["within_tolerance"], "level"]
                quoted = ", ".join(f"{level:.0%}" for level in forecast["level"])
                st.caption(
                    f"{horizon}-day interval from the volatility model, drawn only "
                    f"at the horizon it was scored at. Levels shown ({quoted}) kept "
                    "their promise in a walk-forward coverage test"
                    + (
                        "; " + ", ".join(f"{level:.0%}" for level in withheld)
                        + " failed and is withheld."
                        if len(withheld) else "."
                    )
                    + " It says how far, never which way."
                )

        st.subheader("What usually happened from days like today")
        conditional, unconditional = outlook.rates[0]
        cells = st.columns(3)
        cells[0].metric(
            f"Median {conditional.horizon}-day return, days like today",
            f"{conditional.median:+.1%}",
            delta=f"{conditional.median - unconditional.median:+.1%} vs every day",
        )
        cells[1].metric(
            "Share positive",
            f"{conditional.share_positive:.0%}",
            delta=f"{conditional.share_positive - unconditional.share_positive:+.0%}",
        )
        cells[2].metric("Independent windows", f"{conditional.effective_n}")

        rows = []
        for matched, whole in outlook.rates:
            rows.append({
                "horizon": f"{matched.horizon}d",
                "median %, days like today": 100 * matched.median,
                "positive %, days like today": 100 * matched.share_positive,
                "independent windows": matched.effective_n,
                "median %, every day": 100 * whole.median,
                "positive %, every day": 100 * whole.share_positive,
                "windows, every day": whole.effective_n,
                "quotable": "yes" if matched.usable else "no - too few windows",
            })
        st.dataframe(
            pd.DataFrame(rows), width="stretch", hide_index=True,
            column_config={
                "median %, days like today": st.column_config.NumberColumn(format="%.1f"),
                "positive %, days like today": st.column_config.NumberColumn(format="%.0f"),
                "median %, every day": st.column_config.NumberColumn(format="%.1f"),
                "positive %, every day": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        st.caption(
            "Matched days are not observations. Consecutive days share almost "
            "all of their forward window, so only non-overlapping ones are "
            f"counted, and a row needs {MIN_EFFECTIVE_N} of them before it is "
            "worth quoting. The long-horizon rows are exactly the ones that "
            "look most impressive and rest on the fewest."
        )

        st.subheader("If it moves like that")
        move = st.slider(
            "A move of at least this much over the next 30 days",
            min_value=-50, max_value=100, value=10, step=5, format="%d%%",
        )
        threshold = move / 100.0
        scenario = st.columns(3)
        scenario[0].metric(
            "Days like today that got there",
            f"{conditional.share_above(threshold):.0%}",
        )
        scenario[1].metric(
            "Every day that got there",
            f"{unconditional.share_above(threshold):.0%}",
        )
        scenario[2].metric("Independent windows behind it", f"{conditional.effective_n}")
        st.caption(
            "A count of how often the sample reached that move, not a "
            "probability that it will. The two columns exist so the answer "
            "can be compared with the answer for any day at all - if they "
            "match, the condition is not carrying information."
        )

        st.subheader("What the rules say, and what the tests say")
        signals = cached_signals()
        state_rows = []
        for item in signals["strategies"]:
            report = item["report"]
            state_rows.append({
                "rule": report.name,
                "position": "in" if report.side == "long" else "out",
                "since": "-" if report.since is None else f"{report.since:%Y-%m-%d}",
                "what flips it": report.trigger.text,
            })
        st.dataframe(pd.DataFrame(state_rows), width="stretch", hide_index=True)

        study = cached_halving_study(365)
        scan = cached_scan()
        curve, curve_summary = load_specification_results(config)
        verdict = [
            f"Halving effect after a year: {study.car_summary['car']:+.0%} "
            f"[{study.car_summary['ci_low']:+.0%}, {study.car_summary['ci_high']:+.0%}], "
            f"p={study.car_summary['p_value']:.2f}, n={study.n_events}.",
            summarize(scan) + ".",
        ]
        if not curve_summary.empty:
            row = curve_summary.iloc[0]
            verdict.append(
                f"Specification curve: {int(row['n_significant'])} of "
                f"{int(row['n_specs'])} specifications significant, against "
                f"{row['null_significant_mean']:.1f} for randomly placed dates."
            )
        st.error(
            "  \n".join(verdict)
            + "  \nRead the panels above as description, not as a plan."
        )

    with tab_price:
        st.plotly_chart(
            price_chart(data, halving_window_days, show_events), width="stretch"
        )
        columns = st.columns(4)
        columns[0].metric("Days in sample", f"{len(data.features):,}")
        columns[1].metric("Halvings in sample", len(CONFIRMED_HALVINGS))
        columns[2].metric("Events in registry", len(data.events))
        columns[3].metric("Macro series", data.macro["series"].nunique() if not data.macro.empty else 0)
        with st.expander("Event registry"):
            st.dataframe(
                data.events.loc[:, ["date", "category", "name", "description"]],
                width="stretch", hide_index=True,
            )

    with tab_study:
        study = cached_halving_study(study_post)
        st.plotly_chart(
            car_chart(study, f"Halvings (n={study.n_events})"), width="stretch"
        )
        st.info(study.summary())
        if study.n_events < 10:
            st.warning(
                f"The confidence interval is wide because the sample holds "
                f"{study.n_events} events. That is not a flaw in the method - it is "
                "all the information there is."
            )

        st.subheader("Event categories")
        studies = cached_category_studies(min(study_post, 180))
        if studies:
            rows = [
                {"category": name, "n": result.n_events, **result.car_summary}
                for name, result in studies.items()
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            choice = st.selectbox("Chart for category", sorted(studies))
            st.plotly_chart(
                car_chart(studies[choice], f"{choice} (n={studies[choice].n_events})"),
                width="stretch",
            )
        st.caption(
            "The p-values in this table are RAW. Use the Validation tab for inference."
        )

    with tab_control:
        st.caption(
            "A halving concerns Bitcoin alone, so what assets without a halving "
            "did over the same window is a placebo test. The difference is computed "
            "pairwise across events so that shared macro conditions cancel out."
        )
        report = cached_control(study_post)
        if "error" in report:
            st.warning(report["error"])
        else:
            for name, comparison in report["comparisons"].items():
                st.subheader(f"BTC vs {name}")
                if comparison.table.empty:
                    st.info("no shared events with a complete window")
                    continue
                st.dataframe(comparison.table, width="stretch")
                verdict = comparison.verdict()
                if "NOT common" in verdict:
                    st.success(verdict)
                else:
                    st.warning(verdict)
                placebo = report["placebos"][name]
                if placebo.car_summary:
                    st.caption(
                        f"Placebo - {name} treated like BTC: "
                        f"CAR({placebo.car_summary['offset']}d) = "
                        f"{placebo.car_summary['car']:+.1%}, "
                        f"p = {placebo.car_summary['p_value']:.3f}"
                    )
                with st.expander(f"CAR of individual events ({name})"):
                    st.dataframe(comparison.per_event, width="stretch")

    with tab_validation:
        scan = cached_scan()
        if scan.empty:
            st.info("No hypotheses to check.")
        else:
            st.subheader("Window scan with multiple-testing correction")
            st.info(summarize(scan))
            st.dataframe(
                scan.loc[
                    :, ["hypothesis", "n_in", "mean_in", "mean_out", "difference",
                        "p_value", "p_adjusted", "significant_adjusted"]
                ],
                width="stretch", hide_index=True,
            )
        out_of_sample = cached_out_of_sample()
        if not out_of_sample.empty:
            st.subheader("Out-of-sample replication (split by cycle)")
            st.dataframe(
                out_of_sample.loc[
                    :, ["hypothesis", "train_effect", "test_effect", "same_sign",
                        "significant_in_train", "significant_out_of_sample",
                        "effect_retained", "replicated"]
                ],
                width="stretch", hide_index=True,
            )
            survivors = out_of_sample[out_of_sample["replicated"]]["hypothesis"].tolist()
            if survivors:
                st.success(f"Survived out of sample: {', '.join(survivors)}")
            else:
                st.warning(
                    "No hypothesis replicated outside the training sample. That is "
                    "the most common and most instructive result in this project."
                )

    with tab_specs:
        st.subheader("Vary the analytical choices, not the hypothesis")
        st.caption(
            "Every hypothesis in the Validation tab inherits the same four "
            "decisions: estimation window, abnormal or raw returns, log or "
            "simple returns, and which exchange history to trust. If one of "
            "them is wrong, all of those results are wrong together. This "
            "re-runs the halving study under every combination."
        )
        curve, summary = load_specification_results(config)
        if curve.empty:
            st.info(
                "No saved curve. Run `run.py speccurve` - the 200 permutations "
                "take about ten minutes, which is too slow to compute here."
            )
        else:
            st.plotly_chart(specification_chart(curve), width="stretch")
            st.caption(
                "Red marks a specification significant at 5% before any "
                "correction. The vertical axis is scaled to the estimates, so "
                "the widest confidence bands run off the chart - a "
                "specification built on two halvings genuinely has an interval "
                "of roughly plus or minus 1000%, and the full numbers are in "
                "the table below."
            )
            columns = st.columns(4)
            usable = curve[curve["car"].notna()]
            columns[0].metric("Specifications", len(usable))
            columns[1].metric("Median CAR", f"{usable['car'].median():+.1%}")
            columns[2].metric(
                "Significant (raw)",
                f"{int(usable['significant'].sum())}/{len(usable)}",
            )
            columns[3].metric("Positive CAR", f"{(usable['car'] > 0).mean():.0%}")

            if not summary.empty:
                row = summary.iloc[0]
                st.warning(
                    "The count of significant specifications is NOT a test. "
                    "These are re-analyses of the same four halvings, so they "
                    "move together and their share is not binomial. Inference "
                    "comes from shifting the four dates as a block to a random "
                    "place in the price history and rebuilding the whole curve."
                )
                st.info(
                    f"Under that null, random placement produced "
                    f"**{row['null_significant_mean']:.1f}** significant "
                    f"specifications on average "
                    f"(95th percentile {row['null_significant_p95']:.0f}), "
                    f"against **{int(row['n_significant'])}** observed. "
                    f"Curve-level p = {row['median_p_value']:.3f} on the median "
                    f"effect and {row['significant_count_p_value']:.3f} on the count."
                )
                # The permutation count belongs on screen, not just in the file.
                # A smoke test writes the same filename as a full run, so
                # without this a 5-draw result would look identical to a
                # 200-draw one.
                st.caption(
                    f"From {int(row['n_permutations'])} permutations, "
                    f"{row['share_wrapped']:.0%} of which wrapped past the end of "
                    "the history. Below about 100 draws these numbers are a "
                    "smoke test, not a result."
                )
            with st.expander("Every specification"):
                st.dataframe(
                    curve.loc[:, ["label", "n_events", "car", "ci_low", "ci_high",
                                  "p_value", "significant"]],
                    width="stretch", hide_index=True,
                )

    with tab_backtest:
        table, curves = cached_backtests()
        if table.empty:
            st.info("No backtest results.")
        else:
            st.plotly_chart(equity_chart(curves), width="stretch")
            st.dataframe(
                table.loc[
                    :, ["total_return", "cagr", "sharpe", "sortino", "max_drawdown",
                        "calmar", "win_rate", "time_in_market", "turnover_annual", "total_cost"]
                ],
                width="stretch",
            )
            baseline = table.loc["buy and hold"]
            better = [
                name for name in table.index
                if name != "buy and hold" and table.loc[name, "sharpe"] > baseline["sharpe"]
            ]
            st.caption(
                "Always compared against buy-and-hold. Better Sharpe: "
                + (", ".join(better) if better else "no strategy")
            )

    with tab_sizing:
        st.subheader("How much to hold, never which way")
        st.caption(
            "Direction is not forecastable here and every other tab says so. "
            "The size of the next move is: volatility clusters, and the model "
            "whose interval the Forecast tab reports is the same one used to "
            "size the position. Hold target volatility divided by forecast "
            "volatility, so the RISK stays constant rather than the quantity."
        )
        comparison, today, volatility = load_sizing_results(config)

        if comparison.empty or volatility.empty:
            st.info(
                "No saved sizing results. Run `run.py sizing` — building the "
                "volatility forecast is a GARCH refit every 30 days over 4000 "
                "days, which is too slow to do here."
            )
        else:
            if not today.empty:
                row = today.iloc[0]
                columns = st.columns(4)
                columns[0].metric("Position to hold", f"{row['position']:.2f}")
                columns[1].metric(
                    "Forecast volatility", f"{row['forecast_annual_volatility']:.0%}",
                    delta=f"{row['forecast_annual_volatility'] - row['median_annual_volatility']:+.0%} vs median",
                    delta_color="off",
                )
                columns[2].metric("Price", f"{row['price']:,.0f}")
                columns[3].metric("As of", str(row["as_of"]))

            band = float(today["band"].iloc[0]) if not today.empty else 0.30
            raw = (0.60 / (volatility * (365 ** 0.5))).clip(0, 1)
            position = apply_rebalance_band(raw, band)
            st.plotly_chart(sizing_chart(volatility, position), width="stretch")
            st.caption(
                f"The blue line is the position actually held, after the {band:.0%} "
                "rebalance band — flat stretches are the band doing its job, not "
                "missing data. It steps down where volatility rises, which is the "
                "entire mechanism; any other shape would mean something is wrong. "
                "Drawn without the band it is a dense scribble that hides exactly "
                "the relationship it is meant to show."
            )

            st.dataframe(comparison, width="stretch")
            st.warning(
                "**This is a risk-control tool, not a way to earn more.** "
                "Volatility falls from 67% to 47% and the worst drawdown from "
                "−83% to −66%, but the Sharpe ratio does not improve and the "
                "permutation test on the best variant gives p = 0.40. Returns "
                "fall about as much as risk does — look at the CAGR column, "
                "52% against 70% for simply holding. A smoother ride is bought "
                "with return, not granted."
            )
            st.caption(
                "Two details worth keeping. Without a rebalance band the rule "
                "retrades 8.9 times a year and pays 24.5% of capital in costs, "
                "destroying the benefit it exists to provide. And an "
                "exponentially weighted standard deviation sizes positions at "
                "least as well as the GARCH fit that takes six minutes — the "
                "cheap model was added specifically so the expensive one had to "
                "earn its cost, and it did not."
            )

    with tab_signals:
        st.caption(
            "What each backtested rule says about right now: the position it "
            "is holding, and the price or date that would flip it. The live "
            "quote comes from Binance and is only used for the distance to "
            "the trigger - every statistic on the panel is from the local "
            "sample. Turn sound on to hear entries and exits during a replay."
        )
        bundle = cached_signals()
        if not bundle["strategies"]:
            st.info("No strategies to show. Download the data first.")
        else:
            render_signal_desk(desk_payload(bundle))
            with st.expander("Trade log, every rule"):
                logs = []
                for item in bundle["strategies"]:
                    signal = item["report"]
                    if signal.trades.empty:
                        continue
                    log = signal.trades.copy()
                    log.insert(0, "strategy", signal.name)
                    logs.append(log)
                if logs:
                    st.dataframe(
                        pd.concat(logs, ignore_index=True),
                        width="stretch", hide_index=True,
                    )
                st.caption(
                    "`return_net` charges both sides of the round trip at the "
                    "configured fees plus slippage. A rule can be right about "
                    "direction and still lose here - that is the point of "
                    "charging it."
                )

    with tab_receipts:
        st.caption(
            "Everything else in this app is a backtest, written by someone who "
            "had already seen the outcome. This page is the exception: claims "
            "recorded before their window closed, scored when it closes. It is "
            "worth nothing on the day it starts and a little more every week."
        )
        ledger = load_ledger(processed_dir(config) / "predictions.csv")

        if ledger.empty:
            st.info(
                "No claims recorded yet. Write the first one with:\n\n"
                "`python run.py ledger --record`"
            )
        else:
            scored = score_ledger(ledger, data.features["close"])
            open_claims = int((~scored["matured"].astype(bool)).sum())
            board = ledger_scoreboard(
                scored, base_rate=outlook.rates[0][1].share_positive
            )
            st.info(ledger_verdict(board, open_claims))
            st.caption(
                "Re-run `python run.py ledger --record` whenever the data is "
                "refreshed. Recording the same day twice replaces the row, so "
                "a daily habit costs nothing and a missed day is just a gap."
            )

            waiting = scored[~scored["matured"].astype(bool)]
            if not waiting.empty:
                st.subheader("Still open")
                st.dataframe(
                    waiting.loc[:, ["as_of", "target_date", "horizon", "claim",
                                    "level", "low", "high", "p_up", "note"]],
                    width="stretch", hide_index=True,
                    column_config=LEDGER_DATE_COLUMNS,
                )
            settled = scored[scored["matured"].astype(bool)]
            if not settled.empty:
                st.subheader("Settled")
                st.dataframe(
                    settled.loc[:, ["as_of", "target_date", "claim", "level",
                                    "price_at_origin", "price_at_target",
                                    "realised", "hit"]],
                    width="stretch", hide_index=True,
                    column_config=LEDGER_DATE_COLUMNS,
                )
                st.subheader("The scoreboard")
                st.dataframe(board, width="stretch", hide_index=True)

    with tab_forecast:
        st.caption(
            "Probability that the forward return is positive - not a price "
            "forecast. The number that matters is whether it beats the "
            "baselines, especially `always_up`: Bitcoin rose in most historical "
            "windows, so the reference point is that base rate, not 50%."
        )
        # Streamlit renders every tab body on every run, so an unguarded call
        # here would make the whole page wait for 13 model fits before showing
        # anything - even for someone who never opens this tab. Hence the
        # explicit gate; after the first run the result is cached.
        if not st.session_state.get("forecast_requested"):
            st.info(
                "Fitting the model across 13 walk-forward folds takes about a "
                "minute. It is not run until you ask for it, so the rest of the "
                "dashboard stays fast."
            )
            if st.button("Run the forecast", icon=":material/play_arrow:"):
                st.session_state["forecast_requested"] = True
                st.rerun()
            report = None
        else:
            report = cached_forecast()

        if report is None:
            pass
        elif "error" in report:
            st.warning(report["error"])
        else:
            run = report["run"]
            verdict_text = run.summary()
            if "NO EDGE" in verdict_text:
                st.error(verdict_text)
            else:
                st.success(verdict_text)

            st.subheader("Pooled, non-overlapping rows")
            st.dataframe(
                run.pooled.loc[
                    :, ["n", "brier", "log_loss", "accuracy", "auc",
                        "mean_probability", "base_rate"]
                ],
                width="stretch",
            )
            st.caption(
                f"Scored on {int(run.pooled.loc['model', 'n'])} non-overlapping "
                f"rows out of {int(run.pooled_all_rows.loc['model', 'n'])} daily "
                f"predictions. A {report['horizon']}-day label on consecutive "
                "days repeats almost all of itself, so the smaller number is "
                "the honest one."
            )

            latest = report["latest"]
            if "error" not in latest:
                left, right = st.columns(2)
                left.metric(
                    f"Next {report['horizon']} days up",
                    f"{latest['probability_up']:.1%}",
                    delta=f"{latest['edge_over_base_rate']:+.1%} vs base rate",
                )
                right.metric("Training base rate", f"{latest['train_base_rate']:.1%}")
                if "NO EDGE" in verdict_text:
                    st.warning(
                        "Read that probability as decoration. The evaluation "
                        "above says the model has no edge out of sample, so it "
                        "is not evidence about the future."
                    )

            with st.expander("Per fold"):
                st.dataframe(
                    run.folds.loc[
                        :, ["fold", "n_train", "alpha", "n", "brier", "accuracy",
                            "auc", "base_rate"]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            with st.expander("Calibration: predicted vs actual"):
                if run.calibration.empty:
                    st.info("Not enough data.")
                else:
                    st.dataframe(run.calibration, width="stretch")
                    st.caption(
                        "A well calibrated model has `mean_predicted` close to "
                        "`share_positive` in every row."
                    )


if __name__ == "__main__":
    main()
