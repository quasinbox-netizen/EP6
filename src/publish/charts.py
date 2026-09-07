"""Figures shared by the dashboard and the published site.

They live here rather than in dashboard/app.py because both surfaces draw
them, and a second copy would drift: the site would keep a log radial axis the
dashboard had already fixed, or the other way round. Nothing here computes a
result - every number arrives already computed.
"""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

COLORS = {
    "price": "#e8a33d",
    "car": "#4c8bf5",
    "band": "rgba(76, 139, 245, 0.18)",
    "halving": "rgba(232, 163, 61, 0.14)",
    "zero": "#8a8a8a",
}


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
        # Four laps of daily points is ~5,800 coordinates carried into every
        # page that draws this. On a dial a few hundred per lap is the same
        # picture at a third of the weight.
        if len(series) > 400:
            series = series.iloc[:: max(1, len(series) // 400)]
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


def car_chart_from_frame(frame: pd.DataFrame, title: str) -> go.Figure:
    """The event study as saved by `run.py study`, with its interval.

    Takes the CSV rather than the result object: the site reads what the CLI
    left behind, so publishing cannot quietly recompute a study with different
    settings from the one the terminal reported.
    """
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=list(frame["offset_days"]) + list(frame["offset_days"])[::-1],
        y=list(frame["car_ci_high"]) + list(frame["car_ci_low"])[::-1],
        fill="toself", fillcolor=COLORS["band"], line={"width": 0},
        hoverinfo="skip", name="95% confidence interval",
    ))
    figure.add_trace(go.Scatter(
        x=frame["offset_days"], y=frame["car"], mode="lines",
        line={"color": COLORS["car"], "width": 2},
        name="cumulative abnormal return",
    ))
    figure.add_hline(y=0, line_dash="dash", line_color=COLORS["zero"])
    figure.update_layout(
        title=title, height=380, hovermode="x unified",
        xaxis_title="days after the halving", yaxis_title="cumulative abnormal return",
        yaxis_tickformat=".0%", margin={"l": 60, "r": 20, "t": 46, "b": 46},
        legend={"orientation": "h", "y": -0.22},
    )
    return figure


def band_sweep_chart(frame: pd.DataFrame, adopted: float,
                     frozen_from: float | None = None) -> go.Figure:
    """Sharpe against every value of the rebalance band.

    The shape is the argument. A parameter worth ranking would draw a hill with
    the adopted value somewhere near its top; this draws a jagged line on which
    the adopted value is unremarkable, and whose highest points sit against the
    region where the band is so wide the rule stops trading altogether. That
    region is shaded rather than cropped, because it explains why the scores
    rise towards the right - not skill, but a strategy switching itself off.
    """
    figure = go.Figure()

    if frozen_from is not None:
        figure.add_vrect(
            x0=frozen_from, x1=float(frame["band"].max()),
            fillcolor="rgba(239, 64, 86, 0.10)", line_width=0, layer="below",
            annotation_text="the rule stops trading here",
            annotation_position="top left",
            annotation_font={"size": 11, "color": "#ef4056"},
        )

    figure.add_trace(go.Scatter(
        x=frame["band"], y=frame["sharpe"], mode="lines+markers",
        name="Sharpe", line={"color": COLORS["car"], "width": 1.6},
        marker={"size": 4},
        hovertemplate="band %{x:.0%}<br>Sharpe %{y:.4f}<extra></extra>",
    ))

    row = frame.loc[(frame["band"] - adopted).abs().idxmin()]
    figure.add_trace(go.Scatter(
        x=[row["band"]], y=[row["sharpe"]], mode="markers+text",
        name="the band in use", text=["the band in use"], textposition="bottom center",
        textfont={"color": "#f7931a", "size": 11},
        marker={"symbol": "circle-open", "size": 15, "color": "#f7931a",
                "line": {"width": 2.5}},
        hovertemplate="in use: band %{x:.0%}<br>Sharpe %{y:.4f}<extra></extra>",
    ))

    figure.update_layout(
        height=320, margin={"l": 56, "r": 16, "t": 28, "b": 44},
        showlegend=False, xaxis_title="rebalance band", yaxis_title="Sharpe ratio",
        xaxis={"tickformat": ".0%"},
    )
    return figure


def target_sweep_chart(frame: pd.DataFrame, adopted: float,
                       pinned_from: float | None = None) -> go.Figure:
    """Sharpe and drawdown against every volatility target, on two axes.

    Two series rather than one, because the argument is a comparison between
    them. The Sharpe line wanders inside a tenth of a point across a grid that
    takes the position from a twelfth of the portfolio to nearly all of it; the
    drawdown line falls from -18% to -83% over the same range. One of those is
    what the target chooses and the other is what it was chosen on.

    The shaded region is where the position is pinned at the no-borrowing cap
    on almost every day - past it the rule is buy-and-hold, and the drawdown
    line flattening onto buy-and-hold's own drawdown is what shows it.
    """
    figure = go.Figure()

    if pinned_from is not None:
        figure.add_vrect(
            x0=pinned_from, x1=float(frame["target"].max()),
            fillcolor="rgba(239, 64, 86, 0.10)", line_width=0, layer="below",
            annotation_text="the rule is buy-and-hold here",
            annotation_position="top left",
            annotation_font={"size": 11, "color": "#ef4056"},
        )

    figure.add_trace(go.Scatter(
        x=frame["target"], y=frame["sharpe"], mode="lines+markers", name="Sharpe",
        line={"color": COLORS["car"], "width": 1.6}, marker={"size": 4},
        hovertemplate="target %{x:.0%}<br>Sharpe %{y:.4f}<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=frame["target"], y=frame["max_drawdown"], mode="lines",
        name="worst drawdown", yaxis="y2",
        line={"color": "#ef4056", "width": 1.8, "dash": "dot"},
        hovertemplate="target %{x:.0%}<br>drawdown %{y:.1%}<extra></extra>",
    ))

    row = frame.loc[(frame["target"] - adopted).abs().idxmin()]
    figure.add_trace(go.Scatter(
        x=[row["target"]], y=[row["sharpe"]], mode="markers+text",
        showlegend=False, text=["the target in use"], textposition="top center",
        textfont={"color": "#f7931a", "size": 11},
        marker={"symbol": "circle-open", "size": 15, "color": "#f7931a",
                "line": {"width": 2.5}},
        hovertemplate="in use: target %{x:.0%}<br>Sharpe %{y:.4f}<extra></extra>",
    ))

    figure.update_layout(
        height=340, margin={"l": 56, "r": 62, "t": 28, "b": 44},
        xaxis_title="volatility target", xaxis={"tickformat": ".0%"},
        yaxis={"title": "Sharpe ratio"},
        yaxis2={"title": "worst drawdown", "overlaying": "y", "side": "right",
                "tickformat": ".0%", "showgrid": False},
        legend={"orientation": "h", "y": -0.22},
    )
    return figure
