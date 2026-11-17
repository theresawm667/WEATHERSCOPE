"""Chart builders. Each function returns a Plotly figure; app.py displays it.

Keeping charts here (not inside app.py) means you can change a chart's look in
one place without touching the page logic.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = px.colors.qualitative.Safe
ACCENT = "#0E7C86"      # teal
WARM = "#E4572E"        # orange-red
_LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")


def _style(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(title=title, height=height, **_LAYOUT)
    return fig


def line_chart(df: pd.DataFrame, x: str, y: str, color: str, title: str, y_label: str) -> go.Figure:
    fig = px.line(df, x=x, y=y, color=color, color_discrete_sequence=PALETTE, labels={y: y_label, x: ""})
    return _style(fig, title)


def histogram(df: pd.DataFrame, column: str, color: str, title: str, x_label: str, log_y: bool = False) -> go.Figure:
    fig = px.histogram(df, x=column, color=color, nbins=50, barmode="overlay", opacity=0.65,
                       color_discrete_sequence=PALETTE, labels={column: x_label}, log_y=log_y)
    return _style(fig, title)


def month_lines(profile: pd.DataFrame, column: str, title: str, y_label: str) -> go.Figure:
    fig = px.line(profile, x="month_name", y=column, color="city", markers=True,
                  color_discrete_sequence=PALETTE, labels={column: y_label, "month_name": ""})
    return _style(fig, title)


def heatmap(matrix: pd.DataFrame, title: str, unit: str, colorscale: str = "YlOrRd") -> go.Figure:
    fig = px.imshow(matrix, text_auto=".0f", aspect="auto", color_continuous_scale=colorscale,
                    labels=dict(color=unit, x="", y=""))
    return _style(fig, title, height=380)


def season_bars(seasons: pd.DataFrame, value: str, title: str, y_label: str) -> go.Figure:
    fig = px.bar(seasons, x="season", y=value, color="city", barmode="group",
                 color_discrete_sequence=PALETTE, labels={value: y_label, "season": ""})
    return _style(fig, title)


def bar(df: pd.DataFrame, x: str, y: str, title: str, y_label: str, color: str | None = None,
        horizontal: bool = False) -> go.Figure:
    if horizontal:
        fig = px.bar(df, x=y, y=x, orientation="h", color_discrete_sequence=[ACCENT],
                     labels={y: y_label, x: ""})
        fig.update_yaxes(autorange="reversed")
    else:
        fig = px.bar(df, x=x, y=y, color=color, color_discrete_sequence=[ACCENT] if color is None else PALETTE,
                     labels={y: y_label, x: ""})
    return _style(fig, title)


def model_comparison(table: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """Bar chart of one metric across baselines and models (lower = better)."""
    fig = px.bar(table, x="name", y=metric, color="kind",
                 color_discrete_map={"Baseline": "#9AA5B1", "Model": ACCENT},
                 labels={metric: metric.upper() + " (°C)", "name": ""}, text_auto=".2f")
    return _style(fig, title, height=380)


def actual_vs_predicted(actual, predicted, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=actual, y=predicted, mode="markers",
                               marker=dict(size=4, opacity=0.35, color=ACCENT), name="days"))
    lo, hi = float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color=WARM, dash="dash"), name="perfect prediction"))
    fig.update_xaxes(title="Actual maximum temperature (°C)")
    fig.update_yaxes(title="Predicted (°C)")
    return _style(fig, title, height=450)


def residual_histogram(residuals, title: str) -> go.Figure:
    fig = px.histogram(x=residuals, nbins=60, color_discrete_sequence=[ACCENT],
                       labels={"x": "Error = actual − predicted (°C)"})
    fig.add_vline(x=0, line_dash="dash", line_color=WARM)
    return _style(fig, title, height=380)


def forecast_context(history: pd.DataFrame, forecast_date, forecast: float, actual: float | None) -> go.Figure:
    """Recent maxima as a line, with tomorrow's forecast (and actual) as markers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date"], y=history["tmax"], mode="lines+markers",
                             line=dict(color=ACCENT), name="Observed daily max"))
    fig.add_trace(go.Scatter(x=[forecast_date], y=[forecast], mode="markers",
                             marker=dict(size=14, color=WARM, symbol="diamond"), name="Forecast"))
    if actual is not None:
        fig.add_trace(go.Scatter(x=[forecast_date], y=[actual], mode="markers",
                                 marker=dict(size=12, color="#222", symbol="circle-open", line=dict(width=2)),
                                 name="Actual"))
    fig.update_yaxes(title="Max temperature (°C)")
    return _style(fig, "Recent daily maxima and the forecast", height=360)
