"""Plotly figures for the dashboard. Input: plain pandas objects. Output: go.Figure.

No Streamlit and no finance maths in here: every number plotted is computed
in metrics/contributions/optimize and passed in.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Fixed colour roles (categorical slots in a fixed order, never cycled).
PORTFOLIO = "#2a78d6"  # blue
BENCHMARK = "#eb6834"  # orange
ACCENT = "#1baf7a"  # aqua (min-variance marker)
VIOLET = "#6250d6"  # parametric VaR line / max-Sharpe marker
MUTED = "#898781"  # individual assets, secondary marks
# Diverging scale for correlations: blue (-1) -> neutral grey (0) -> red (+1).
# Red = highly correlated = little diversification benefit.
CORR_SCALE = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]


def _layout(fig: go.Figure, *, y_title: str, y_format: str = ".0%", height: int = 380) -> None:
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode="x unified",
    )
    fig.update_yaxes(title=y_title, tickformat=y_format, zeroline=False)
    fig.update_xaxes(showgrid=False)


def cumulative_performance(portfolio: pd.Series, benchmark: pd.Series | None) -> go.Figure:
    """Growth of 100 invested at the start, portfolio vs benchmark."""
    fig = go.Figure()
    for series, name, color in (
        (portfolio, "Portfolio", PORTFOLIO),
        (benchmark, "Benchmark", BENCHMARK),
    ):
        if series is None:
            continue
        value = 100 * (1 + series).cumprod()
        fig.add_scatter(
            x=value.index,
            y=value,
            name=name,
            line={"color": color, "width": 2},
            hovertemplate="%{y:.1f}",
        )
    _layout(fig, y_title="Value of 100 invested", y_format=",.0f")
    return fig


def drawdown_chart(portfolio_dd: pd.Series, benchmark_dd: pd.Series | None) -> go.Figure:
    """Percentage below the previous high ("underwater" chart)."""
    fig = go.Figure()
    fig.add_scatter(
        x=portfolio_dd.index,
        y=portfolio_dd,
        name="Portfolio",
        fill="tozeroy",
        fillcolor="rgba(42, 120, 214, 0.25)",
        line={"color": PORTFOLIO, "width": 2},
        hovertemplate="%{y:.1%}",
    )
    if benchmark_dd is not None:
        fig.add_scatter(
            x=benchmark_dd.index,
            y=benchmark_dd,
            name="Benchmark",
            line={"color": BENCHMARK, "width": 2},
            hovertemplate="%{y:.1%}",
        )
    _layout(fig, y_title="Drawdown")
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Pairwise correlation of daily returns, annotated with the values."""
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1,
            zmax=1,
            colorscale=CORR_SCALE,
            texttemplate="%{z:.2f}",
            textfont={"color": "#0b0b0b"},
            hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
            colorbar={"title": "ρ", "thickness": 12},
        )
    )
    fig.update_layout(
        height=max(320, 60 * len(corr) + 80),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis={"autorange": "reversed"},
    )
    return fig


def weight_vs_risk_chart(contributions: pd.DataFrame) -> go.Figure:
    """Each position's weight next to its share of portfolio volatility."""
    df = contributions.sort_values("risk_share")
    fig = go.Figure()
    fig.add_bar(
        y=df.index,
        x=df["weight"],
        name="Weight",
        orientation="h",
        marker_color=PORTFOLIO,
        hovertemplate="Weight: %{x:.1%}<extra>%{y}</extra>",
    )
    fig.add_bar(
        y=df.index,
        x=df["risk_share"],
        name="Share of portfolio risk",
        orientation="h",
        marker_color=BENCHMARK,
        hovertemplate="Risk share: %{x:.1%}<extra>%{y}</extra>",
    )
    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        height=max(300, 55 * len(df) + 80),
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(tickformat=".0%")
    return fig


def return_contribution_chart(contributions: pd.DataFrame) -> go.Figure:
    """Contribution of each position to the annualised (arithmetic) mean return."""
    df = contributions.sort_values("return_contribution")
    fig = go.Figure(
        go.Bar(
            y=df.index,
            x=df["return_contribution"],
            orientation="h",
            marker_color=[PORTFOLIO if v >= 0 else "#e34948" for v in df["return_contribution"]],
            text=[f"{v:+.2%}" for v in df["return_contribution"]],
            textposition="outside",
            cliponaxis=False,  # keep the value label of the longest bar visible
            hovertemplate="%{x:.2%} p.a.<extra>%{y}</extra>",
        )
    )
    fig.update_layout(
        height=max(300, 45 * len(df) + 80),
        margin={"l": 10, "r": 70, "t": 10, "b": 10},
        showlegend=False,
    )
    fig.update_xaxes(tickformat=".1%", title="Contribution to annual mean return")
    return fig


def return_histogram(
    returns: pd.Series, historical_var: float, parametric_var: float, confidence: float
) -> go.Figure:
    """Distribution of daily portfolio returns with both 1-day VaR thresholds marked."""
    fig = go.Figure(
        go.Histogram(
            x=returns,
            nbinsx=80,
            marker_color=PORTFOLIO,
            opacity=0.85,
            name="Daily returns",
            hovertemplate="%{x:.2%}: %{y} days<extra></extra>",
        )
    )
    # The two thresholds are often close together: put the label of the more
    # negative one on its left and the other on its right so they never overlap.
    lines = sorted(
        [
            (historical_var, "Historical VaR", BENCHMARK, "dash"),
            (parametric_var, "Normal VaR", VIOLET, "dot"),
        ],
        key=lambda item: item[0],
        reverse=True,
    )
    for (value, label, color, dash), side in zip(lines, ("top left", "top right"), strict=True):
        fig.add_vline(
            x=-value,
            line={"color": color, "width": 2, "dash": dash},
            annotation_text=f"{label} {confidence:.1%}: {-value:.2%}",
            annotation_position=side,
            annotation_font_size=11,
        )
    fig.update_layout(
        height=340,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        showlegend=False,
        bargap=0.05,
    )
    fig.update_xaxes(tickformat=".1%", title="Daily return")
    fig.update_yaxes(title="Number of days")
    return fig


def frontier_chart(
    frontier: pd.DataFrame,
    assets: pd.DataFrame,
    points: dict[str, tuple[float, float]],
) -> go.Figure:
    """Efficient frontier with single assets and labelled special portfolios.

    `assets` has columns "return" and "volatility" indexed by ticker;
    `points` maps a label (e.g. "Max Sharpe") to (volatility, return).
    """
    fig = go.Figure()
    fig.add_scatter(
        x=frontier["volatility"],
        y=frontier["return"],
        mode="lines",
        name="Efficient frontier",
        line={"color": PORTFOLIO, "width": 2},
        hovertemplate="Vol %{x:.1%} | Return %{y:.1%}<extra>Frontier</extra>",
    )
    fig.add_scatter(
        x=assets["volatility"],
        y=assets["return"],
        mode="markers+text",
        text=list(assets.index),
        textposition="top center",
        name="Single assets",
        marker={"color": MUTED, "size": 9},
        hovertemplate="%{text}: vol %{x:.1%}, return %{y:.1%}<extra></extra>",
    )
    styles = {
        "Min variance": (ACCENT, "diamond"),
        "Max Sharpe": (VIOLET, "star"),
        "Your portfolio": (BENCHMARK, "circle"),
    }
    for label, (vol, ret) in points.items():
        color, symbol = styles.get(label, (MUTED, "circle"))
        fig.add_scatter(
            x=[vol],
            y=[ret],
            mode="markers+text",
            text=[label],
            textposition="bottom right",
            name=label,
            marker={
                "color": color,
                "size": 14,
                "symbol": symbol,
                "line": {"width": 2, "color": "white"},
            },
            hovertemplate=f"{label}: vol %{{x:.1%}}, return %{{y:.1%}}<extra></extra>",
        )
    fig.update_layout(
        height=460,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode="closest",
    )
    fig.update_xaxes(title="Annualised volatility", tickformat=".0%")
    fig.update_yaxes(title="Expected annual return (historical mean)", tickformat=".0%")
    return fig
