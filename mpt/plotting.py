from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_frontier(points: pd.DataFrame, frontier: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # sample/top points
    df_main = points[points["kind"].isin(["sample", "top"])].copy()
    if not df_main.empty:
        fig.add_trace(go.Scattergl(
            x=df_main["ann_vol"],
            y=df_main["ann_return"],
            mode="markers",
            name="Simulated Portfolios",
            text=df_main["tickers"],
            customdata=df_main[["sharpe", "kind"]].to_numpy(),
            hovertemplate=(
                "<b>Portfolio</b><br>%{text}<br>"
                "Return: %{y:.2%}<br>"
                "Vol: %{x:.2%}<br>"
                "Sharpe: %{customdata[0]:.3f}<br>"
                "<extra></extra>"
            ),
            marker=dict(size=5, opacity=0.6),
        ))

    # frontier line (approx)
    if frontier is not None and not frontier.empty:
        fig.add_trace(go.Scatter(
            x=frontier["ann_vol"],
            y=frontier["ann_return"],
            mode="lines",
            name="Approx. Frontier",
            hoverinfo="skip",
        ))

    # key points (always)
    df_key = points[points["kind"] == "key"].copy()
    if not df_key.empty:
        # label in hover
        label = df_key.get("key_label", "")
        fig.add_trace(go.Scatter(
            x=df_key["ann_vol"],
            y=df_key["ann_return"],
            mode="markers+text",
            name="Key Portfolios",
            text=label,
            textposition="top center",
            hovertext=df_key["tickers"],
            customdata=df_key[["sharpe", "key_label"]].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>%{hovertext}<br>"
                "Return: %{y:.2%}<br>"
                "Vol: %{x:.2%}<br>"
                "Sharpe: %{customdata[0]:.3f}<br>"
                "<extra></extra>"
            ),
            marker=dict(size=10),
        ))

    fig.update_layout(
        title="Efficient Frontier (sampled)",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        legend_title="",
        height=650,
    )
    return fig


def plot_equity_curve(strategy: pd.Series, benchmark: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strategy.index, y=strategy.values, mode="lines", name="Strategy"))
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, mode="lines", name="SPY"))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity (normalized)",
        height=550,
    )
    return fig
