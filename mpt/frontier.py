from __future__ import annotations

import pandas as pd


def approximate_efficient_frontier(points: pd.DataFrame) -> pd.DataFrame:
    """Approximate frontier from sampled points: sort by vol and keep cumulative max return."""
    df = points.copy()
    df = df.dropna(subset=["ann_vol", "ann_return"])
    df = df.sort_values("ann_vol").reset_index(drop=True)
    # cumulative max return
    df["cummax_ret"] = df["ann_return"].cummax()
    frontier = df[df["ann_return"] >= df["cummax_ret"] - 1e-12].copy()
    return frontier.drop(columns=["cummax_ret"])
