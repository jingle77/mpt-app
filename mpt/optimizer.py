from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import sharpe_ratio


@dataclass(frozen=True)
class PortfolioPoint:
    tickers: Tuple[str, ...]
    ann_return: float
    ann_vol: float
    sharpe: float
    kind: str  # sample | key | top


@dataclass
class SimulationResult:
    best_sharpe: PortfolioPoint
    key_points: Dict[str, PortfolioPoint]
    plot_points: pd.DataFrame  # columns: ann_return, ann_vol, sharpe, tickers, kind
    meta: Dict[str, float]


def _compute_batch_metrics(
    mu_ann: np.ndarray,
    cov_ann: np.ndarray,
    idx: np.ndarray,  # (batch, k)
    rf: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = idx.shape[1]
    ann_ret = mu_ann[idx].mean(axis=1)

    # cov_sub: (batch, k, k) then sum -> variance * (1/k^2)
    cov_sub = cov_ann[idx[:, :, None], idx[:, None, :]]
    ann_var = cov_sub.sum(axis=(1, 2)) / (k * k)
    ann_vol = np.sqrt(np.maximum(ann_var, 0.0))
    sr = sharpe_ratio(ann_ret, ann_vol, rf)
    return ann_ret, ann_vol, sr


def simulate_random_equal_weight(
    mu_ann: np.ndarray,
    cov_ann: np.ndarray,
    tickers: List[str],
    k: int,
    trials: int,
    rf: float,
    batch_size: int,
    seed: Optional[int],
    plot_sample_size: int,
    top_sharpe_to_plot: int,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> SimulationResult:
    """Randomly sample equal-weight k-stock portfolios and return best-found + plot dataset.

    Includes (and guarantees) key portfolios:
      - min_vol, max_vol, min_ret, max_ret, max_sharpe
    """

    n = len(tickers)
    if k < 2 or k > n:
        raise ValueError(f"k must be in [2, {n}]")

    trials = int(trials)
    rng = np.random.default_rng(seed)

    # Pre-select which trials to keep for plotting (fast, exact size, no Python per-trial loop)
    keep_n = int(min(plot_sample_size, trials))
    keep_positions = None
    if keep_n > 0:
        keep_positions = np.sort(rng.choice(trials, size=keep_n, replace=False))
        keep_ptr = 0
        plot_rows = []
    else:
        keep_ptr = 0
        plot_rows = []

    # Track top sharpe (for plotting only)
    top_rows = []

    # Track key points
    best_sharpe = None
    min_vol = None
    max_vol = None
    min_ret = None
    max_ret = None
    max_sr = None

    def to_point(idx_row, ann_ret, ann_vol, sr, kind):
        t = tuple(sorted([tickers[i] for i in idx_row]))
        return PortfolioPoint(tickers=t, ann_return=float(ann_ret), ann_vol=float(ann_vol), sharpe=float(sr), kind=kind)

    processed = 0
    while processed < trials:
        b = min(batch_size, trials - processed)

        # unique k picks per trial via random matrix + argpartition
        rand = rng.random((b, n), dtype=np.float64)
        idx = np.argpartition(rand, kth=k-1, axis=1)[:, :k]

        ann_ret, ann_vol, sr = _compute_batch_metrics(mu_ann, cov_ann, idx, rf)

        # Update key points (vectorized reductions)
        # Note: for ties, first occurrence wins (fine)
        loc_min_vol = int(np.nanargmin(ann_vol))
        loc_max_vol = int(np.nanargmax(ann_vol))
        loc_min_ret = int(np.nanargmin(ann_ret))
        loc_max_ret = int(np.nanargmax(ann_ret))
        loc_max_sr = int(np.nanargmax(sr))

        cand = [
            ("min_vol", loc_min_vol),
            ("max_vol", loc_max_vol),
            ("min_ret", loc_min_ret),
            ("max_ret", loc_max_ret),
            ("max_sharpe", loc_max_sr),
        ]

        for name, j in cand:
            p = to_point(idx[j], ann_ret[j], ann_vol[j], sr[j], kind="key")
            if name == "min_vol":
                if (min_vol is None) or (p.ann_vol < min_vol.ann_vol):
                    min_vol = p
            elif name == "max_vol":
                if (max_vol is None) or (p.ann_vol > max_vol.ann_vol):
                    max_vol = p
            elif name == "min_ret":
                if (min_ret is None) or (p.ann_return < min_ret.ann_return):
                    min_ret = p
            elif name == "max_ret":
                if (max_ret is None) or (p.ann_return > max_ret.ann_return):
                    max_ret = p
            elif name == "max_sharpe":
                if (max_sr is None) or (p.sharpe > max_sr.sharpe):
                    max_sr = p

        # best_sharpe = max_sr
        best_sharpe = max_sr

        # Keep top sharpe rows for plotting (merge-prune each batch)
        if top_sharpe_to_plot > 0:
            local_top_n = min(top_sharpe_to_plot, b)
            top_idx = np.argpartition(sr, kth=local_top_n-1)[-local_top_n:]
            for j in top_idx:
                top_rows.append({
                    "ann_return": float(ann_ret[j]),
                    "ann_vol": float(ann_vol[j]),
                    "sharpe": float(sr[j]),
                    "tickers": ", ".join(sorted([tickers[i] for i in idx[j]])),
                    "kind": "top",
                })
            if len(top_rows) > 2 * top_sharpe_to_plot:
                # prune
                top_rows.sort(key=lambda r: r["sharpe"], reverse=True)
                top_rows = top_rows[:top_sharpe_to_plot]

        # Keep plotting sample positions for this batch
        if keep_positions is not None and keep_ptr < keep_n:
            start = processed
            end = processed + b
            # find positions in [start, end)
            # keep_positions[keep_ptr:keep_ptr+m] lie in this batch
            # use searchsorted to get slice bounds
            import numpy as _np
            left = _np.searchsorted(keep_positions, start, side="left", sorter=None)
            right = _np.searchsorted(keep_positions, end, side="left", sorter=None)
            # advance keep_ptr to right
            if right > left:
                pos = keep_positions[left:right] - start
                for j in pos:
                    plot_rows.append({
                        "ann_return": float(ann_ret[j]),
                        "ann_vol": float(ann_vol[j]),
                        "sharpe": float(sr[j]),
                        "tickers": ", ".join(sorted([tickers[i] for i in idx[j]])),
                        "kind": "sample",
                    })
            keep_ptr = right

        processed += b
        if progress_cb:
            progress_cb(processed, trials)

    key_points = {
        "min_vol": min_vol,
        "max_vol": max_vol,
        "min_ret": min_ret,
        "max_ret": max_ret,
        "max_sharpe": max_sr,
    }

    # Build plot dataframe: sample + top + guaranteed key points (dedup)
    rows = []
    rows.extend(plot_rows)
    if top_rows:
        # final prune
        top_rows.sort(key=lambda r: r["sharpe"], reverse=True)
        rows.extend(top_rows[:top_sharpe_to_plot])

    # Always include key points
    for name, p in key_points.items():
        rows.append({
            "ann_return": p.ann_return,
            "ann_vol": p.ann_vol,
            "sharpe": p.sharpe,
            "tickers": ", ".join(p.tickers),
            "kind": "key",
            "key_label": name,
        })

    df = pd.DataFrame(rows)
    if "key_label" not in df.columns:
        df["key_label"] = ""

    # Deduplicate by tickers+kind, but keep keys
    df = df.drop_duplicates(subset=["tickers", "kind", "key_label"], keep="first").reset_index(drop=True)

    meta = {
        "trials": float(trials),
        "universe_size": float(n),
        "k": float(k),
        "rf": float(rf),
        "plot_points": float(len(df)),
    }
    return SimulationResult(best_sharpe=best_sharpe, key_points=key_points, plot_points=df, meta=meta)


def best_portfolio_only(
    mu_ann: np.ndarray,
    cov_ann: np.ndarray,
    tickers: List[str],
    k: int,
    trials: int,
    rf: float,
    batch_size: int,
    seed: Optional[int],
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> PortfolioPoint:
    """Fast walk-forward helper: returns only best Sharpe portfolio."""
    n = len(tickers)
    rng = np.random.default_rng(seed)
    best = None

    processed = 0
    while processed < trials:
        b = min(batch_size, trials - processed)
        rand = rng.random((b, n), dtype=np.float64)
        idx = np.argpartition(rand, kth=k-1, axis=1)[:, :k]
        ann_ret, ann_vol, sr = _compute_batch_metrics(mu_ann, cov_ann, idx, rf)

        j = int(np.nanargmax(sr))
        if best is None or float(sr[j]) > best.sharpe:
            t = tuple(sorted([tickers[i] for i in idx[j]]))
            best = PortfolioPoint(
                tickers=t,
                ann_return=float(ann_ret[j]),
                ann_vol=float(ann_vol[j]),
                sharpe=float(sr[j]),
                kind="best",
            )

        processed += b
        if progress_cb:
            progress_cb(processed, trials)

    return best
