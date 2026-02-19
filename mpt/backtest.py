from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from .metrics import annualize_cov, annualize_return
from .optimizer import best_portfolio_only


@dataclass
class BacktestResult:
    equity_strategy: pd.Series
    equity_spy: pd.Series
    windows: pd.DataFrame  # per-window log


def walk_forward_backtest(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    lookback_days: int,
    hold_days: int,
    trials: int,
    k: int,
    rf: float,
    batch_size: int,
    seed: Optional[int],
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> BacktestResult:
    """Walk-forward backtest stepping by hold_days.

    Eligibility is computed per window: a symbol must have complete price data
    for the combined lookback+hold window (so both selection and realized returns work).
    """

    spy_prices = spy_prices.sort_index()
    common_idx = prices.index.intersection(spy_prices.index)
    prices = prices.loc[common_idx].sort_index()
    spy_prices = spy_prices.loc[common_idx].sort_index()

    n = len(prices)
    if n < lookback_days + hold_days + 2:
        raise ValueError("Not enough history for the chosen lookback + holding period.")

    total_windows = max(((n - (lookback_days + 1)) // hold_days), 0)

    eq_s = [1.0]
    eq_b = [1.0]
    eq_dates = []
    rows = []

    win = 0
    start = lookback_days  # first day in holding window (index position)

    while start + hold_days < n:
        lb_start = start - lookback_days
        end_inclusive = start + hold_days

        window_prices = prices.iloc[lb_start:end_inclusive + 1]
        eligible_mask = window_prices.notna().all(axis=0)
        eligible = window_prices.columns[eligible_mask].tolist()

        if len(eligible) < k:
            rows.append({
                "window": win + 1,
                "status": "skipped_insufficient_eligible",
                "eligible": len(eligible),
                "required_k": k,
                "lookback_start": prices.index[lb_start].date(),
                "lookback_end": prices.index[start - 1].date(),
                "hold_start": prices.index[start].date(),
                "hold_end": prices.index[end_inclusive].date(),
            })
            start += hold_days
            win += 1
            if progress_cb:
                progress_cb(win, total_windows)
            continue

        lb_prices = prices.iloc[lb_start:start + 1][eligible]
        lb_rets = lb_prices.pct_change().iloc[1:]

        mu = lb_rets.mean(axis=0).to_numpy()
        cov = lb_rets.cov().to_numpy()
        mu_ann = annualize_return(mu)
        cov_ann = annualize_cov(cov)

        wseed = None if seed is None else int(seed + win)

        best = best_portfolio_only(
            mu_ann=mu_ann,
            cov_ann=cov_ann,
            tickers=eligible,
            k=k,
            trials=trials,
            rf=rf,
            batch_size=batch_size,
            seed=wseed,
            progress_cb=None,
        )

        hold_prices = prices.iloc[start:end_inclusive + 1][list(best.tickers)]
        hold_rets = hold_prices.pct_change().iloc[1:]
        strat_daily = hold_rets.mean(axis=1)
        strat_hold = float((1.0 + strat_daily).prod() - 1.0)

        spy_hold_prices = spy_prices.iloc[start:end_inclusive + 1]
        spy_hold_rets = spy_hold_prices.pct_change().iloc[1:]
        spy_hold = float((1.0 + spy_hold_rets).prod() - 1.0)

        eq_s.append(eq_s[-1] * (1.0 + strat_hold))
        eq_b.append(eq_b[-1] * (1.0 + spy_hold))
        eq_dates.append(prices.index[end_inclusive])

        rows.append({
            "window": win + 1,
            "status": "ok",
            "eligible": len(eligible),
            "required_k": k,
            "lookback_start": prices.index[lb_start].date(),
            "lookback_end": prices.index[start - 1].date(),
            "hold_start": prices.index[start].date(),
            "hold_end": prices.index[end_inclusive].date(),
            "best_sharpe": best.sharpe,
            "lookback_ann_return": best.ann_return,
            "lookback_ann_vol": best.ann_vol,
            "hold_return": strat_hold,
            "spy_hold_return": spy_hold,
            "tickers": ", ".join(best.tickers),
        })

        start += hold_days
        win += 1
        if progress_cb:
            progress_cb(win, total_windows)

    equity_strategy = pd.Series(eq_s[1:], index=pd.to_datetime(eq_dates), name="strategy")
    equity_spy = pd.Series(eq_b[1:], index=pd.to_datetime(eq_dates), name="spy")
    windows = pd.DataFrame(rows)

    return BacktestResult(equity_strategy=equity_strategy, equity_spy=equity_spy, windows=windows)
