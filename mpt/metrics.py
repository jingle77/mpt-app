from __future__ import annotations

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR


def annualize_return(mean_daily: np.ndarray) -> np.ndarray:
    return mean_daily * TRADING_DAYS_PER_YEAR


def annualize_cov(cov_daily: np.ndarray) -> np.ndarray:
    return cov_daily * TRADING_DAYS_PER_YEAR


def sharpe_ratio(ann_return: np.ndarray, ann_vol: np.ndarray, rf: float) -> np.ndarray:
    ann_vol = np.where(ann_vol <= 0, np.nan, ann_vol)
    return (ann_return - rf) / ann_vol


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def cagr(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    start = equity.index.min()
    end = equity.index.max()
    years = (end - start).days / 365.25
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def ann_vol_from_daily_returns(daily_returns: pd.Series) -> float:
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def ann_return_from_daily_returns(daily_returns: pd.Series) -> float:
    return float(daily_returns.mean() * TRADING_DAYS_PER_YEAR)
