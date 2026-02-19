from __future__ import annotations

import pandas as pd


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(int(x), hi))


def format_tickers(tickers: str, max_len: int = 120) -> str:
    if len(tickers) <= max_len:
        return tickers
    return tickers[: max_len - 3] + "..."
