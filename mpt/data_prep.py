from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .fmp_client import FMPClient


@dataclass(frozen=True)
class PricePanel:
    prices: pd.DataFrame          # index=date (datetime), columns=symbol, values=adjClose
    symbols: List[str]
    start_date: pd.Timestamp
    end_date: pd.Timestamp


def _download_one(client: FMPClient, symbol: str) -> Tuple[str, pd.Series]:
    df = client.historical_price_eod_full(symbol)
    if df.empty:
        return symbol, pd.Series(dtype="float64")
    s = df.set_index("date")["adjClose"].astype("float64")
    s.name = symbol
    return symbol, s


def build_price_panel_on_calendar(
    client: FMPClient,
    symbols: List[str],
    calendar: pd.DatetimeIndex,
    max_workers: int = 24,
    progress_cb=None,
) -> PricePanel:
    """Download adjClose for all symbols and align to a shared trading-day calendar.

    Keeps NaNs for missing symbol history and filters eligibility dynamically later
    (by user-selected lookback/holding windows). This avoids collapsing the date
    range like a strict intersection across all symbols.
    """

    series_list: List[pd.Series] = []
    total = len(symbols)
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_download_one, client, sym): sym for sym in symbols}
        for fut in as_completed(futs):
            try:
                sym2, s = fut.result()
                if not s.empty:
                    series_list.append(s.reindex(calendar).rename(sym2))
            except Exception:
                pass
            done += 1
            if progress_cb:
                progress_cb(done, total)

    if not series_list:
        raise RuntimeError("No price data returned. Check API key and endpoint access.")

    panel = pd.concat(series_list, axis=1)
    panel = panel.loc[calendar].sort_index()
    panel = panel.dropna(axis=1, how="all")
    symbols_kept = list(panel.columns)

    return PricePanel(
        prices=panel,
        symbols=symbols_kept,
        start_date=panel.index.min(),
        end_date=panel.index.max(),
    )
