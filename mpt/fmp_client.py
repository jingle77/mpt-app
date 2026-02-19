from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests
import pandas as pd

from .rate_limiter import SlidingWindowRateLimiter


@dataclass(frozen=True)
class FMPConfig:
    base_url: str
    api_key: str
    timeout_sec: int = 30
    max_retries: int = 3
    max_calls_per_min: int = 740


class FMPClient:
    def __init__(self, cfg: FMPConfig):
        if not cfg.api_key:
            raise ValueError("FMP API key is empty. Set FMP_API_KEY in your environment (.env).")
        self.cfg = cfg
        self.session = requests.Session()
        self.limiter = SlidingWindowRateLimiter(cfg.max_calls_per_min, 60.0)

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self.cfg.base_url.rstrip("/") + "/" + path.lstrip("/")
        params = dict(params or {})
        params["apikey"] = self.cfg.api_key

        last_err = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                self.limiter.acquire()
                r = self.session.get(url, params=params, timeout=self.cfg.timeout_sec)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                # backoff (small)
                time.sleep(min(0.5 * attempt, 2.0))
        raise RuntimeError(f"GET failed after {self.cfg.max_retries} retries: {url}") from last_err

    def sp500_constituents(self) -> List[str]:
        data = self._get("stable/sp500-constituent")
        # Expected: list[dict] with key "symbol"
        symbols = []
        if isinstance(data, list):
            for row in data:
                sym = row.get("symbol")
                if sym:
                    symbols.append(sym.strip().upper())
        return sorted(set(symbols))

    def historical_price_eod_full(self, symbol: str) -> pd.DataFrame:
        data = self._get("stable/historical-price-eod/full", params={"symbol": symbol})
        # Expected stable format: {"symbol": "...", "historical": [ {date, adjClose, ...}, ... ] }
        hist = None
        if isinstance(data, dict):
            hist = data.get("historical")
        elif isinstance(data, list):
            # sometimes APIs return list of rows directly
            hist = data
        if not hist:
            return pd.DataFrame(columns=["date", "adjClose"]).astype({"date": "datetime64[ns]", "adjClose": "float64"})

        df = pd.DataFrame(hist)
        if "date" not in df.columns:
            return pd.DataFrame(columns=["date", "adjClose"]).astype({"date": "datetime64[ns]", "adjClose": "float64"})

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # prefer adjClose, fallback to close
        if "adjClose" not in df.columns and "adjclose" in df.columns:
            df.rename(columns={"adjclose": "adjClose"}, inplace=True)
        if "adjClose" not in df.columns:
            df["adjClose"] = df.get("close")
        df = df[["date", "adjClose"]].dropna().sort_values("date").drop_duplicates("date")
        df["adjClose"] = pd.to_numeric(df["adjClose"], errors="coerce")
        df = df.dropna()
        return df.reset_index(drop=True)
