from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

BINANCE_REST = "https://api.binance.com/api/v3/klines"
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
]
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def fetch_klines(
    symbol: str,
    interval: str = "1m",
    days: int = 30,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch klines in bounded pages and return only compact numeric OHLCV data.

    Pages are held only in memory; no raw datasets are persisted to disk.
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    client = session or requests.Session()
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    cursor = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    rows: list[list] = []
    while cursor < end_ms:
        response = client.get(
            BINANCE_REST,
            params={
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1][0]) + INTERVAL_MS[interval]
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.04)
    if not rows:
        raise RuntimeError(f"No market data returned for {symbol}")
    frame = pd.DataFrame(rows, columns=COLUMNS)
    numeric = ["open", "high", "low", "close", "volume"]
    frame[numeric] = frame[numeric].astype("float64")
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    return frame[["open_time", *numeric]].drop_duplicates("open_time").sort_values("open_time")
