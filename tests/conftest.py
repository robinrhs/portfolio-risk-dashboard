"""Shared synthetic market data so no test ever needs the network."""

import datetime as dt
import zlib

import numpy as np
import pandas as pd


def synthetic_prices(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Deterministic geometric-Brownian-motion prices on business days.

    Tickers starting with "BAD" come back empty, mimicking an invalid or
    delisted symbol on Yahoo Finance.
    """
    index = pd.bdate_range(start, end)
    common = np.random.default_rng(0).normal(0.0003, 0.01, len(index))  # shared "market" factor
    columns = {}
    for ticker in tickers:
        if ticker.startswith("BAD"):
            columns[ticker] = np.full(len(index), np.nan)
            continue
        rng = np.random.default_rng(zlib.crc32(ticker.encode()))
        own = rng.normal(0.0001, 0.008, len(index))
        loading = rng.uniform(0.3, 1.2)
        columns[ticker] = 100 * np.cumprod(1 + loading * common + own)
    return pd.DataFrame(columns, index=index)


def synthetic_currencies(tickers: list[str]) -> dict[str, str | None]:
    return {t: "EUR" if t.endswith(".DE") else "USD" for t in tickers}
