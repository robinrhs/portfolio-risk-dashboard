"""Price data: download from Yahoo Finance and clean it into an aligned price table.

`download_prices` and `fetch_currency` are thin wrappers around yfinance (the
only functions here that touch the network). `clean_prices` is pure pandas and
does the data-quality work that real market data always needs:

* tickers that returned no data at all (typo, delisted, not on Yahoo),
* tickers whose history starts later than requested (or than the others) or ends earlier,
* short gaps from different exchange holidays (e.g. a Xetra ETF next to a US stock).

No Streamlit in here: the app turns the returned issues into st.warning messages.
"""

from __future__ import annotations

import datetime as dt
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

# Gaps of up to this many consecutive trading days are forward-filled (price held
# flat, i.e. a 0 % return), which mainly covers different exchange holidays.
MAX_FILL_DAYS = 5
# Below this many overlapping daily returns the statistics are too noisy to show.
MIN_OBSERVATIONS = 60

# yfinance triggers pandas deprecation warnings internally; hide those (and only those).
# Must come after `import yfinance`, which installs its own warning filter on import.
warnings.filterwarnings("ignore", module=r"yfinance(\..*)?$")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


class DataError(RuntimeError):
    """Raised when no usable price data could be obtained."""


@dataclass(frozen=True)
class CleanPrices:
    """Aligned adjusted-close prices plus a record of every data problem found."""

    prices: pd.DataFrame
    invalid: list[str] = field(default_factory=list)
    late_start: dict[str, pd.Timestamp] = field(default_factory=dict)
    early_end: dict[str, pd.Timestamp] = field(default_factory=dict)
    filled_cells: int = 0
    dropped_days: int = 0
    # What late_start was measured against: the requested start date, or None if
    # none was given (then the earliest first date among the tickers).
    requested_start: pd.Timestamp | None = None

    def messages(self) -> list[str]:
        """Plain-English description of the data issues, one sentence each."""
        out: list[str] = []
        if self.invalid:
            out.append(
                "No price data found for "
                + ", ".join(self.invalid)
                + " (typo, delisted, or not covered by Yahoo Finance). These were excluded."
            )
        if self.late_start:
            names = ", ".join(f"{t} ({d:%Y-%m-%d})" for t, d in self.late_start.items())
            if self.requested_start is not None:
                out.append(
                    f"{names} start later than the requested start date "
                    f"({self.requested_start:%Y-%m-%d}), so the analysis begins on "
                    f"{self.prices.index[0]:%Y-%m-%d}."
                )
            else:
                out.append(
                    f"Histories do not fully overlap: {names} start later than the other "
                    "tickers, so the analysis begins on the latest first date."
                )
        if self.early_end:
            names = ", ".join(f"{t} ({d:%Y-%m-%d})" for t, d in self.early_end.items())
            out.append(f"{names} stop trading early, so the analysis ends on that date.")
        if self.filled_cells:
            out.append(
                f"{self.filled_cells} missing daily prices (e.g. exchange holidays) were "
                f"filled with the previous close (max {MAX_FILL_DAYS} days in a row)."
            )
        if self.dropped_days:
            out.append(
                f"{self.dropped_days} trading days were dropped because a ticker had no price "
                f"for more than {MAX_FILL_DAYS} days in a row. The return after such a gap "
                "covers several days, which slightly distorts daily volatility and VaR."
            )
        return out


def clean_prices(
    raw: pd.DataFrame,
    tickers: list[str],
    tolerance_days: int = 7,
    requested_start: dt.date | None = None,
) -> CleanPrices:
    """Turn a raw (possibly ragged) price table into a fully aligned one.

    Steps:
    1. Tickers that are missing or entirely empty are flagged as invalid and dropped.
    2. The sample is cut to the window where ALL remaining tickers have prices
       (latest first date to earliest last date). Statistics such as correlation
       and portfolio volatility need every asset on every day.
    3. Short gaps inside that window are forward-filled; any row still incomplete
       (a gap longer than MAX_FILL_DAYS) is dropped and counted in `dropped_days`.

    Late starters are measured against `requested_start` if given (so a warning
    appears even when every ticker starts after it), otherwise against the
    earliest first date among the tickers. `tolerance_days` avoids warning about
    a ticker that starts a few days late just because of a weekend or holiday.
    """
    invalid = [t for t in tickers if t not in raw.columns or raw[t].dropna().empty]
    valid = [t for t in tickers if t not in invalid]
    if not valid:
        raise DataError("None of the tickers returned any price data.")

    prices = raw[valid].sort_index()
    first = {t: prices[t].first_valid_index() for t in valid}
    last = {t: prices[t].last_valid_index() for t in valid}
    earliest_first = min(first.values())
    latest_last = max(last.values())
    common_start = max(first.values())
    common_end = min(last.values())
    if common_start >= common_end:
        raise DataError("The price histories of these tickers do not overlap.")

    tolerance = pd.Timedelta(days=tolerance_days)
    reference = earliest_first if requested_start is None else pd.Timestamp(requested_start)
    late_start = {t: d for t, d in first.items() if d - reference > tolerance}
    early_end = {t: d for t, d in last.items() if latest_last - d > tolerance}

    window = prices.loc[common_start:common_end]
    filled = window.ffill(limit=MAX_FILL_DAYS)
    filled_cells = int(window.isna().sum().sum() - filled.isna().sum().sum())
    aligned = filled.dropna(how="any")
    dropped_days = len(filled) - len(aligned)
    return CleanPrices(
        aligned,
        invalid,
        late_start,
        early_end,
        filled_cells,
        dropped_days,
        None if requested_start is None else pd.Timestamp(requested_start),
    )


def currency_warning(currencies: dict[str, str | None]) -> str | None:
    """Warn if the positions are quoted in different currencies.

    Returns are computed in each asset's own quote currency. Mixing e.g. USD
    and EUR prices therefore ignores exchange-rate moves that a euro investor
    actually experiences, and the correlation/beta figures mix two currency views.
    """
    known = {t: c for t, c in currencies.items() if c}
    unknown = sorted(t for t, c in currencies.items() if not c)
    distinct = sorted(set(known.values()))
    parts: list[str] = []
    if len(distinct) > 1:
        by_ccy = "; ".join(
            f"{ccy}: {', '.join(sorted(t for t, c in known.items() if c == ccy))}"
            for ccy in distinct
        )
        parts.append(
            f"Mixed quote currencies ({by_ccy}). Returns are measured in each asset's own "
            "currency without FX conversion, so currency risk is NOT reflected."
        )
    if unknown:
        parts.append(f"Quote currency unknown for {', '.join(unknown)}.")
    return " ".join(parts) or None


# ---------------------------------------------------------------------------
# Network access (yfinance)
# ---------------------------------------------------------------------------


def download_prices(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download daily adjusted close prices (dividends and splits reinvested).

    `auto_adjust=True` makes Yahoo's "Close" the total-return adjusted price,
    which is what return and risk statistics should be based on. yfinance
    treats `end` as exclusive, so one day is added. Returns one column per
    requested ticker; tickers without data come back as all-NaN columns.
    """
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end + dt.timedelta(days=1),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:  # network errors, rate limits, ...
        raise DataError(f"Download from Yahoo Finance failed: {exc}") from exc

    if raw is None or raw.empty:
        raise DataError("Yahoo Finance returned no data (check tickers, dates or connection).")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(close, pd.DataFrame):
        close = close.to_frame()
    if list(close.columns) == ["Close"]:
        close.columns = tickers[:1]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.reindex(columns=tickers).astype(float)


def fetch_currency(ticker: str) -> str | None:
    """Quote currency of a ticker according to Yahoo (None if unknown)."""
    try:
        return yf.Ticker(ticker).fast_info["currency"]
    except Exception:  # unknown ticker, network error, missing field
        return None


def fetch_currencies(tickers: list[str]) -> dict[str, str | None]:
    """Quote currency per ticker, looked up in parallel (one small request each)."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(tickers, pool.map(fetch_currency, tickers), strict=True))
