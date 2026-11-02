"""Portfolio definition: cleaning user input, normalising weights, portfolio returns.

UI-free: the Streamlit app hands in a table of tickers and weights and gets
back clean, normalised weights plus human-readable notes about what changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO

import numpy as np
import pandas as pd

# A diversified, long-only default so the dashboard is useful on first load.
# All US-listed USD ETFs, so there is no currency mix in the default case.
DEFAULT_PORTFOLIO: dict[str, float] = {
    "VTI": 30.0,  # US total stock market
    "VEA": 20.0,  # developed markets ex-US
    "VWO": 10.0,  # emerging markets
    "BND": 25.0,  # US investment-grade bonds
    "GLD": 10.0,  # gold
    "VNQ": 5.0,  # US real estate (REITs)
}
DEFAULT_BENCHMARK = "URTH"  # iShares MSCI World ETF (USD)


@dataclass(frozen=True)
class CleanPortfolio:
    """Validated portfolio: weights sum to exactly 1, plus notes for the user."""

    weights: pd.Series
    notes: list[str] = field(default_factory=list)


def clean_portfolio(table: pd.DataFrame) -> CleanPortfolio:
    """Validate a ticker/weight table and normalise the weights to sum to 1.

    Expects columns "ticker" and "weight" (weights in any unit: %, fractions or
    amounts in EUR all work, because only the proportions matter).

    Rules (long-only portfolio):
    * rows with a blank ticker, and zero weights, are dropped,
    * tickers with a blank weight are dropped with a note naming them,
    * a weight that is not a plain finite number ("50%", "1,000", "abc", "inf")
      is rejected with an error naming the ticker, instead of being dropped silently,
    * tickers are upper-cased; duplicates are added together,
    * negative weights (short positions) are rejected,
    * weights are rescaled so they sum to 100 %; the user is told if that changed anything.
    """
    missing = {"ticker", "weight"} - set(table.columns)
    if missing:
        raise ValueError(f"Portfolio table is missing column(s): {', '.join(sorted(missing))}")

    df = table[["ticker", "weight"]].copy()
    df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()
    df = df[df["ticker"].notna() & (df["ticker"] != "")]

    raw = df["weight"]
    blank = raw.astype("string").str.strip().fillna("").eq("")
    numeric = pd.to_numeric(raw.where(~blank), errors="coerce")
    unusable = ~blank & ~np.isfinite(numeric)
    if unusable.any():
        bad = ", ".join(
            f"{t} ({str(v).strip()!r})"
            for t, v in zip(df.loc[unusable, "ticker"], raw[unusable], strict=True)
        )
        raise ValueError(
            "Weights must be plain numbers such as 50 or 0.5 (no % sign, no thousands "
            f"separator): {bad}"
        )
    notes: list[str] = []
    if blank.any():
        notes.append(f"No weight given for {', '.join(df.loc[blank, 'ticker'])}; left out.")
    df["weight"] = numeric
    df = df[~blank]

    if (df["weight"] < 0).any():
        shorts = ", ".join(df.loc[df["weight"] < 0, "ticker"])
        raise ValueError(f"Negative weights are not supported (long-only): {shorts}")

    df = df[df["weight"] > 0]
    if df.empty:
        raise ValueError("Enter at least one ticker with a positive weight.")

    if df["ticker"].duplicated().any():
        dupes = ", ".join(sorted(df.loc[df["ticker"].duplicated(), "ticker"].unique()))
        notes.append(f"Duplicate tickers were combined: {dupes}.")
    weights = df.groupby("ticker", sort=False)["weight"].sum().astype(float)

    total = float(weights.sum())
    # Accept inputs that already sum to 100 (percent) or 1 (fractions) without a note.
    if not (np.isclose(total, 100.0) or np.isclose(total, 1.0)):
        notes.append(f"Weights summed to {total:,.2f} and were rescaled proportionally to 100 %.")
    weights = weights / total
    weights.index.name = "ticker"
    weights.name = "weight"
    return CleanPortfolio(weights=weights, notes=notes)


def read_portfolio_csv(source: str | IO) -> pd.DataFrame:
    """Read a CSV with a ticker column and a weight column (header names case-insensitive)."""
    df = pd.read_csv(source)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "ticker" not in df.columns or "weight" not in df.columns:
        raise ValueError("CSV needs two columns named 'ticker' and 'weight'.")
    return df[["ticker", "weight"]]


def default_portfolio_table() -> pd.DataFrame:
    """The default portfolio as an editable table (weights in %)."""
    return pd.DataFrame(
        {"ticker": list(DEFAULT_PORTFOLIO), "weight": list(DEFAULT_PORTFOLIO.values())}
    )


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily portfolio return of a constant-weight portfolio: r_p,t = sum_i w_i * r_i,t.

    Assumes the portfolio is rebalanced back to the target weights every day
    (a "constant-mix" portfolio) at zero cost. A buy-and-hold portfolio would
    instead let the weights drift with prices.
    """
    w = weights.reindex(asset_returns.columns)
    if w.isna().any():
        raise ValueError("Every return column needs a weight.")
    return (asset_returns * w).sum(axis=1).rename("Portfolio")
