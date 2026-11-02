"""Command-line report: the same numbers as the dashboard, printed as a table.

Example:
    uv run portfolio-risk --start 2020-11-01 --end 2025-10-31
    uv run portfolio-risk --csv data/example_portfolio.csv --benchmark SPY
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from portfolio_risk import data
from portfolio_risk.analysis import Settings, analyze, summary_table
from portfolio_risk.portfolio import (
    DEFAULT_BENCHMARK,
    clean_portfolio,
    default_portfolio_table,
    read_portfolio_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    today = dt.date.today()
    p = argparse.ArgumentParser(description="Portfolio risk & performance report")
    p.add_argument("--csv", help="CSV with columns ticker, weight (default: built-in portfolio)")
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument(
        "--start", type=dt.date.fromisoformat, default=today - dt.timedelta(days=5 * 365)
    )
    p.add_argument("--end", type=dt.date.fromisoformat, default=today)
    p.add_argument("--rf", type=float, default=0.02, help="annual risk-free rate, e.g. 0.02")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--horizon", type=int, default=1, help="VaR horizon in trading days")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    table = read_portfolio_csv(args.csv) if args.csv else default_portfolio_table()
    portfolio = clean_portfolio(table)
    positions = list(portfolio.weights.index)
    tickers = list(dict.fromkeys([*positions, args.benchmark]))

    cleaned = data.clean_prices(data.download_prices(tickers, args.start, args.end), tickers)
    for message in cleaned.messages() + portfolio.notes:
        print(f"NOTE: {message}")
    note = data.currency_warning(data.fetch_currencies(list(cleaned.prices.columns)))
    if note:
        print(f"NOTE: {note}")

    valid = [t for t in positions if t in cleaned.prices.columns]
    weights = portfolio.weights[valid] / portfolio.weights[valid].sum()
    bench = cleaned.prices.get(args.benchmark)
    settings = Settings(args.rf, args.confidence, args.horizon)
    result = analyze(cleaned.prices[valid], weights, bench, settings)

    idx = result.portfolio.index
    print(f"Period: {idx[0]:%Y-%m-%d} to {idx[-1]:%Y-%m-%d} ({len(idx)} daily returns)")
    print(f"Weights: {', '.join(f'{t} {w:.1%}' for t, w in weights.items())}\n")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(summary_table(result).to_string(na_rep="-"))
        print("\nContributions (annualised):")
        print(result.contributions.to_string())
    mdd = result.portfolio_stats.max_drawdown
    recovery = f"{mdd.recovery_date:%Y-%m-%d}" if mdd.recovery_date is not None else "not recovered"
    if mdd.trough_date is not None:
        print(
            f"\nMax drawdown: peak {mdd.peak_date:%Y-%m-%d}, trough {mdd.trough_date:%Y-%m-%d}, "
            f"recovery {recovery}"
        )


if __name__ == "__main__":
    main()
