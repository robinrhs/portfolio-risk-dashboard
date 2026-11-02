"""Command-line report: the same numbers as the dashboard, printed as a table.

Example:
    uv run portfolio-risk --start 2020-11-01 --end 2025-10-31
    uv run portfolio-risk --csv data/example_portfolio.csv --benchmark SPY

It applies the same input checks as the app (start before end, risk-free rate
between -2 % and 20 %, confidence between 0 and 1, at least
`data.MIN_OBSERVATIONS` overlapping trading days)
and exits with a one-line error message instead of a traceback.
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from portfolio_risk import data
from portfolio_risk.analysis import Analysis, Settings, analyze, summary_table
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
    p.add_argument(
        "--confidence", type=float, default=0.95, help="VaR / ES confidence, e.g. 0.95 (not 95)"
    )
    p.add_argument("--horizon", type=int, default=1, help="VaR horizon in trading days")
    args = p.parse_args(argv)

    # p.error prints the usage line plus the message and exits with status 2.
    if args.start >= args.end:
        p.error(f"--start ({args.start}) must be before --end ({args.end})")
    # Same range as the app's input (-2 % to 20 %); catches "2" typed for 2 %.
    if not -0.02 <= args.rf <= 0.20:
        p.error(f"--rf is an annual rate as a fraction, e.g. 0.02 for 2 % (got {args.rf})")
    if not 0.0 < args.confidence < 1.0:
        p.error(f"--confidence must be between 0 and 1, e.g. 0.95 (got {args.confidence})")
    if args.horizon < 1:
        p.error(f"--horizon must be at least 1 trading day (got {args.horizon})")
    return args


def run(args: argparse.Namespace) -> Analysis:
    """Download, clean and analyse; print data notes on the way. Raises on unusable input."""
    table = read_portfolio_csv(args.csv) if args.csv else default_portfolio_table()
    portfolio = clean_portfolio(table)
    positions = list(portfolio.weights.index)
    tickers = list(dict.fromkeys([*positions, args.benchmark]))

    cleaned = data.clean_prices(data.download_prices(tickers, args.start, args.end), tickers)
    for message in cleaned.messages() + portfolio.notes:
        print(f"NOTE: {message}")

    valid = [t for t in positions if t in cleaned.prices.columns]
    if not valid:
        raise data.DataError("None of the portfolio tickers has usable price data.")
    # Same rule as the app: too few days make annualised figures meaningless noise.
    if len(cleaned.prices) < data.MIN_OBSERVATIONS:
        raise data.DataError(
            f"Only {len(cleaned.prices)} overlapping trading days. Choose a longer period "
            f"(at least {data.MIN_OBSERVATIONS} days are needed for meaningful statistics)."
        )
    note = data.currency_warning(data.fetch_currencies(list(cleaned.prices.columns)))
    if note:
        print(f"NOTE: {note}")

    weights = portfolio.weights[valid] / portfolio.weights[valid].sum()
    bench = cleaned.prices.get(args.benchmark)
    settings = Settings(args.rf, args.confidence, args.horizon)
    return analyze(cleaned.prices[valid], weights, bench, settings)


def print_report(result: Analysis) -> None:
    idx = result.portfolio.index
    weights = result.contributions["weight"]
    print(f"Period: {idx[0]:%Y-%m-%d} to {idx[-1]:%Y-%m-%d} ({len(idx)} daily returns)")
    print(f"Weights: {', '.join(f'{t} {w:.1%}' for t, w in weights.items())}\n")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(summary_table(result).to_string(na_rep="-"))
        print("\nContributions (annualised):")
        print(result.contributions.to_string())
    t = result.tail_risk
    if t.horizon_days > 1:
        print(
            f"\nVaR/ES above: 1-day figures x sqrt({t.horizon_days}) (approximation). "
            f"Historical VaR of overlapping {t.horizon_days}-day returns for comparison: "
            f"{t.historical_var_overlapping:.4f}"
        )
    mdd = result.portfolio_stats.max_drawdown
    recovery = f"{mdd.recovery_date:%Y-%m-%d}" if mdd.recovery_date is not None else "not recovered"
    if mdd.trough_date is not None:
        print(
            f"\nMax drawdown: peak {mdd.peak_date:%Y-%m-%d}, trough {mdd.trough_date:%Y-%m-%d}, "
            f"recovery {recovery}"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = run(args)
    except (data.DataError, ValueError, OSError) as exc:
        # OSError covers a --csv path that does not exist.
        raise SystemExit(f"Error: {exc}") from None
    print_report(result)


if __name__ == "__main__":
    main()
