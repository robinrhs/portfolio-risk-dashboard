"""One call that turns aligned prices + weights into every number the dashboard shows.

This is the glue between the maths modules and the UI. The Streamlit app and
the command-line report both call `analyze`, so they always agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from portfolio_risk import metrics as m
from portfolio_risk.contributions import return_contributions, risk_contributions
from portfolio_risk.portfolio import portfolio_returns


@dataclass(frozen=True)
class Settings:
    risk_free: float = 0.02  # annual, also used as the Sortino MAR
    confidence: float = 0.95  # VaR / ES confidence level
    horizon_days: int = 1  # VaR / ES horizon (> 1: see TailRisk)


@dataclass(frozen=True)
class SeriesStats:
    """Headline statistics of one return series."""

    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: m.MaxDrawdown


@dataclass(frozen=True)
class TailRisk:
    """VaR / ES as positive loss fractions at the chosen confidence and horizon.

    For horizon_days > 1 the historical VaR/ES are 1-day values x sqrt(h) (an
    approximation), while the parametric VaR/ES use the h-day normal distribution
    implied by i.i.d. normal daily returns (mean h * mu, volatility sigma * sqrt(h)).
    historical_var_overlapping is the empirical VaR of the overlapping h-day
    returns in the sample, shown next to the scaled figure as a reality check
    (identical to historical_var when h = 1).
    """

    confidence: float
    horizon_days: int
    historical_var: float
    historical_es: float
    parametric_var: float
    parametric_es: float
    historical_var_overlapping: float


@dataclass(frozen=True)
class Analysis:
    asset_returns: pd.DataFrame
    portfolio: pd.Series
    benchmark: pd.Series | None
    portfolio_stats: SeriesStats
    benchmark_stats: SeriesStats | None
    beta: float
    correlation_to_benchmark: float
    tail_risk: TailRisk
    correlation_matrix: pd.DataFrame
    contributions: pd.DataFrame  # weight, return contribution, risk contribution, risk share


def series_stats(returns: pd.Series, risk_free: float) -> SeriesStats:
    return SeriesStats(
        total_return=m.total_return(returns),
        annual_return=m.annualized_return(returns),
        annual_volatility=m.annualized_volatility(returns),
        sharpe=m.sharpe_ratio(returns, risk_free),
        # MAR = risk-free rate: "downside" means earning less than cash would have.
        sortino=m.sortino_ratio(returns, mar_annual=risk_free),
        max_drawdown=m.max_drawdown(returns),
    )


def tail_risk(returns: pd.Series, confidence: float, horizon_days: int) -> TailRisk:
    """VaR/ES from daily data: historical ones scaled by sqrt(h), normal ones in closed form."""

    def scale(x: float) -> float:
        return m.scale_to_horizon(x, horizon_days)

    # Reality check for the sqrt-time rule; undefined if the sample is shorter than h.
    windows = m.overlapping_returns(returns, horizon_days)
    overlapping_var = m.historical_var(windows, confidence) if len(windows) else float("nan")

    return TailRisk(
        confidence=confidence,
        horizon_days=horizon_days,
        historical_var=scale(m.historical_var(returns, confidence)),
        historical_es=scale(m.historical_es(returns, confidence)),
        parametric_var=m.parametric_var(returns, confidence, horizon_days),
        parametric_es=m.parametric_es(returns, confidence, horizon_days),
        historical_var_overlapping=overlapping_var,
    )


def analyze(
    prices: pd.DataFrame,
    weights: pd.Series,
    benchmark_prices: pd.Series | None,
    settings: Settings,
) -> Analysis:
    """Compute all dashboard figures from aligned daily prices.

    `prices` holds one column per portfolio position (already cleaned and
    aligned), `weights` sums to 1 over those columns.
    """
    asset_returns = m.simple_returns(prices[list(weights.index)])
    port = portfolio_returns(asset_returns, weights)

    bench = None
    bench_stats = None
    beta = corr = float("nan")
    if benchmark_prices is not None:
        bench = m.simple_returns(benchmark_prices).reindex(port.index).dropna()
        bench.name = "Benchmark"
        if len(bench) > 1:
            bench_stats = series_stats(bench, settings.risk_free)
            beta = m.beta(port, bench)
            corr = m.correlation(port, bench)
        else:
            bench = None

    annual_cov = asset_returns.cov() * m.TRADING_DAYS
    rc = risk_contributions(annual_cov, weights)
    contributions = pd.DataFrame(
        {
            "weight": weights,
            "return_contribution": return_contributions(asset_returns, weights),
            "risk_contribution": rc["contribution"],
            "risk_share": rc["share"],
        }
    )

    return Analysis(
        asset_returns=asset_returns,
        portfolio=port,
        benchmark=bench,
        portfolio_stats=series_stats(port, settings.risk_free),
        benchmark_stats=bench_stats,
        beta=beta,
        correlation_to_benchmark=corr,
        tail_risk=tail_risk(port, settings.confidence, settings.horizon_days),
        correlation_matrix=asset_returns.corr(),
        contributions=contributions,
    )


@dataclass(frozen=True)
class MeanVarianceInputs:
    """Annualised inputs for the optimiser, on the same basis as the headline Sharpe ratio."""

    mu: pd.Series  # arithmetic mean daily return x N
    cov: pd.DataFrame  # daily covariance (ddof = 1) x N
    risk_free: float  # daily risk-free rate x N (not the raw annual rate)


def mean_variance_inputs(asset_returns: pd.DataFrame, risk_free: float) -> MeanVarianceInputs:
    """Historical mu, Sigma and risk-free rate for the efficient frontier.

    The headline Sharpe ratio is mean(r_t - rf_d) * N / (std(r_t) * sqrt(N)) with the
    geometrically de-compounded daily rate rf_d = (1 + rf) ** (1/N) - 1. That equals
    (w' mu - N * rf_d) / sqrt(w' Sigma w) with the annualised mu and Sigma below, so
    passing N * rf_d (not rf itself) to the optimiser makes the frontier tab's
    Sharpe figures identical to the headline one for the same weights.
    """
    return MeanVarianceInputs(
        mu=asset_returns.mean() * m.TRADING_DAYS,
        cov=asset_returns.cov() * m.TRADING_DAYS,
        risk_free=m.per_period_rate(risk_free) * m.TRADING_DAYS,
    )


def summary_table(result: Analysis) -> pd.DataFrame:
    """Portfolio vs benchmark headline figures as a tidy table (for display and CSV export)."""

    def column(s: SeriesStats) -> dict[str, float]:
        return {
            "Total return": s.total_return,
            "Annualised return (CAGR)": s.annual_return,
            "Annualised volatility": s.annual_volatility,
            "Sharpe ratio": s.sharpe,
            "Sortino ratio": s.sortino,
            "Max drawdown": s.max_drawdown.depth,
        }

    data = {"Portfolio": column(result.portfolio_stats)}
    if result.benchmark_stats is not None:
        data["Benchmark"] = column(result.benchmark_stats)
    table = pd.DataFrame(data)
    table.loc["Beta to benchmark", "Portfolio"] = result.beta
    table.loc["Correlation to benchmark", "Portfolio"] = result.correlation_to_benchmark
    t = result.tail_risk
    label = f"{t.confidence:.1%} {t.horizon_days}-day"
    table.loc[f"Historical VaR ({label})", "Portfolio"] = t.historical_var
    table.loc[f"Historical ES ({label})", "Portfolio"] = t.historical_es
    table.loc[f"Parametric VaR ({label})", "Portfolio"] = t.parametric_var
    table.loc[f"Parametric ES ({label})", "Portfolio"] = t.parametric_es
    return table
