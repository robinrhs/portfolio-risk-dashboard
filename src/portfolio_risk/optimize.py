"""Mean-variance (Markowitz) optimisation: efficient frontier, min-variance and max-Sharpe.

Inputs are an annualised vector of expected returns mu and an annualised
covariance matrix Sigma. In the app both are estimated from historical
returns, which is the weakest part of mean-variance analysis: small changes
in mu move the "optimal" weights a lot. Treat the output as an illustration
of the risk/return trade-off, not as a recommended allocation.

Constraints everywhere: long-only (0 <= w_i <= 1) and fully invested (sum w_i = 1).

`risk_free` must be on the same basis as mu. The app passes N x the daily
risk-free rate (see analysis.mean_variance_inputs), so the Sharpe figures here
match the headline Sharpe ratio for the same weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class OptimizedPortfolio:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """Expected portfolio return: mu_p = w' mu."""
    return float(weights @ mu)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio volatility: sigma_p = sqrt(w' Sigma w)."""
    return float(np.sqrt(weights @ cov @ weights))


def _solve(objective, n_assets: int, extra_constraints: list[dict] | None = None) -> np.ndarray:
    """Minimise `objective` over long-only, fully-invested weights with SLSQP."""
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    constraints += extra_constraints or []
    start = np.full(n_assets, 1.0 / n_assets)  # equal weights as a neutral starting point
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Optimisation failed: {result.message}")
    # Clip tiny negative values produced by floating-point noise and re-normalise.
    w = np.clip(result.x, 0.0, None)
    return w / w.sum()


def _package(w: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float) -> OptimizedPortfolio:
    ret = portfolio_return(w, mu.to_numpy())
    vol = portfolio_volatility(w, cov.to_numpy())
    sharpe = (ret - rf) / vol if vol > 0 else float("nan")
    return OptimizedPortfolio(pd.Series(w, index=mu.index, name="weight"), ret, vol, sharpe)


def min_variance_portfolio(
    mu: pd.Series, cov: pd.DataFrame, risk_free: float = 0.0
) -> OptimizedPortfolio:
    """Global minimum-variance portfolio: min_w w' Sigma w.

    Does not use expected returns at all, which is why it is considered more
    robust than max-Sharpe (expected returns are the hardest input to estimate).
    """
    sigma = cov.to_numpy()
    w = _solve(lambda w: w @ sigma @ w, len(mu))
    return _package(w, mu, cov, risk_free)


def max_sharpe_portfolio(
    mu: pd.Series, cov: pd.DataFrame, risk_free: float = 0.0
) -> OptimizedPortfolio:
    """Tangency portfolio: max_w (w' mu - rf) / sqrt(w' Sigma w).

    The point where a line from the risk-free rate touches the efficient
    frontier (the capital market line in textbook CAPM).

    Raises ValueError if no asset's expected return is above the risk-free rate.
    Every long-only portfolio then has a Sharpe ratio <= 0, and "maximising" it
    picks the least negative ratio, which favours the most volatile asset even
    if it has the lowest return. That is not a meaningful tangency portfolio.
    """
    if float(mu.max()) <= risk_free:
        raise ValueError(
            "No asset had an expected return above the risk-free rate in this period, "
            "so a max-Sharpe (tangency) portfolio is not meaningful."
        )
    m, sigma = mu.to_numpy(), cov.to_numpy()

    def negative_sharpe(w: np.ndarray) -> float:
        return -(w @ m - risk_free) / np.sqrt(w @ sigma @ w)

    w = _solve(negative_sharpe, len(mu))
    return _package(w, mu, cov, risk_free)


def efficient_frontier(
    mu: pd.Series, cov: pd.DataFrame, n_points: int = 30, risk_free: float = 0.0
) -> pd.DataFrame:
    """Trace the long-only efficient frontier.

    For each target return between the min-variance portfolio's return and the
    highest single-asset return (the top of a long-only frontier), solve

        min_w  w' Sigma w   s.t.  w' mu = target,  sum w = 1,  0 <= w <= 1.

    Returns one row per frontier point with columns "return", "volatility",
    "sharpe" and one weight column per asset.
    """
    m, sigma = mu.to_numpy(), cov.to_numpy()
    low = min_variance_portfolio(mu, cov, risk_free).expected_return
    high = float(m.max())
    rows = []
    for target in np.linspace(low, high, n_points):
        on_target = {"type": "eq", "fun": lambda w, t=target: w @ m - t}
        w = _solve(lambda w: w @ sigma @ w, len(mu), [on_target])
        p = _package(w, mu, cov, risk_free)
        rows.append(
            {"return": p.expected_return, "volatility": p.volatility, "sharpe": p.sharpe}
            | p.weights.to_dict()
        )
    return pd.DataFrame(rows)
