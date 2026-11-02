"""Which positions drive the portfolio's return and its risk?

Two decompositions, both of which add up exactly to the portfolio total:

* Return contribution: w_i * mu_i, summing to the portfolio's arithmetic mean return.
* Risk contribution (Euler allocation): w_i * (Sigma w)_i / sigma_p, summing to the
  portfolio volatility sigma_p.

A position can have a small weight but a large risk contribution if it is
volatile and correlated with the rest of the portfolio; that is the whole
point of looking at risk contributions instead of weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_risk.metrics import TRADING_DAYS


def return_contributions(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Contribution of each position to the annualised ARITHMETIC mean portfolio return.

    contrib_i = w_i * mean(r_i) * N

    Arithmetic means are linear, so the contributions add up exactly to the
    portfolio's annualised mean return. They do NOT add up to the geometric
    CAGR shown in the headline metrics (the gap is roughly sigma_p^2 / 2).
    """
    w = weights.reindex(asset_returns.columns)
    mu = asset_returns.mean() * periods_per_year
    return (w * mu).rename("return_contribution")


def risk_contributions(cov: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Euler decomposition of portfolio volatility into per-position contributions.

    sigma_p = sqrt(w' Sigma w)
    MCR_i   = d sigma_p / d w_i = (Sigma w)_i / sigma_p      (marginal contribution)
    RC_i    = w_i * MCR_i                                     (risk contribution)
    sum_i RC_i = w' Sigma w / sigma_p = sigma_p               (Euler's theorem)

    Volatility is homogeneous of degree 1 in the weights (double all weights,
    volatility doubles), which is exactly the condition for Euler's theorem to
    make the contributions add up to the total.

    Pass an annualised covariance matrix to get annualised contributions.
    Returns columns: weight, marginal, contribution, share (contribution / sigma_p).
    """
    w = weights.reindex(cov.columns).to_numpy(dtype=float)
    sigma = cov.to_numpy(dtype=float)
    port_vol = float(np.sqrt(w @ sigma @ w))
    if np.isclose(port_vol, 0.0):
        raise ValueError(
            "Portfolio volatility is zero (the prices did not move in this period), so "
            "volatility-based figures and risk contributions are undefined."
        )
    marginal = sigma @ w / port_vol
    contribution = w * marginal
    return pd.DataFrame(
        {
            "weight": w,
            "marginal": marginal,
            "contribution": contribution,
            "share": contribution / port_vol,
        },
        index=cov.columns,
    )
