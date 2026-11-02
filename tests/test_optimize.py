import numpy as np
import pandas as pd
import pytest

from portfolio_risk.optimize import (
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    portfolio_volatility,
)


def two_assets(s1=0.20, s2=0.30, rho=0.2, mu1=0.06, mu2=0.10):
    cov = pd.DataFrame(
        [[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]], index=["A", "B"], columns=["A", "B"]
    )
    mu = pd.Series([mu1, mu2], index=["A", "B"])
    return mu, cov


def four_assets() -> tuple[pd.Series, pd.DataFrame]:
    vols = np.array([0.15, 0.20, 0.25, 0.05])
    corr = np.array(
        [
            [1.0, 0.6, 0.4, 0.0],
            [0.6, 1.0, 0.5, 0.1],
            [0.4, 0.5, 1.0, -0.1],
            [0.0, 0.1, -0.1, 1.0],
        ]
    )
    names = ["Stocks", "Tech", "EM", "Bonds"]
    cov = pd.DataFrame(np.outer(vols, vols) * corr, index=names, columns=names)
    return pd.Series([0.07, 0.10, 0.09, 0.03], index=names), cov


def test_two_asset_min_variance_matches_closed_form():
    mu, cov = two_assets()
    s1sq, s2sq, s12 = cov.iloc[0, 0], cov.iloc[1, 1], cov.iloc[0, 1]
    # w_A = (sigma_B^2 - sigma_AB) / (sigma_A^2 + sigma_B^2 - 2 sigma_AB)
    w_a = (s2sq - s12) / (s1sq + s2sq - 2 * s12)
    result = min_variance_portfolio(mu, cov)
    assert result.weights["A"] == pytest.approx(w_a, abs=1e-6)
    assert result.weights.sum() == pytest.approx(1.0)


def test_two_asset_max_sharpe_matches_tangency_formula():
    mu, cov = two_assets()
    rf = 0.02
    # Unconstrained tangency portfolio: w proportional to Sigma^-1 (mu - rf).
    raw = np.linalg.solve(cov.to_numpy(), mu.to_numpy() - rf)
    expected = raw / raw.sum()
    assert (expected > 0).all()  # so the long-only constraint is not binding
    result = max_sharpe_portfolio(mu, cov, rf)
    np.testing.assert_allclose(result.weights.to_numpy(), expected, atol=1e-5)


def test_frontier_weights_are_valid_and_frontier_is_efficient():
    mu, cov = four_assets()
    frontier = efficient_frontier(mu, cov, n_points=15, risk_free=0.02)
    weights = frontier[list(mu.index)]
    assert (weights >= -1e-9).all().all()  # long-only
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)  # fully invested
    # More return must cost (weakly) more risk along the efficient frontier.
    assert (np.diff(frontier["volatility"]) >= -1e-6).all()
    # First point is the min-variance portfolio, last point is the highest-return asset.
    assert frontier["volatility"].iloc[0] == pytest.approx(
        min_variance_portfolio(mu, cov).volatility, abs=1e-6
    )
    assert frontier["return"].iloc[-1] == pytest.approx(mu.max(), abs=1e-6)


def test_max_sharpe_beats_every_frontier_point_and_min_var_is_lowest_risk():
    mu, cov = four_assets()
    rf = 0.02
    frontier = efficient_frontier(mu, cov, n_points=25, risk_free=rf)
    best = max_sharpe_portfolio(mu, cov, rf)
    assert best.sharpe >= frontier["sharpe"].max() - 1e-6
    min_var = min_variance_portfolio(mu, cov, rf)
    equal = np.full(len(mu), 1 / len(mu))
    assert min_var.volatility <= portfolio_volatility(equal, cov.to_numpy())
    assert min_var.volatility <= frontier["volatility"].min() + 1e-6
