import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk import metrics
from portfolio_risk.contributions import return_contributions, risk_contributions
from portfolio_risk.portfolio import (
    DEFAULT_PORTFOLIO,
    clean_portfolio,
    default_portfolio_table,
    portfolio_returns,
    read_portfolio_csv,
)

EXAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "example_portfolio.csv"


def table(tickers, weights) -> pd.DataFrame:
    return pd.DataFrame({"ticker": tickers, "weight": weights})


# --- weights ------------------------------------------------------------------


def test_weights_are_normalised_with_a_note():
    result = clean_portfolio(table(["aaa", "BBB"], [30, 30]))
    assert result.weights.to_dict() == {"AAA": 0.5, "BBB": 0.5}
    assert any("rescaled" in n for n in result.notes)


@pytest.mark.parametrize("weights", [[60, 40], [0.6, 0.4]])
def test_no_note_when_weights_already_sum_to_100_percent(weights):
    result = clean_portfolio(table(["A", "B"], weights))
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.notes == []


def test_duplicates_blanks_and_zero_weights():
    result = clean_portfolio(table(["A", " a ", "", None, "C"], [25, 25, 10, 10, 0]))
    assert result.weights.to_dict() == {"A": 1.0}
    assert any("Duplicate" in n for n in result.notes)


def test_invalid_weights_raise():
    with pytest.raises(ValueError, match="long-only"):
        clean_portfolio(table(["A", "B"], [120, -20]))
    with pytest.raises(ValueError, match="at least one"):
        clean_portfolio(table(["A"], [0]))


def test_default_portfolio_sums_to_100():
    assert sum(DEFAULT_PORTFOLIO.values()) == pytest.approx(100.0)
    assert clean_portfolio(default_portfolio_table()).notes == []


def test_read_csv_with_any_header_case():
    df = read_portfolio_csv(io.StringIO("Ticker,Weight\nspy,70\nagg,30\n"))
    assert clean_portfolio(df).weights.to_dict() == {"SPY": 0.7, "AGG": 0.3}


def test_example_csv_is_valid():
    weights = clean_portfolio(read_portfolio_csv(EXAMPLE_CSV)).weights
    assert weights.sum() == pytest.approx(1.0)
    assert len(weights) >= 3


# --- portfolio returns and contributions -------------------------------------


def random_returns(n_assets: int = 4, n_days: int = 750, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_assets, n_assets))
    cov = a @ a.T * 1e-5  # a valid (positive semi-definite) daily covariance
    data = rng.multivariate_normal(np.full(n_assets, 3e-4), cov, size=n_days)
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(n_assets)])


def test_portfolio_returns_are_weighted_sum():
    r = pd.DataFrame({"A": [0.01, -0.02], "B": [0.03, 0.01]})
    w = pd.Series({"B": 0.25, "A": 0.75})  # order must not matter
    np.testing.assert_allclose(portfolio_returns(r, w), [0.015, -0.0125])


def test_return_contributions_sum_to_portfolio_mean_return():
    r = random_returns()
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=r.columns)
    contrib = return_contributions(r, w)
    port_mean = metrics.annualized_mean_return(portfolio_returns(r, w))
    assert contrib.sum() == pytest.approx(port_mean)


def test_risk_contributions_sum_to_portfolio_volatility():
    r = random_returns()
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=r.columns)
    cov = r.cov() * 252
    rc = risk_contributions(cov, w)
    port_vol = metrics.annualized_volatility(portfolio_returns(r, w))
    assert rc["contribution"].sum() == pytest.approx(port_vol)
    assert rc["share"].sum() == pytest.approx(1.0)


def test_equal_uncorrelated_assets_contribute_equally():
    cov = pd.DataFrame(np.eye(3) * 0.04, index=list("ABC"), columns=list("ABC"))
    w = pd.Series(1 / 3, index=list("ABC"))
    rc = risk_contributions(cov, w)
    np.testing.assert_allclose(rc["share"], [1 / 3] * 3)
    # sigma_p = sqrt(3 * (1/3)^2 * 0.04) = 0.2 / sqrt(3): diversification lowers volatility.
    assert rc["contribution"].sum() == pytest.approx(0.2 / np.sqrt(3))
