"""Metric tests on synthetic return series whose correct answers are known in closed form."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from portfolio_risk import metrics as m


def dated(values) -> pd.Series:
    """Wrap values in a Series with a business-day index (like real price data)."""
    values = np.asarray(values, dtype=float)
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)))


def exact_moments(n: int, mean: float, std: float, seed: int = 0) -> pd.Series:
    """Random series rescaled so its sample mean and sample std (ddof=1) are EXACTLY as given."""
    z = np.random.default_rng(seed).standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=1)
    return dated(mean + std * z)


# --- returns and volatility -------------------------------------------------


def test_simple_returns_from_prices():
    prices = dated([100.0, 110.0, 99.0])
    np.testing.assert_allclose(m.simple_returns(prices).to_numpy(), [0.10, -0.10])


def test_per_period_rate_compounds_back_to_annual():
    daily = m.per_period_rate(0.05)
    assert (1 + daily) ** 252 == pytest.approx(1.05)


def test_constant_returns():
    r = dated([0.001] * 252)
    assert m.annualized_return(r) == pytest.approx(1.001**252 - 1)
    assert m.total_return(r) == pytest.approx(1.001**252 - 1)
    assert m.annualized_volatility(r) == pytest.approx(0.0, abs=1e-12)
    # Sharpe is undefined without volatility.
    assert np.isnan(m.sharpe_ratio(r, 0.0))


def test_annualized_return_is_geometric():
    # Two years (504 days) that grow wealth by 21 % in total -> exactly 10 % p.a.
    daily = 1.21 ** (1 / 504) - 1
    assert m.annualized_return(dated([daily] * 504)) == pytest.approx(0.10)
    # +50 % then -50 % has a positive arithmetic mean but loses 25 %.
    r = dated([0.5, -0.5])
    assert m.total_return(r) == pytest.approx(-0.25)
    assert m.annualized_mean_return(r) == pytest.approx(0.0)


def test_known_volatility_and_sharpe():
    r = exact_moments(1000, mean=0.0004, std=0.01)
    assert m.annualized_volatility(r) == pytest.approx(0.01 * np.sqrt(252))
    rf = 0.02
    expected = (0.0004 - m.per_period_rate(rf)) / 0.01 * np.sqrt(252)
    assert m.sharpe_ratio(r, rf) == pytest.approx(expected)


def test_sortino_hand_calculation():
    r = dated([0.03, -0.01, 0.02, -0.02])
    # Shortfalls below MAR = 0: [0, -0.01, 0, -0.02]; mean of squares over ALL 4 days.
    dd_daily = np.sqrt((0.01**2 + 0.02**2) / 4)
    assert m.downside_deviation(r, 0.0) == pytest.approx(dd_daily * np.sqrt(252))
    assert m.sortino_ratio(r, 0.0) == pytest.approx(0.005 * 252 / (dd_daily * np.sqrt(252)))


def test_sortino_undefined_without_downside():
    assert np.isnan(m.sortino_ratio(dated([0.01, 0.02, 0.005]), 0.0))


# --- drawdown -----------------------------------------------------------------


def test_max_drawdown_with_dates():
    prices = dated([100, 120, 90, 60, 130])
    r = m.simple_returns(prices)
    mdd = m.max_drawdown(r)
    assert mdd.depth == pytest.approx(60 / 120 - 1)  # -50 %
    assert mdd.peak_date == prices.index[1]
    assert mdd.trough_date == prices.index[3]
    assert mdd.recovery_date == prices.index[4]


def test_drawdown_counts_loss_on_first_day_and_no_recovery():
    r = dated([-0.10, 0.05])
    dd = m.drawdown_series(r)
    assert dd.iloc[0] == pytest.approx(-0.10)
    mdd = m.max_drawdown(r)
    assert mdd.depth == pytest.approx(-0.10)
    assert mdd.peak_date == r.index[0]  # the starting capital was the peak
    assert mdd.recovery_date is None


def test_no_drawdown_for_rising_series():
    mdd = m.max_drawdown(dated([0.01, 0.02, 0.0]))
    assert mdd.depth == 0.0
    assert mdd.trough_date is None


# --- benchmark relation -------------------------------------------------------


def test_beta_and_correlation_of_scaled_benchmark():
    bench = exact_moments(500, 0.0003, 0.012, seed=1)
    assert m.beta(2.0 * bench, bench) == pytest.approx(2.0)
    assert m.correlation(2.0 * bench, bench) == pytest.approx(1.0)
    assert m.beta(-1.5 * bench, bench) == pytest.approx(-1.5)
    assert m.correlation(-1.5 * bench, bench) == pytest.approx(-1.0)


def test_beta_uses_only_overlapping_dates():
    bench = exact_moments(300, 0.0, 0.01, seed=2)
    port = 0.8 * bench.iloc[50:]  # portfolio history starts later
    assert m.beta(port, bench) == pytest.approx(0.8)


# --- VaR / ES -----------------------------------------------------------------


def test_historical_var_and_es_on_uniform_grid():
    # 100 equally spaced returns from -5.0 % to +4.9 % in 0.1 % steps.
    r = dated(np.round(np.arange(-50, 50) / 1000, 6))
    # 5 % quantile, linear interpolation: position 0.05 * 99 = 4.95 between -4.6 % and -4.5 %.
    assert m.historical_var(r, 0.95) == pytest.approx(0.04505)
    # Tail = the five returns at or below -4.505 %: mean of -5.0 ... -4.6 % = -4.8 %.
    assert m.historical_es(r, 0.95) == pytest.approx(0.048)


def test_parametric_var_and_es_closed_form():
    mu, sigma = 0.0005, 0.01
    r = exact_moments(2000, mu, sigma, seed=3)
    assert m.parametric_var(r, 0.95) == pytest.approx(1.6448536 * sigma - mu, rel=1e-6)
    assert m.parametric_var(r, 0.99) == pytest.approx(2.3263479 * sigma - mu, rel=1e-6)
    expected_es = sigma * norm.pdf(norm.ppf(0.95)) / 0.05 - mu
    assert m.parametric_es(r, 0.95) == pytest.approx(expected_es)


def test_historical_matches_parametric_on_large_normal_sample():
    # For truly normal data the two methods should agree (up to sampling noise).
    r = exact_moments(200_000, 0.0003, 0.01, seed=4)
    assert m.historical_var(r, 0.99) == pytest.approx(m.parametric_var(r, 0.99), rel=0.02)
    assert m.historical_es(r, 0.975) == pytest.approx(m.parametric_es(r, 0.975), rel=0.02)


def test_expected_shortfall_is_at_least_var():
    r = dated(np.random.default_rng(5).standard_t(df=3, size=5000) * 0.01)
    for c in (0.9, 0.95, 0.99):
        assert m.historical_es(r, c) >= m.historical_var(r, c)
        assert m.parametric_es(r, c) >= m.parametric_var(r, c)


def test_sqrt_time_scaling_and_input_checks():
    assert m.scale_to_horizon(0.02, 10) == pytest.approx(0.02 * np.sqrt(10))
    assert m.scale_to_horizon(0.02, 1) == pytest.approx(0.02)
    with pytest.raises(ValueError):
        m.scale_to_horizon(0.02, 0)
    with pytest.raises(ValueError):
        m.historical_var(dated([0.01, -0.01]), 95)  # percent instead of fraction


def test_overlapping_returns_compound_known_windows():
    r = dated([0.10, -0.10, 0.10, 0.0])
    two_day = m.overlapping_returns(r, 2)
    # Windows: (1.1 * 0.9), (0.9 * 1.1), (1.1 * 1.0), each minus 1.
    np.testing.assert_allclose(two_day.to_numpy(), [-0.01, -0.01, 0.10])
    assert list(two_day.index) == list(r.index[1:])  # labelled by the window's last day
    np.testing.assert_allclose(m.overlapping_returns(r, 1).to_numpy(), r.to_numpy())
    assert m.overlapping_returns(r, 5).empty  # sample shorter than the horizon
    with pytest.raises(ValueError):
        m.overlapping_returns(r, 0)


def test_overlapping_var_of_constant_loss_is_compounded_loss():
    r = dated([-0.01] * 50)
    # Every 10-day window lost 1 - 0.99^10; sqrt-time would say 0.01 * sqrt(10) instead.
    expected = 1 - 0.99**10
    assert m.historical_var(m.overlapping_returns(r, 10), 0.95) == pytest.approx(expected)
    assert m.scale_to_horizon(m.historical_var(r, 0.95), 10) != pytest.approx(expected)


def test_peak_date_is_first_return_date_when_start_capital_is_the_peak():
    prices = dated([100.0, 90.0, 95.0])
    mdd = m.max_drawdown(m.simple_returns(prices))
    # Returns are indexed from the second price on: the peak is reported as that first
    # return date (the day the loss began), not as the date of the starting price.
    assert mdd.peak_date == prices.index[1]
    assert mdd.trough_date == prices.index[1]
