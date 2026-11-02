"""Performance and risk metrics for a series of periodic (daily) returns.

Everything in this module is pure maths on pandas/numpy objects: no Streamlit,
no downloads, no plotting. That keeps the finance logic easy to test with
synthetic return series whose correct answers are known in advance.

Conventions used throughout:

* Returns are *simple* returns per period, r_t = P_t / P_{t-1} - 1.
* "periods_per_year" is 252 for daily trading data.
* Annual input rates (risk-free rate, MAR) are converted to a per-period rate
  with geometric de-compounding: r_period = (1 + r_annual) ** (1 / 252) - 1.
* VaR and Expected Shortfall are reported as POSITIVE numbers meaning
  "loss as a fraction of portfolio value" (0.02 = a 2 % loss).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Period-over-period simple returns from a price series (first row dropped)."""
    return prices.pct_change().iloc[1:]


def per_period_rate(annual_rate: float, periods_per_year: int = TRADING_DAYS) -> float:
    """Convert an annual rate (e.g. 3 % risk-free) into the equivalent daily rate.

    Geometric de-compounding: compounding the daily rate 252 times gives back
    exactly the annual rate, (1 + r_d) ** 252 = 1 + r_annual.
    """
    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


def wealth_index(returns: pd.Series) -> pd.Series:
    """Growth of 1 unit of money invested at the start: W_t = prod_{s<=t} (1 + r_s)."""
    return (1.0 + returns).cumprod()


# ---------------------------------------------------------------------------
# Return and volatility
# ---------------------------------------------------------------------------


def total_return(returns: pd.Series) -> float:
    """Cumulative return over the whole sample: prod(1 + r_t) - 1."""
    return float(np.prod(1.0 + returns.to_numpy()) - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric annualised return (CAGR).

    CAGR = (prod(1 + r_t)) ** (periods_per_year / n) - 1

    Geometric, not arithmetic: this is the constant yearly rate that would have
    turned the starting value into the ending value. It is what an investor
    actually earned, and it is lower than the arithmetic mean whenever returns
    are volatile ("volatility drag" ~ sigma^2 / 2).
    """
    n = len(returns)
    if n == 0:
        return float("nan")
    growth = np.prod(1.0 + returns.to_numpy())
    return float(growth ** (periods_per_year / n) - 1.0)


def annualized_mean_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Arithmetic mean return scaled to a year: mean(r_t) * periods_per_year.

    Used where linearity matters (Sharpe ratio, return contributions,
    mean-variance optimisation), because arithmetic means add up across assets.
    """
    return float(returns.mean() * periods_per_year)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised standard deviation: std(r_t, ddof=1) * sqrt(periods_per_year).

    The square-root-of-time scaling assumes returns are independent from day
    to day (variance of a sum of i.i.d. returns grows linearly with time).
    """
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Risk-adjusted return
# ---------------------------------------------------------------------------


def sharpe_ratio(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe ratio: excess return per unit of total volatility.

    Sharpe = mean(r_t - rf_d) * N / (std(r_t - rf_d) * sqrt(N))

    rf_d is the daily risk-free rate and N = periods_per_year. A constant
    risk-free rate is assumed, so std(r - rf) = std(r). Returns NaN when
    volatility is zero (the ratio is undefined for a riskless series).
    """
    excess = returns - per_period_rate(risk_free_annual, periods_per_year)
    vol = excess.std(ddof=1)
    if not np.isfinite(vol) or np.isclose(vol, 0.0):
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def downside_deviation(
    returns: pd.Series,
    mar_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised downside deviation relative to a minimum acceptable return (MAR).

    DD = sqrt( mean( min(r_t - MAR_d, 0)^2 ) ) * sqrt(N)

    Only returns *below* the MAR count as risk; upside surprises are ignored.
    The mean is taken over ALL observations (not just the negative ones),
    which is the standard Sortino & Price definition.
    """
    shortfall = np.minimum(returns.to_numpy() - per_period_rate(mar_annual, periods_per_year), 0.0)
    return float(np.sqrt(np.mean(shortfall**2)) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    mar_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised Sortino ratio: return above the MAR per unit of downside risk.

    Sortino = mean(r_t - MAR_d) * N / DD

    Same idea as Sharpe, but the denominator only penalises bad volatility.
    Returns NaN when there is no downside at all (DD = 0).
    """
    dd = downside_deviation(returns, mar_annual, periods_per_year)
    if np.isclose(dd, 0.0):
        return float("nan")
    excess = returns.mean() - per_period_rate(mar_annual, periods_per_year)
    return float(excess * periods_per_year / dd)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxDrawdown:
    """Largest peak-to-trough fall of the wealth index.

    depth is negative (e.g. -0.25 = the portfolio fell 25 % from its high).
    peak_date is the last high before the trough. If the high was the starting
    capital itself, peak_date is the first RETURN date, i.e. the first trading
    day after the start price (the day the loss began), because the series is
    indexed by return dates. recovery_date is the
    first date the old high was regained, or None if it never was.
    """

    depth: float
    peak_date: pd.Timestamp | None
    trough_date: pd.Timestamp | None
    recovery_date: pd.Timestamp | None


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Percentage below the running maximum of wealth: D_t = W_t / max_{s<=t} W_s - 1.

    The starting capital (W_0 = 1) counts as the first peak, so a loss on the
    very first day already shows up as a drawdown.
    """
    wealth = wealth_index(returns)
    running_peak = np.maximum(wealth.cummax(), 1.0)
    return wealth / running_peak - 1.0


def max_drawdown(returns: pd.Series) -> MaxDrawdown:
    """Depth and dates of the worst drawdown in the sample."""
    if len(returns) == 0:
        return MaxDrawdown(float("nan"), None, None, None)
    dd = drawdown_series(returns)
    depth = float(dd.min())
    if depth >= 0.0:
        # Wealth never fell below a previous high: no drawdown.
        return MaxDrawdown(0.0, None, None, None)

    trough_date = dd.idxmin()
    before_trough = dd.loc[:trough_date]
    at_peak = before_trough[before_trough >= 0.0]
    # Last date at the high before the trough; if none, the peak was the start capital.
    peak_date = at_peak.index[-1] if len(at_peak) else dd.index[0]

    after_trough = dd.loc[trough_date:]
    recovered = after_trough[after_trough >= 0.0]
    recovery_date = recovered.index[0] if len(recovered) else None
    return MaxDrawdown(depth, peak_date, trough_date, recovery_date)


# ---------------------------------------------------------------------------
# Relation to a benchmark
# ---------------------------------------------------------------------------


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """CAPM beta: beta = Cov(r_p, r_b) / Var(r_b).

    How much the portfolio moves, on average, for a 1 % move in the benchmark.
    Only dates present in both series are used.
    """
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)
    if np.isclose(cov[1, 1], 0.0):
        return float("nan")
    return float(cov[0, 1] / cov[1, 1])


def correlation(returns: pd.Series, benchmark: pd.Series) -> float:
    """Pearson correlation of portfolio and benchmark returns (between -1 and 1)."""
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


# ---------------------------------------------------------------------------
# Value at Risk and Expected Shortfall (1-period horizon)
# ---------------------------------------------------------------------------


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical-simulation VaR: the loss exceeded on only (1 - c) of past days.

    VaR_c = -quantile(r, 1 - c)

    No distribution is assumed; the empirical return distribution IS the model.
    numpy's default linear interpolation is used between observations.
    """
    _check_confidence(confidence)
    return float(-np.quantile(returns.to_numpy(), 1.0 - confidence))


def historical_es(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Expected Shortfall (CVaR): average loss on the days beyond VaR.

    ES_c = -mean( r_t | r_t <= -VaR_c )

    VaR says "how bad is a bad day"; ES says "if it is that bad, how bad on
    average". ES is always >= VaR and captures the size of the tail.
    """
    _check_confidence(confidence)
    r = returns.to_numpy()
    cutoff = np.quantile(r, 1.0 - confidence)
    tail = r[r <= cutoff]
    return float(-tail.mean())


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (variance-covariance) VaR assuming normally distributed returns.

    VaR_c = -(mu + z_{1-c} * sigma) = z_c * sigma - mu

    mu and sigma are the sample mean and standard deviation of daily returns,
    z_c the standard normal quantile (1.645 for 95 %, 2.326 for 99 %). Real
    returns have fatter tails than the normal, so this tends to understate
    extreme losses.
    """
    _check_confidence(confidence)
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    return float(norm.ppf(confidence) * sigma - mu)


def parametric_es(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric Expected Shortfall under the normal assumption.

    ES_c = sigma * phi(z_c) / (1 - c) - mu

    phi is the standard normal density. Closed form for the mean of a normal
    distribution beyond its (1 - c) quantile.
    """
    _check_confidence(confidence)
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    z = norm.ppf(confidence)
    return float(sigma * norm.pdf(z) / (1.0 - confidence) - mu)


def overlapping_returns(returns: pd.Series, horizon_days: int) -> pd.Series:
    """Compounded returns over every window of h consecutive days ("rolling" h-day returns).

    R_t = prod_{s=t-h+1}^{t} (1 + r_s) - 1

    Used to check the sqrt-time approximation against what h-day losses
    actually looked like in the sample. The windows overlap (day t is in h
    of them), so neighbouring observations are not independent and quantiles
    estimated from them are noisier than the number of windows suggests.
    Returns n - h + 1 values; h = 1 gives back the daily returns.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    # Summing log growth over the window and converting back = compounding.
    log_growth = np.log1p(returns).rolling(horizon_days).sum()
    return np.expm1(log_growth).dropna()


def scale_to_horizon(one_day_risk: float, horizon_days: int) -> float:
    """Square-root-of-time scaling of a 1-day VaR/ES to an h-day horizon.

    VaR_h ~ VaR_1 * sqrt(h)

    APPROXIMATION ONLY: exact for i.i.d. normal returns with zero mean.
    It ignores autocorrelation, volatility clustering and fat tails, and
    assumes the portfolio is not traded during the horizon.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    return float(one_day_risk * np.sqrt(horizon_days))
