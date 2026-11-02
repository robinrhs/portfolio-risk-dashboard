import datetime as dt

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_prices

from portfolio_risk import charts, data, metrics
from portfolio_risk.analysis import Settings, analyze, mean_variance_inputs, summary_table

# --- clean_prices ------------------------------------------------------------


def ragged_prices() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=100)
    df = pd.DataFrame(
        {
            "A": np.linspace(100, 120, 100),
            "B": np.linspace(50, 40, 100),
            "LATE": np.linspace(10, 11, 100),
            "DEAD": np.nan,
        },
        index=idx,
    )
    df.loc[idx[:30], "LATE"] = np.nan  # listed 30 days later
    df.loc[idx[50:52], "B"] = np.nan  # two-day holiday gap
    return df


def test_clean_prices_flags_invalid_late_and_gaps():
    raw = ragged_prices()
    cleaned = data.clean_prices(raw, ["A", "B", "LATE", "DEAD", "MISSING"])
    assert cleaned.invalid == ["DEAD", "MISSING"]
    assert list(cleaned.late_start) == ["LATE"]
    assert cleaned.prices.index[0] == raw.index[30]  # common start = latest first date
    assert cleaned.filled_cells == 2
    assert not cleaned.prices.isna().any().any()
    # Forward-filled holiday: price held flat -> 0 % return on those days.
    assert cleaned.prices["B"].iloc[20] == cleaned.prices["B"].iloc[21]
    text = " ".join(cleaned.messages())
    assert "DEAD" in text and "LATE" in text and "filled" in text


def test_clean_prices_reports_days_dropped_for_long_gaps():
    raw = ragged_prices()[["A", "B"]].copy()
    raw.iloc[60:70, 1] = np.nan  # B has no price for 10 days (e.g. a trading halt)
    cleaned = data.clean_prices(raw, ["A", "B"])
    # The first MAX_FILL_DAYS are filled (plus the 2-day holiday in ragged_prices),
    # the rest of the long gap is dropped and reported rather than silently removed.
    assert cleaned.filled_cells == 2 + data.MAX_FILL_DAYS
    assert cleaned.dropped_days == 10 - data.MAX_FILL_DAYS
    assert len(cleaned.prices) == 100 - cleaned.dropped_days
    assert "5 trading days were dropped" in " ".join(cleaned.messages())


def test_late_benchmark_cuts_the_window_and_is_named():
    raw = ragged_prices()[["A", "B"]].copy()
    raw["BENCH"] = np.linspace(200, 210, 100)
    raw.iloc[:40, 2] = np.nan  # benchmark history starts 40 days later
    cleaned = data.clean_prices(raw, ["A", "B", "BENCH"])
    assert cleaned.prices.index[0] == raw.index[40]
    assert list(cleaned.late_start) == ["BENCH"]
    assert "BENCH (2024-02-26) start later" in " ".join(cleaned.messages())


def test_late_start_is_measured_against_the_requested_start_date():
    raw = ragged_prices()[["A", "B"]].copy()  # both start on 2024-01-01
    # Requested a year earlier: every ticker starts late, and the message says so.
    cleaned = data.clean_prices(raw, ["A", "B"], requested_start=dt.date(2023, 1, 2))
    assert list(cleaned.late_start) == ["A", "B"]
    text = " ".join(cleaned.messages())
    assert "A (2024-01-01), B (2024-01-01) start later than the requested start date" in text
    assert "(2023-01-02), so the analysis begins on 2024-01-01." in text
    # A start a weekend after the request (Sat 2023-12-30 -> Mon 2024-01-01) is not "late".
    assert not data.clean_prices(raw, ["A", "B"], requested_start=dt.date(2023, 12, 30)).late_start


def test_clean_prices_errors():
    raw = ragged_prices()
    with pytest.raises(data.DataError):
        data.clean_prices(raw, ["DEAD"])
    no_overlap = pd.DataFrame(
        {"X": [1.0, 2.0, np.nan, np.nan], "Y": [np.nan, np.nan, 3.0, 4.0]},
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    with pytest.raises(data.DataError, match="overlap"):
        data.clean_prices(no_overlap, ["X", "Y"])


def test_clean_prices_tolerates_holiday_at_window_start():
    raw = ragged_prices()[["A", "B"]].copy()
    raw.iloc[:2, 1] = np.nan  # B starts two days later (e.g. a holiday)
    assert data.clean_prices(raw, ["A", "B"]).late_start == {}


def test_currency_warning():
    assert data.currency_warning({"A": "USD", "B": "USD"}) is None
    mixed = data.currency_warning({"A": "USD", "B": "EUR", "C": None})
    assert "Mixed quote currencies" in mixed and "EUR: B" in mixed and "unknown for C" in mixed


def test_download_prices_extracts_adjusted_close(monkeypatch):
    import yfinance as yf

    idx = pd.date_range("2024-01-01", periods=3)
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["AAA", "BBB"]])
    fake = pd.DataFrame(np.arange(12.0).reshape(3, 4) + 1, index=idx, columns=cols)
    seen = {}

    def fake_download(tickers, **kwargs):
        seen.update(kwargs)
        return fake

    monkeypatch.setattr(yf, "download", fake_download)
    out = data.download_prices(["BBB", "AAA", "ZZZ"], dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    assert list(out.columns) == ["BBB", "AAA", "ZZZ"]
    assert out["ZZZ"].isna().all()
    assert seen["auto_adjust"] is True
    assert seen["end"] == dt.date(2024, 1, 4)  # yfinance's end date is exclusive


# --- analyze -----------------------------------------------------------------


def test_analyze_is_consistent_with_metric_functions():
    prices = synthetic_prices(["A", "B", "C", "BM"], dt.date(2021, 1, 1), dt.date(2023, 12, 31))
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    settings = Settings(risk_free=0.03, confidence=0.99, horizon_days=10)
    result = analyze(prices[["A", "B", "C"]], weights, prices["BM"], settings)

    port = result.portfolio
    assert result.portfolio_stats.annual_volatility == pytest.approx(
        metrics.annualized_volatility(port)
    )
    assert result.portfolio_stats.sortino == pytest.approx(metrics.sortino_ratio(port, 0.03))
    assert result.tail_risk.historical_var == pytest.approx(
        metrics.historical_var(port, 0.99) * np.sqrt(10)
    )
    assert result.contributions["risk_contribution"].sum() == pytest.approx(
        result.portfolio_stats.annual_volatility
    )
    assert np.isfinite(result.beta) and 0 < result.correlation_to_benchmark <= 1

    table = summary_table(result)
    assert {"Portfolio", "Benchmark"} <= set(table.columns)
    assert "Historical VaR (99.0% 10-day)" in table.index


def test_analyze_known_answers_for_levered_twin_assets():
    # Asset B always moves exactly 3x asset A, and A is also the benchmark. With 50/50
    # weights the portfolio return is 0.5 a + 0.5 (3 a) = 2 a, so every figure is known:
    a = np.random.default_rng(7).normal(0.0002, 0.005, 400)
    idx = pd.bdate_range("2022-01-03", periods=401)
    prices = pd.DataFrame(
        {
            "A": 100 * np.cumprod(np.r_[1.0, 1 + a]),
            "B": 100 * np.cumprod(np.r_[1.0, 1 + 3 * a]),
        },
        index=idx,
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})
    result = analyze(prices, weights, prices["A"], Settings(confidence=0.95, horizon_days=5))

    np.testing.assert_allclose(result.portfolio.to_numpy(), 2 * a)
    sigma_a = a.std(ddof=1) * np.sqrt(252)
    assert result.portfolio_stats.annual_volatility == pytest.approx(2 * sigma_a)
    assert result.beta == pytest.approx(2.0)
    assert result.correlation_to_benchmark == pytest.approx(1.0)
    # Sigma w = 0.5 * [s^2 + 3 s^2, 3 s^2 + 9 s^2] = [2 s^2, 6 s^2], sigma_p = 2 s
    # => RC_A = 0.5 * 2 s^2 / 2 s = 0.5 s, RC_B = 1.5 s, shares 25 % / 75 %.
    rc = result.contributions
    assert rc.loc["A", "risk_contribution"] == pytest.approx(0.5 * sigma_a)
    assert rc.loc["B", "risk_contribution"] == pytest.approx(1.5 * sigma_a)
    assert rc["risk_share"].tolist() == pytest.approx([0.25, 0.75])
    # Return contributions: w_i * mean_i * 252 = 0.5 * mean(a) * 252 and 0.5 * 3 * mean(a) * 252.
    assert rc["return_contribution"].tolist() == pytest.approx(
        [0.5 * a.mean() * 252, 1.5 * a.mean() * 252]
    )
    # Tail risk: the 1-day historical VaR of 2a, scaled by sqrt(5); overlapping check is unscaled.
    one_day = -np.quantile(2 * a, 0.05)
    assert result.tail_risk.historical_var == pytest.approx(one_day * np.sqrt(5))
    five_day = np.convolve(np.log1p(2 * a), np.ones(5), mode="valid")
    assert result.tail_risk.historical_var_overlapping == pytest.approx(
        -np.quantile(np.expm1(five_day), 0.05)
    )


def test_analyze_horizon_longer_than_sample_gives_nan_overlapping_var():
    prices = synthetic_prices(["A", "B"], dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    result = analyze(prices, pd.Series({"A": 0.5, "B": 0.5}), None, Settings(horizon_days=60))
    assert np.isnan(result.tail_risk.historical_var_overlapping)
    assert np.isfinite(result.tail_risk.historical_var)


def test_analyze_without_benchmark_and_charts_build():
    prices = synthetic_prices(["A", "B"], dt.date(2022, 1, 1), dt.date(2022, 12, 31))
    weights = pd.Series({"A": 0.6, "B": 0.4})
    result = analyze(prices, weights, None, Settings())
    assert result.benchmark is None and np.isnan(result.beta)
    assert "Benchmark" not in summary_table(result).columns
    # Chart builders should accept the analysis output without error.
    charts.cumulative_performance(result.portfolio, None)
    charts.drawdown_chart(metrics.drawdown_series(result.portfolio), None)
    charts.correlation_heatmap(result.correlation_matrix)
    charts.weight_vs_risk_chart(result.contributions)
    charts.return_contribution_chart(result.contributions)


def test_frontier_sharpe_uses_the_same_risk_free_convention_as_the_headline():
    prices = synthetic_prices(["A", "B", "C"], dt.date(2021, 1, 1), dt.date(2023, 12, 31))
    returns = metrics.simple_returns(prices)
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    result = analyze(prices, weights, None, Settings(risk_free=0.05))

    inputs = mean_variance_inputs(returns, 0.05)
    # Independent value: 252 x the de-compounded daily rate, not the raw 5 %.
    assert inputs.risk_free == pytest.approx(252 * (1.05 ** (1 / 252) - 1), rel=1e-12)
    assert inputs.risk_free == pytest.approx(0.0487949, abs=1e-7)
    w = weights[inputs.mu.index].to_numpy()
    frontier_style = (w @ inputs.mu.to_numpy() - inputs.risk_free) / np.sqrt(
        w @ inputs.cov.to_numpy() @ w
    )
    assert frontier_style == pytest.approx(result.portfolio_stats.sharpe, rel=1e-10)
