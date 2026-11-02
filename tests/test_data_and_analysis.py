import datetime as dt

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_prices

from portfolio_risk import charts, data, metrics
from portfolio_risk.analysis import Settings, analyze, summary_table

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
