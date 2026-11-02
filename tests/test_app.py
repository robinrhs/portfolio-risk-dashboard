"""Run the real Streamlit script headlessly with streamlit.testing (offline, synthetic prices)."""

from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from conftest import synthetic_currencies, synthetic_prices
from streamlit.testing.v1 import AppTest

from portfolio_risk import app as app_module
from portfolio_risk import data

APP_FILE = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


DOWNLOADS: list[tuple[str, ...]] = []


def fake_download(tickers, start, end):
    DOWNLOADS.append(tuple(tickers))
    return synthetic_prices(tickers, start, end)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Replace the two network functions with synthetic data and start with a clean cache."""
    monkeypatch.setattr(data, "download_prices", fake_download)
    monkeypatch.setattr(data, "fetch_currencies", synthetic_currencies)
    DOWNLOADS.clear()
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def run_app() -> AppTest:
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def texts(elements) -> str:
    return " ".join(str(e.value) for e in elements)


def test_default_portfolio_renders_all_sections():
    at = run_app()
    assert DOWNLOADS == [("VTI", "VEA", "VWO", "BND", "GLD", "VNQ", "URTH")]  # mock was used
    labels = [m.label for m in at.metric]
    for expected in (
        "Return p.a. (CAGR)",
        "Volatility p.a.",
        "Sharpe",
        "Sortino",
        "Max drawdown",
        "Beta",
        "Correlation",
        "Historical VaR",
        "Historical ES",
        "Normal VaR",
        "Normal ES",
    ):
        assert expected in labels
    assert all(m.help for m in at.metric)  # every metric has a tooltip
    assert [t.label for t in at.tabs] == [
        "Performance",
        "Risk",
        "Contributions",
        "Efficient frontier",
        "Data",
    ]
    assert not at.error
    assert not at.warning or "Mixed" not in texts(at.warning)  # default is all-USD


def test_invalid_benchmark_and_currency_mix_warn():
    at = run_app()
    at.sidebar.text_input[0].set_value("BADTICKER").run()
    assert not at.exception
    assert "No price data found for BADTICKER" in texts(at.warning)
    beta = next(m for m in at.metric if m.label == "Beta")
    assert beta.value == "n/a"  # app still renders, benchmark figures are just unavailable

    at.sidebar.text_input[0].set_value("EUNL.DE").run()
    assert "Mixed quote currencies" in texts(at.warning)


def test_weights_are_normalised_with_visible_note(monkeypatch):
    table = pd.DataFrame({"ticker": ["AAA", "BBB", "BADX"], "weight": [3.0, 1.0, 1.0]})
    monkeypatch.setattr(app_module, "default_portfolio_table", lambda: table.copy())
    at = run_app()
    assert "rescaled proportionally" in texts(at.info)
    # BADX has no data: it is dropped and the remaining weights are rescaled again.
    assert "AAA 75.0%" in texts(at.warning)


def test_horizon_scaling_is_labelled_as_approximation():
    at = run_app()
    assert "of past days the 1-day loss was larger than the VaR" in texts(at.caption)
    horizon = next(n for n in at.sidebar.number_input if "horizon" in n.label)
    horizon.set_value(10).run()
    assert not at.exception
    assert "approximation" in texts(at.subheader)
    captions = texts(at.caption)
    # The caption must say the 10-day figure is scaled, not a measured 10-day statistic,
    # and show the historical VaR of overlapping 10-day returns next to it.
    assert "1-day values × √10, an approximation, not measured 10-day losses" in captions
    assert "historical VaR of overlapping 10-day returns" in captions
    assert "of 10-day periods" not in captions


def test_start_after_end_shows_error():
    import datetime as dt

    at = run_app()
    at.sidebar.date_input[0].set_value(dt.date.today()).run()
    assert "start date must be before" in texts(at.error)


def test_too_short_window_shows_minimum_observations_error():
    import datetime as dt

    at = run_app()
    end = at.sidebar.date_input[1].value
    at.sidebar.date_input[0].set_value(end - dt.timedelta(days=30)).run()
    assert not at.exception
    errors = texts(at.error)
    assert "overlapping trading days" in errors
    assert f"at least {data.MIN_OBSERVATIONS} days" in errors
    assert not at.metric  # the app stops before showing any annualised figure


def test_csv_upload_is_used_and_bad_csv_is_reported():
    at = run_app()
    at.sidebar.radio(key="source").set_value("Upload CSV").run()
    assert "No file uploaded yet" in texts(at.sidebar.caption)

    upload = b"ticker,weight\nAAA,50\nCCC,30\nDDD,20\n"
    at.sidebar.file_uploader[0].set_value(("mine.csv", upload, "text/csv")).run()
    assert not at.exception
    assert DOWNLOADS[-1] == ("AAA", "CCC", "DDD", "URTH")  # uploaded tickers were analysed
    assert not at.error

    at.sidebar.file_uploader[0].set_value(("bad.csv", b"symbol,amount\nAAA,1\n", "text/csv"))
    at.run()
    assert "Could not read CSV" in texts(at.error)
    assert not at.exception  # falls back to the default portfolio instead of crashing


def test_flat_prices_show_an_error_instead_of_a_traceback(monkeypatch):
    def flat_download(tickers, start, end):
        index = pd.bdate_range(start, end)
        return pd.DataFrame({t: 100.0 for t in tickers}, index=index)

    monkeypatch.setattr(data, "download_prices", flat_download)
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert "Portfolio volatility is zero" in texts(at.error)
    assert not at.metric


def test_frontier_tab_sharpe_matches_the_headline_sharpe_convention():
    import datetime as dt

    from portfolio_risk import metrics
    from portfolio_risk.portfolio import portfolio_returns

    prices = synthetic_prices(["AAA", "CCC", "DDD"], dt.date(2021, 1, 1), dt.date(2023, 12, 31))
    returns = metrics.simple_returns(prices)
    f = app_module.compute_frontier(returns, 0.05)
    for point in (f["max_sharpe"], f["min_var"]):
        headline = metrics.sharpe_ratio(portfolio_returns(returns, point.weights), 0.05)
        assert point.sharpe == pytest.approx(headline, rel=1e-8)


def test_frontier_tab_skips_max_sharpe_when_no_asset_beats_the_risk_free_rate(monkeypatch):
    import numpy as np

    def falling_download(tickers, start, end):
        prices = synthetic_prices(tickers, start, end)
        drift = np.exp(-0.002 * np.arange(len(prices)))  # about -40 % a year on top
        return prices.mul(drift, axis=0)

    monkeypatch.setattr(data, "download_prices", falling_download)
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert "max-Sharpe portfolio is not meaningful" in texts(at.info)
    assert not at.error
