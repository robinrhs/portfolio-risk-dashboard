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
