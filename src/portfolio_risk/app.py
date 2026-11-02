"""Streamlit front end. Only layout and input handling live here; all maths is imported."""

from __future__ import annotations

import datetime as dt
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from portfolio_risk import charts, data, metrics, optimize
from portfolio_risk.analysis import Analysis, Settings, TailRisk, analyze, summary_table
from portfolio_risk.portfolio import (
    DEFAULT_BENCHMARK,
    clean_portfolio,
    default_portfolio_table,
    read_portfolio_csv,
)

PRICE_CACHE_TTL = 6 * 60 * 60  # seconds; daily data does not need fresher prices
EXAMPLE_CSV_CANDIDATES = [
    Path.cwd() / "data" / "example_portfolio.csv",
    Path(__file__).resolve().parents[2] / "data" / "example_portfolio.csv",
]

# Plain-English tooltips shown next to every metric (the "?" icon).
HELP = {
    "cagr": "Compound annual growth rate: the constant yearly return that turns the "
    "starting value into the ending value. Geometric, so it is what an investor actually earned.",
    "total": "Cumulative return over the whole period, dividends reinvested.",
    "vol": "Annualised standard deviation of daily returns: how much returns typically "
    "swing around their average. Daily volatility × √252.",
    "sharpe": "Excess return over the risk-free rate per unit of total volatility. "
    "Higher is better; it treats upside and downside swings as equally 'risky'.",
    "sortino": "Like Sharpe, but only counts volatility BELOW a minimum acceptable return "
    "(here: the risk-free rate). Rewards portfolios whose swings are mostly upward.",
    "mdd": "Largest peak-to-trough fall of the portfolio value in the period: the worst "
    "loss an investor who bought at the top would have sat through.",
    "beta": "Sensitivity to the benchmark: Cov(portfolio, benchmark) / Var(benchmark). "
    "Beta 0.8 means the portfolio moved about 0.8 % on average for a 1 % benchmark move.",
    "corr": "Correlation of daily portfolio and benchmark returns (-1 to +1). Near +1 "
    "means the portfolio behaves very much like the benchmark.",
    "hvar": "Historical VaR: the loss that was exceeded on only (1 − confidence) of past "
    "days. No distribution assumed. It says nothing about how bad losses beyond it get.",
    "hes": "Historical Expected Shortfall: the AVERAGE loss on the days that were worse "
    "than the VaR. Describes the tail that VaR ignores.",
    "pvar": "Parametric VaR assuming normally distributed returns: z × σ − μ. Real returns "
    "have fatter tails than the normal, so this often understates extreme losses.",
    "pes": "Parametric Expected Shortfall under the normal distribution: the average loss "
    "beyond the normal VaR, σ·φ(z)/(1 − c) − μ.",
}


@dataclass(frozen=True)
class Inputs:
    holdings: pd.DataFrame
    benchmark: str
    start: dt.date
    end: dt.date
    settings: Settings
    portfolio_value: float


# ---------------------------------------------------------------------------
# Cached data access (the only place that talks to Yahoo Finance)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=PRICE_CACHE_TTL, show_spinner="Downloading prices from Yahoo Finance ...")
def load_market_data(
    tickers: tuple[str, ...], start: dt.date, end: dt.date
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Prices and quote currencies, fetched at the same time to keep the first load short."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        currencies = pool.submit(data.fetch_currencies, list(tickers))
        prices = data.download_prices(list(tickers), start, end)
        return prices, currencies.result()


@st.cache_data(ttl=PRICE_CACHE_TTL, show_spinner="Optimising ...")
def compute_frontier(asset_returns: pd.DataFrame, risk_free: float) -> dict:
    # Annualised historical estimates of expected return and covariance.
    mu = asset_returns.mean() * metrics.TRADING_DAYS
    cov = asset_returns.cov() * metrics.TRADING_DAYS
    return {
        "mu": mu,
        "cov": cov,
        "frontier": optimize.efficient_frontier(mu, cov, n_points=30, risk_free=risk_free),
        "min_var": optimize.min_variance_portfolio(mu, cov, risk_free),
        "max_sharpe": optimize.max_sharpe_portfolio(mu, cov, risk_free),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def pct(x: float, digits: int = 2) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.{digits}%}"


def num(x: float, digits: int = 2) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.{digits}f}"


def date_str(d: pd.Timestamp | None) -> str:
    return "n/a" if d is None else f"{d:%d %b %Y}"


def example_csv_bytes() -> bytes:
    for path in EXAMPLE_CSV_CANDIDATES:
        if path.exists():
            return path.read_bytes()
    return default_portfolio_table().to_csv(index=False).encode()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def sidebar() -> Inputs:
    sb = st.sidebar
    sb.header("Portfolio")
    source = sb.radio("Input", ["Edit table", "Upload CSV"], horizontal=True, key="source")
    holdings = default_portfolio_table()
    if source == "Edit table":
        holdings = sb.data_editor(
            default_portfolio_table(),
            key="holdings",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn(
                    "Ticker", help="Yahoo Finance symbol, e.g. VTI, AAPL, EUNL.DE"
                ),
                "weight": st.column_config.NumberColumn(
                    "Weight",
                    min_value=0.0,
                    format="%.1f",
                    help="Any unit (%, fractions or money amounts). Normalised to 100 %.",
                ),
            },
        )
    else:
        uploaded = sb.file_uploader("CSV with columns ticker, weight", type="csv")
        if uploaded is not None:
            try:
                holdings = read_portfolio_csv(uploaded)
            except (ValueError, pd.errors.ParserError) as exc:
                sb.error(f"Could not read CSV: {exc}")
        else:
            sb.caption("No file uploaded yet: showing the default portfolio.")
        sb.download_button(
            "Download example CSV",
            example_csv_bytes(),
            file_name="example_portfolio.csv",
            mime="text/csv",
        )
    sb.caption("Weights are normalised to sum to 100 %. Long-only.")

    sb.header("Settings")
    benchmark = sb.text_input(
        "Benchmark ticker",
        DEFAULT_BENCHMARK,
        help="URTH = iShares MSCI World ETF. Alternatives: SPY (S&P 500), ACWI, EUNL.DE.",
    )
    today = dt.date.today()
    # Stacked, not side by side: two date fields do not fit next to each other in the sidebar.
    start = sb.date_input(
        "Start", today - dt.timedelta(days=5 * 365), max_value=today, format="YYYY-MM-DD"
    )
    end = sb.date_input("End", today, max_value=today, format="YYYY-MM-DD")
    rf = sb.number_input(
        "Risk-free rate (% p.a.)",
        min_value=-2.0,
        max_value=20.0,
        value=2.0,
        step=0.25,
        help="Constant annual rate used for Sharpe, and as the minimum acceptable "
        "return (MAR) for Sortino.",
    )
    confidence = sb.select_slider(
        "VaR / ES confidence",
        options=[0.90, 0.95, 0.975, 0.99],
        value=0.95,
        format_func=lambda c: f"{c:.1%}",
    )
    horizon = sb.number_input(
        "VaR horizon (trading days)",
        min_value=1,
        max_value=60,
        value=1,
        help="Risk is estimated on 1-day returns. Longer horizons use √time scaling, "
        "an approximation that assumes independent, normally distributed returns.",
    )
    value = sb.number_input(
        "Portfolio value (for VaR in money)", min_value=0.0, value=100_000.0, step=10_000.0
    )
    return Inputs(
        holdings=holdings,
        benchmark=benchmark.strip().upper(),
        start=start,
        end=end,
        settings=Settings(risk_free=rf / 100, confidence=confidence, horizon_days=int(horizon)),
        portfolio_value=value,
    )


# ---------------------------------------------------------------------------
# Main page sections
# ---------------------------------------------------------------------------


def headline_metrics(result: Analysis) -> None:
    p, b = result.portfolio_stats, result.benchmark_stats

    def vs_bench(attr: str, as_pct: bool = True) -> str | None:
        """Difference portfolio minus benchmark for one statistic, e.g. "+1.20% vs benchmark"."""
        if b is None:
            return None
        diff = getattr(p, attr) - getattr(b, attr)
        if not math.isfinite(diff):
            return None
        # Percentage-point difference for percentages, plain difference for ratios.
        return f"{diff * 100:+.2f} pp" if as_pct else f"{diff:+.2f}"

    row1 = st.columns(4)
    row1[0].metric(
        "Return p.a. (CAGR)",
        pct(p.annual_return),
        vs_bench("annual_return"),
        help=HELP["cagr"],
        border=True,
    )
    row1[1].metric(
        "Volatility p.a.",
        pct(p.annual_volatility),
        vs_bench("annual_volatility"),
        delta_color="inverse",  # more volatility than the benchmark is shown in red
        help=HELP["vol"],
        border=True,
    )
    row1[2].metric(
        "Sharpe",
        num(p.sharpe),
        vs_bench("sharpe", as_pct=False),
        help=HELP["sharpe"],
        border=True,
    )
    row1[3].metric(
        "Sortino",
        num(p.sortino),
        vs_bench("sortino", as_pct=False),
        help=HELP["sortino"],
        border=True,
    )

    row2 = st.columns(4)
    mdd = p.max_drawdown
    row2[0].metric(
        "Max drawdown",
        pct(mdd.depth),
        None if b is None else f"{(mdd.depth - b.max_drawdown.depth) * 100:+.2f} pp",
        help=HELP["mdd"],
        border=True,
    )
    row2[1].metric("Total return", pct(p.total_return), help=HELP["total"], border=True)
    row2[2].metric("Beta", num(result.beta), help=HELP["beta"], border=True)
    row2[3].metric(
        "Correlation",
        num(result.correlation_to_benchmark),
        help=HELP["corr"],
        border=True,
    )
    recovery = (
        f"recovered {date_str(mdd.recovery_date)}" if mdd.recovery_date else "not yet recovered"
    )
    notes = []
    if b is not None:
        notes.append(
            "Small figures under a value: difference to the benchmark (pp = percentage points)."
        )
    if mdd.trough_date is not None:
        notes.append(
            f"Max drawdown: peak {date_str(mdd.peak_date)} → trough "
            f"{date_str(mdd.trough_date)}, {recovery}."
        )
    if notes:
        st.caption(" ".join(notes))


def tail_risk_metrics(result: Analysis, portfolio_value: float) -> None:
    t = result.tail_risk
    horizon = f"{t.horizon_days}-day" + (
        " (√time-scaled, approximation)" if t.horizon_days > 1 else ""
    )
    st.subheader(f"Tail risk · {t.confidence:.1%} confidence · {horizon}")
    cols = st.columns(4)
    items = [
        ("Historical VaR", t.historical_var, "hvar"),
        ("Historical ES", t.historical_es, "hes"),
        ("Normal VaR", t.parametric_var, "pvar"),
        ("Normal ES", t.parametric_es, "pes"),
    ]
    for col, (label, value, key) in zip(cols, items, strict=True):
        money = f"≈ {value * portfolio_value:,.0f} on {portfolio_value:,.0f}"
        col.metric(
            label,
            pct(value),
            money,
            delta_color="off",
            delta_arrow="off",
            help=HELP[key],
            border=True,
        )
    st.caption(tail_risk_caption(t))


def tail_risk_caption(t: TailRisk) -> str:
    """How to read the tail-risk row; for h > 1 it must not pretend to be an h-day statistic."""
    tail = f"{1 - t.confidence:.1%}"
    if t.horizon_days == 1:
        return (
            f"Read: on {tail} of past days the 1-day loss was larger than the VaR; when it "
            "was, the average loss was the ES. Based on past data only: not a forecast."
        )
    h = t.horizon_days
    return (
        f"Read: on {tail} of past days the 1-day loss was larger than the 1-day VaR. The "
        f"{h}-day figures shown are those 1-day values × √{h}, an approximation, not "
        f"measured {h}-day losses. For comparison, the historical VaR of overlapping "
        f"{h}-day returns in this sample is {pct(t.historical_var_overlapping)} (overlapping "
        "windows share days, so this estimate is noisy). Based on past data only: not a "
        "forecast."
    )


def performance_tab(result: Analysis) -> None:
    st.markdown("**Growth of 100** (dividends reinvested, daily rebalanced to target weights)")
    st.plotly_chart(charts.cumulative_performance(result.portfolio, result.benchmark))
    st.markdown("**Drawdown** (percentage below the previous high)")
    bench_dd = None if result.benchmark is None else metrics.drawdown_series(result.benchmark)
    st.plotly_chart(charts.drawdown_chart(metrics.drawdown_series(result.portfolio), bench_dd))


def risk_tab(result: Analysis) -> None:
    t = result.tail_risk
    one_day_hist = metrics.historical_var(result.portfolio, t.confidence)
    one_day_param = metrics.parametric_var(result.portfolio, t.confidence)
    st.markdown("**Distribution of daily portfolio returns** with 1-day VaR thresholds")
    st.plotly_chart(
        charts.return_histogram(result.portfolio, one_day_hist, one_day_param, t.confidence)
    )
    st.info(
        "VaR is a threshold, not a worst case: it says how often a loss of this size was "
        "exceeded, not by how much. Expected Shortfall answers the second question. If the "
        "historical figures are well above the normal ones, the return distribution has "
        "fatter tails than the normal distribution assumes.",
        icon=":material/info:",
    )
    st.markdown("**Correlation of daily returns**")
    st.caption(
        "Diversification works through correlation: the lower the correlations, the lower "
        "portfolio volatility is relative to the weighted average of single-asset volatilities."
    )
    st.plotly_chart(charts.correlation_heatmap(result.correlation_matrix))


def contributions_tab(result: Analysis) -> None:
    c = result.contributions
    port_vol = result.portfolio_stats.annual_volatility
    left, right = st.columns(2)
    with left:
        st.markdown("**Weight vs share of portfolio risk**")
        st.caption(
            f"Euler risk contributions w·(Σw)/σ add up exactly to the portfolio volatility "
            f"of {pct(port_vol)}."
        )
        st.plotly_chart(charts.weight_vs_risk_chart(c))
    with right:
        st.markdown("**Contribution to annual mean return**")
        st.caption(
            "w × average annual return of each position. Sums to the arithmetic mean "
            f"return ({pct(c['return_contribution'].sum())}), which is above the CAGR "
            "because of volatility drag."
        )
        st.plotly_chart(charts.return_contribution_chart(c))
    table = c.rename(
        columns={
            "weight": "Weight",
            "return_contribution": "Return contribution (p.a.)",
            "risk_contribution": "Risk contribution (vol, p.a.)",
            "risk_share": "Share of risk",
        }
    )
    st.dataframe(table.style.format("{:.2%}"))


def frontier_tab(result: Analysis, settings: Settings) -> None:
    returns = result.asset_returns
    if returns.shape[1] < 2:
        st.info("Add at least two positions to compute an efficient frontier.")
        return
    try:
        f = compute_frontier(returns, settings.risk_free)
    except RuntimeError as exc:
        st.warning(f"The optimiser did not converge: {exc}")
        return
    weights = result.contributions["weight"]
    current = (
        optimize.portfolio_volatility(weights.to_numpy(), f["cov"].to_numpy()),
        optimize.portfolio_return(weights.to_numpy(), f["mu"].to_numpy()),
    )
    # Single-asset volatility = square root of the covariance matrix diagonal.
    assets = pd.DataFrame({"return": f["mu"], "volatility": np.sqrt(np.diag(f["cov"]))})
    points = {
        "Min variance": (f["min_var"].volatility, f["min_var"].expected_return),
        "Max Sharpe": (f["max_sharpe"].volatility, f["max_sharpe"].expected_return),
        "Your portfolio": current,
    }
    st.warning(
        "Illustration, not a recommendation. Expected returns are simply the historical "
        "averages of this period; mean-variance weights react very strongly to them.",
        icon=":material/warning:",
    )
    st.plotly_chart(charts.frontier_chart(f["frontier"], assets, points))
    table = pd.DataFrame(
        {
            "Your portfolio": weights,
            "Min variance": f["min_var"].weights,
            "Max Sharpe": f["max_sharpe"].weights,
        }
    )
    stats = pd.DataFrame(
        {
            "Expected return (hist. mean)": [
                current[1],
                f["min_var"].expected_return,
                f["max_sharpe"].expected_return,
            ],
            "Volatility": [current[0], f["min_var"].volatility, f["max_sharpe"].volatility],
            "Sharpe": [
                (current[1] - settings.risk_free) / current[0],
                f["min_var"].sharpe,
                f["max_sharpe"].sharpe,
            ],
        },
        index=["Your portfolio", "Min variance", "Max Sharpe"],
    )
    left, right = st.columns(2)
    left.markdown("**Weights** (long-only, sum to 100 %)")
    left.dataframe(table.style.format("{:.1%}"))
    right.markdown("**Risk / return** (annualised)")
    right.dataframe(
        stats.style.format(
            {"Expected return (hist. mean)": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}"}
        )
    )


def data_tab(result: Analysis, prices: pd.DataFrame) -> None:
    summary = summary_table(result)
    st.markdown("**Summary**")
    st.dataframe(summary.style.format("{:.4f}", na_rep="–"))
    col1, col2 = st.columns(2)
    col1.download_button(
        "Download summary (CSV)", summary.to_csv().encode(), "portfolio_summary.csv", "text/csv"
    )
    col2.download_button(
        "Download prices (CSV)", prices.to_csv().encode(), "adjusted_prices.csv", "text/csv"
    )
    st.markdown(f"**Adjusted close prices** ({len(prices)} trading days, last 10 shown)")
    st.dataframe(prices.tail(10).style.format("{:.2f}"))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Portfolio Risk Dashboard",
        page_icon=":material/monitoring:",
        layout="wide",
    )
    inputs = sidebar()
    st.title("Portfolio Risk Dashboard")
    st.caption(
        "Risk and performance of a stock/ETF portfolio from historical Yahoo Finance data. "
        "Educational tool. Not investment advice."
    )

    if inputs.start >= inputs.end:
        st.error("The start date must be before the end date.")
        st.stop()
    try:
        portfolio = clean_portfolio(inputs.holdings)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    for note in portfolio.notes:
        st.info(note, icon=":material/balance:")

    positions = list(portfolio.weights.index)
    benchmark = inputs.benchmark or None
    tickers = tuple(dict.fromkeys(positions + ([benchmark] if benchmark else [])))

    try:
        raw, currencies = load_market_data(tickers, inputs.start, inputs.end)
        cleaned = data.clean_prices(raw, list(tickers))
    except data.DataError as exc:
        st.error(str(exc))
        st.stop()
    for message in cleaned.messages():
        st.warning(message, icon=":material/warning:")

    valid = [t for t in positions if t in cleaned.prices.columns]
    if not valid:
        st.error("None of the portfolio tickers has usable price data.")
        st.stop()
    weights = portfolio.weights[valid]
    if len(valid) < len(positions):
        weights = weights / weights.sum()
        st.warning(
            "Weights of the remaining positions were rescaled to 100 %: "
            + ", ".join(f"{t} {w:.1%}" for t, w in weights.items()),
            icon=":material/balance:",
        )
    if len(cleaned.prices) < data.MIN_OBSERVATIONS:
        st.error(
            f"Only {len(cleaned.prices)} overlapping trading days. Choose a longer period "
            f"(at least {data.MIN_OBSERVATIONS} days are needed for meaningful statistics)."
        )
        st.stop()

    # Only check currencies of tickers that actually have data.
    currency_note = data.currency_warning({t: currencies.get(t) for t in cleaned.prices.columns})
    if currency_note:
        st.warning(currency_note, icon=":material/currency_exchange:")

    bench_prices = cleaned.prices[benchmark] if benchmark in cleaned.prices.columns else None
    prices = cleaned.prices[valid]
    result = analyze(prices, weights, bench_prices, inputs.settings)
    first, last = result.portfolio.index[0], result.portfolio.index[-1]
    st.caption(
        f"Period analysed: {date_str(first)} – {date_str(last)} · {len(result.portfolio)} daily "
        f"returns · benchmark: {benchmark if bench_prices is not None else 'none'}"
    )

    headline_metrics(result)
    tail_risk_metrics(result, inputs.portfolio_value)

    tabs = st.tabs(["Performance", "Risk", "Contributions", "Efficient frontier", "Data"])
    with tabs[0]:
        performance_tab(result)
    with tabs[1]:
        risk_tab(result)
    with tabs[2]:
        contributions_tab(result)
    with tabs[3]:
        frontier_tab(result, inputs.settings)
    with tabs[4]:
        data_tab(result, cleaned.prices)

    st.divider()
    st.caption(
        "Assumptions: daily rebalancing to target weights, no transaction costs or taxes, "
        "constant risk-free rate, returns in each asset's quote currency. Past performance "
        "does not predict future results. Data: Yahoo Finance via the unofficial yfinance "
        "library. Not investment advice."
    )
