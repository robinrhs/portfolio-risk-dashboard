# portfolio-risk-dashboard

**Work in progress.** A Streamlit web app (plus a small command-line report) that will show the risk and performance of a stock/ETF portfolio: return, volatility, Sharpe, Sortino, drawdown, beta, VaR and Expected Shortfall, risk contributions and an efficient frontier.

> Educational project built on historical data. **Not investment advice**, not a forecast.

## Status

This repository is being integrated step by step through pull requests. `main` started as a project skeleton without any analytics code:

- Python package layout (`src/portfolio_risk/`) managed with [uv](https://docs.astral.sh/uv/), dependencies pinned in `uv.lock`
- `requirements.txt` exported from `uv.lock` (for Streamlit Community Cloud later)
- GitHub Actions CI on Python 3.11 and 3.12: `ruff check`, `ruff format --check`, a check that `requirements.txt` matches `uv.lock`, and `pytest`
- one smoke test that imports the package
- MIT license

```bash
uv sync
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

## Planned pull requests

The features are developed on separate branches and reach `main` in this order:

1. **Core analytics:** risk and return metrics (CAGR, volatility, Sharpe, Sortino, max drawdown, beta, historical and normal VaR/ES), weight cleaning, Euler risk contributions and a long-only efficient frontier, with known-answer tests.
2. **Data pipeline and CLI:** Yahoo Finance download with price cleaning, one `analyze()` entry point and a `portfolio-risk` terminal report.
3. **Streamlit dashboard:** charts, the app itself and offline tests (synthetic prices, no network).
4. **Documentation:** the full README with method, real example output, limitations, deploy guide and a screenshot. It replaces this placeholder.
5. **Input validation** fixes from a code audit.
6. **Finance consistency** fixes from the same audit.

Until step 4 is merged, this README is a short placeholder and may lag behind the code: the merged pull requests show what is already on `main`. There is no live demo yet.

## License

[MIT](LICENSE) © 2026 Robin-Philip Rehschuh
