"""Command-line report, offline: the Yahoo download is replaced by synthetic prices."""

import pytest
from conftest import synthetic_currencies, synthetic_prices

from portfolio_risk import cli, data


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(data, "download_prices", synthetic_prices)
    monkeypatch.setattr(data, "fetch_currencies", synthetic_currencies)


def test_report_prints_summary_contributions_and_drawdown(capsys):
    cli.main(["--start", "2021-01-01", "--end", "2023-12-31"])
    out = capsys.readouterr().out
    assert "Period: 2021-01-04 to 2023-12-29" in out
    assert "Weights: VTI 30.0%, VEA 20.0%, VWO 10.0%, BND 25.0%, GLD 10.0%, VNQ 5.0%" in out
    for row in ("Annualised return (CAGR)", "Sortino ratio", "Historical VaR (95.0% 1-day)"):
        assert row in out
    assert "risk_share" in out and "Max drawdown: peak" in out
    assert "sqrt(" not in out  # no horizon note for the 1-day default


def test_report_notes_bad_tickers_and_horizon_scaling(tmp_path, capsys):
    csv = tmp_path / "p.csv"
    csv.write_text("ticker,weight\nAAA,60\nBADX,20\nCCC.DE,20\n")
    cli.main(["--csv", str(csv), "--start", "2022-01-01", "--end", "2023-06-30", "--horizon", "10"])
    out = capsys.readouterr().out
    assert "NOTE: No price data found for BADX" in out
    assert "NOTE: Mixed quote currencies" in out
    assert "Weights: AAA 75.0%, CCC.DE 25.0%" in out
    assert "1-day figures x sqrt(10) (approximation)" in out
    assert "overlapping 10-day returns" in out


def test_short_window_is_refused_like_in_the_app():
    with pytest.raises(SystemExit, match=f"at least {data.MIN_OBSERVATIONS} days"):
        cli.main(["--start", "2025-10-20", "--end", "2025-10-31"])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--start", "2025-10-31", "--end", "2025-01-01"], "must be before --end"),
        (["--confidence", "95"], "between 0 and 1"),
        (["--horizon", "0"], "at least 1 trading day"),
    ],
)
def test_invalid_arguments_exit_with_usage_error(argv, message, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2  # argparse usage error, no traceback
    assert message in capsys.readouterr().err


def test_data_errors_exit_with_one_line_message(tmp_path):
    with pytest.raises(SystemExit, match="Error: None of the tickers returned any price data"):
        cli.main(["--benchmark", "BADBENCH", "--csv", str(write_csv(tmp_path, "BADX"))])
    with pytest.raises(SystemExit, match="Error: .*No such file"):
        cli.main(["--csv", str(tmp_path / "missing.csv")])


def write_csv(folder, ticker: str):
    path = folder / "one.csv"
    path.write_text(f"ticker,weight\n{ticker},1\n")
    return path
