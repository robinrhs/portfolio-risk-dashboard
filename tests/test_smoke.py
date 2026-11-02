"""Smoke test: the package imports without network access or optional UI code."""

import portfolio_risk


def test_package_imports_and_has_a_version():
    assert portfolio_risk.__version__ == "0.1.0"
