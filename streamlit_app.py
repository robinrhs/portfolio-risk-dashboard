"""Entry point for Streamlit Community Cloud and local use: `streamlit run streamlit_app.py`."""

import sys
from pathlib import Path

# Make the src/ package importable even when the project was not pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from portfolio_risk.app import main  # noqa: E402

main()
