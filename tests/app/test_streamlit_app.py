"""Smoke tests of the Streamlit dashboard with Streamlit's AppTest harness."""

from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")


def _app() -> object:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.select_slider[0].set_value(2**12)
    at.run()
    return at


def test_default_view_runs_without_exception() -> None:
    at = _app()
    assert not at.exception
    assert at.metric[0].label == "Monte Carlo price"
    assert "Black-Scholes" in at.metric[2].help


@pytest.mark.parametrize(
    ("model", "product"),
    [("heston", "european"), ("merton", "asian"), ("gbm", "barrier"), ("gbm", "american")],
)
def test_models_and_products(model: str, product: str) -> None:
    at = _app()
    at.sidebar.selectbox[0].set_value(model)
    at.run()
    at.sidebar.selectbox[1].set_value(product)
    at.run()
    assert not at.exception
