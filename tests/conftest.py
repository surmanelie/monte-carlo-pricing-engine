"""Shared fixtures and tolerances.

Statistical assertions are expressed in standard errors with fixed seeds, so the
suite is deterministic: ``|MC - ref| < TOL_SE * SE``.
"""

from __future__ import annotations

import pytest

from mcengine.models.gbm import GBM
from mcengine.products.european import EuropeanOption

#: Tolerance in standard errors for Monte Carlo vs reference comparisons.
TOL_SE = 4.0


@pytest.fixture
def bs_model() -> GBM:
    """The textbook case S0=100, r=5 %, sigma=20 %."""
    return GBM(s0=100.0, r=0.05, sigma=0.2)


@pytest.fixture
def atm_call() -> EuropeanOption:
    return EuropeanOption(strike=100.0, maturity=1.0, option_type="call")


@pytest.fixture
def atm_put() -> EuropeanOption:
    return EuropeanOption(strike=100.0, maturity=1.0, option_type="put")
