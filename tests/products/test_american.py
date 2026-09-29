from __future__ import annotations

import numpy as np
import pytest

from mcengine.products.american import AmericanOption


def test_exercise_values_and_payoff() -> None:
    put = AmericanOption(40.0, 1.0, "put", 4)
    np.testing.assert_allclose(put.exercise_value(np.array([30.0, 50.0])), [10.0, 0.0])
    np.testing.assert_allclose(put.monitoring_times(), [0.25, 0.5, 0.75, 1.0])
    times = np.array([0.0, 0.5, 1.0])
    assert put.payoff(np.array([[40.0, 20.0, 35.0]]), times)[0] == 5.0
    assert "American put" in put.label


def test_invalid() -> None:
    with pytest.raises(ValueError, match="n_exercise"):
        AmericanOption(40.0, 1.0, "put", 0)
