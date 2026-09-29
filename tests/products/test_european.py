from __future__ import annotations

import numpy as np
import pytest

from mcengine.products.base import time_indices, uniform_monitoring
from mcengine.products.european import DigitalOption, EuropeanOption


def test_vanilla_payoffs() -> None:
    times = np.array([0.0, 1.0])
    paths = np.array([[100.0, 120.0], [100.0, 80.0]])
    np.testing.assert_array_equal(EuropeanOption(100.0, 1.0, "call").payoff(paths, times), [20, 0])
    np.testing.assert_array_equal(EuropeanOption(100.0, 1.0, "put").payoff(paths, times), [0, 20])


def test_digital_payoffs() -> None:
    times = np.array([0.0, 0.5, 1.0])
    paths = np.array([[100.0, 50.0, 120.0], [100.0, 150.0, 80.0]])
    call = DigitalOption(100.0, 1.0, "call", payout=2.0)
    np.testing.assert_array_equal(call.payoff(paths, times), [2.0, 0.0])
    np.testing.assert_array_equal(DigitalOption(100.0, 1.0, "put").payoff(paths, times), [0, 1])
    assert not call.is_path_dependent
    assert "Digital call" in call.label
    assert "European put" in EuropeanOption(100.0, 1.0, "put").label


@pytest.mark.parametrize(
    "args", [(0.0, 1.0, "call"), (100.0, -1.0, "call"), (100.0, 1.0, "straddle")]
)
def test_invalid_european(args: tuple[float, float, str]) -> None:
    with pytest.raises(ValueError, match=r"must be|unsupported"):
        EuropeanOption(*args)


def test_invalid_digital() -> None:
    with pytest.raises(ValueError, match="payout"):
        DigitalOption(100.0, 1.0, "call", payout=0.0)


def test_time_indices() -> None:
    grid = np.linspace(0.0, 1.0, 5)
    np.testing.assert_array_equal(time_indices(grid, np.array([0.5, 1.0])), [2, 4])
    with pytest.raises(ValueError, match="monitoring"):
        time_indices(grid, np.array([0.3]))


def test_uniform_monitoring() -> None:
    np.testing.assert_allclose(uniform_monitoring(1.0, 4), [0.25, 0.5, 0.75, 1.0])
    with pytest.raises(ValueError, match="monitoring"):
        uniform_monitoring(1.0, 0)
