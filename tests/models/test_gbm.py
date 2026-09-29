from __future__ import annotations

import numpy as np
import pytest

from mcengine.models.base import validate_time_grid
from mcengine.models.gbm import GBM


def test_paths_shape_and_start(bs_model: GBM) -> None:
    times = np.linspace(0.0, 1.0, 13)
    paths = bs_model.simulate_paths(times, 1000, seed=0)
    assert paths.shape == (1000, 13)
    assert np.all(paths[:, 0] == bs_model.s0)
    assert np.all(paths > 0.0)


def test_terminal_moments_exact(bs_model: GBM) -> None:
    """E[S_T] = S0 e^{(r-q)T} and Var[ln S_T] = sigma^2 T (checked in SE units)."""
    n, t = 200_000, 2.0
    s_t = bs_model.simulate_paths(np.array([0.0, t]), n, seed=1)[:, -1]
    mean_se = s_t.std(ddof=1) / np.sqrt(n)
    assert abs(s_t.mean() - bs_model.forward(t)) < 4 * mean_se
    log_var = np.log(s_t).var(ddof=1)
    var_se = bs_model.sigma**2 * t * np.sqrt(2.0 / (n - 1))
    assert abs(log_var - bs_model.sigma**2 * t) < 4 * var_se


def test_grid_invariance_of_terminal_distribution(bs_model: GBM) -> None:
    """Exact scheme: log-increments on a fine grid sum to the one-step increment."""
    z = np.random.default_rng(2).standard_normal((5, 4, 1))
    fine = bs_model.paths_from_normals(np.linspace(0, 1, 5), z)
    one_step = bs_model.paths_from_normals(
        np.array([0.0, 1.0]), (z.sum(axis=1, keepdims=True) / 2.0)
    )
    np.testing.assert_allclose(fine[:, -1], one_step[:, -1], rtol=1e-12)


def test_char_func_matches_moments(bs_model: GBM) -> None:
    t = 1.5
    assert bs_model.char_func(np.array([0.0]), t)[0] == pytest.approx(1.0)
    # E[S_T] = phi(-i)
    assert bs_model.char_func(np.array([-1j]), t)[0].real == pytest.approx(bs_model.forward(t))
    assert bs_model.has_char_func


@pytest.mark.parametrize(
    "kwargs",
    [
        {"s0": 0.0, "r": 0.05, "sigma": 0.2},
        {"s0": 100.0, "r": 0.05, "sigma": -0.2},
        {"s0": 100.0, "r": float("nan"), "sigma": 0.2},
        {"s0": 100.0, "r": 0.05, "sigma": 0.2, "q": -0.01},
    ],
)
def test_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"must be"):
        GBM(**kwargs)


def test_invalid_grid_and_normals(bs_model: GBM) -> None:
    with pytest.raises(ValueError, match="start at 0"):
        validate_time_grid(np.array([0.1, 1.0]))
    with pytest.raises(ValueError, match="increasing"):
        validate_time_grid(np.array([0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="at least two"):
        validate_time_grid(np.array([0.0]))
    with pytest.raises(ValueError, match="shape"):
        bs_model.paths_from_normals(np.array([0.0, 1.0]), np.zeros((3, 2, 1)))
    with pytest.raises(ValueError, match="n_paths"):
        bs_model.simulate_paths(np.array([0.0, 1.0]), 0)
