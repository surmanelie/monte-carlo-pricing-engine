"""Brownian-bridge construction and scrambled Sobol normals."""

from __future__ import annotations

import numpy as np
import pytest

from mcengine.random.qmc import (
    MAX_SOBOL_DIM,
    SobolNormals,
    bridge_increments,
    bridge_plan,
    is_power_of_two,
)

GRIDS = [np.linspace(0.0, 1.0, 9), np.array([0.0, 0.1, 0.15, 0.5, 0.7, 1.3, 2.0])]


@pytest.mark.parametrize("times", GRIDS)
def test_bridge_is_an_orthogonal_map(times: np.ndarray) -> None:
    """increments = A z with A A^T = I: bridge normals are again i.i.d. N(0, 1)."""
    m = times.size - 1
    a = bridge_increments(np.eye(m), times).T
    np.testing.assert_allclose(a @ a.T, np.eye(m), atol=1e-12)


def test_bridge_order_starts_with_terminal_value() -> None:
    plan = bridge_plan(np.linspace(0.0, 1.0, 9))
    assert plan.target[0] == 8
    assert plan.target[1] == 4
    assert sorted(plan.target.tolist()) == list(range(1, 9))
    # the first normal alone determines W_T
    z = np.zeros((1, 8))
    z[0, 0] = 1.0
    increments = bridge_increments(z, np.linspace(0.0, 1.0, 9))
    assert np.sum(increments * np.sqrt(1 / 8)) == pytest.approx(1.0)


def test_bridge_validation() -> None:
    with pytest.raises(ValueError, match="bridge grid"):
        bridge_plan(np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="normals per path"):
        bridge_increments(np.zeros((2, 3)), np.linspace(0.0, 1.0, 3))


@pytest.mark.parametrize("bridge", [True, False])
def test_sobol_normals_shape_and_moments(bridge: bool) -> None:
    times = np.linspace(0.0, 1.0, 5)
    sampler = SobolNormals(times, 2, np.random.default_rng(0), bridge=bridge)
    z = sampler.draw(2**14)
    assert z.shape == (2**14, 4, 2)
    flat = z.reshape(-1, 8)
    np.testing.assert_allclose(flat.mean(axis=0), 0.0, atol=5e-3)
    np.testing.assert_allclose(np.cov(flat.T), np.eye(8), atol=2e-2)


def test_sobol_sequential_draws_are_chunk_invariant() -> None:
    times = np.linspace(0.0, 1.0, 3)
    one = SobolNormals(times, 1, np.random.default_rng(1)).draw(64)
    sampler = SobolNormals(times, 1, np.random.default_rng(1))
    parts = np.concatenate([sampler.draw(32), sampler.draw(16), sampler.draw(16)])
    np.testing.assert_array_equal(one, parts)


def test_sobol_validation() -> None:
    times = np.linspace(0.0, 1.0, 3)
    with pytest.raises(ValueError, match="powers of two"):
        SobolNormals(times, 1, np.random.default_rng(0)).draw(100)
    with pytest.raises(ValueError, match="dimension"):
        SobolNormals(np.linspace(0.0, 1.0, MAX_SOBOL_DIM + 2), 1, np.random.default_rng(0))
    assert is_power_of_two(1024)
    assert not is_power_of_two(0)
    assert not is_power_of_two(96)
