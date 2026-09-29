from __future__ import annotations

import numpy as np
import pytest

from mcengine.random.generators import make_rng, spawn, standard_normals


def test_make_rng_is_reproducible() -> None:
    a = make_rng(42).standard_normal(5)
    b = make_rng(42).standard_normal(5)
    np.testing.assert_array_equal(a, b)


def test_make_rng_passes_generator_through() -> None:
    gen = np.random.default_rng(0)
    assert make_rng(rng=gen) is gen


@pytest.mark.parametrize(
    ("seed", "rng"),
    [(1, np.random.default_rng(0)), (-1, None), (1.5, None), (True, None), (None, "x")],
)
def test_make_rng_rejects_bad_input(seed: object, rng: object) -> None:
    with pytest.raises(ValueError, match=r"seed|rng"):
        make_rng(seed, rng)  # type: ignore[arg-type]


def test_spawn_independent_streams() -> None:
    children = spawn(make_rng(7), 3)
    draws = [c.standard_normal(4) for c in children]
    assert not np.allclose(draws[0], draws[1])
    with pytest.raises(ValueError, match="n must"):
        spawn(make_rng(7), 0)


def test_sequential_draws_equal_one_draw() -> None:
    """The chunk-size invariance of the MC engine relies on this property."""
    one = standard_normals(make_rng(3), (10, 4, 2))
    gen = make_rng(3)
    parts = [standard_normals(gen, (n, 4, 2)) for n in (3, 5, 2)]
    np.testing.assert_array_equal(one, np.concatenate(parts))
