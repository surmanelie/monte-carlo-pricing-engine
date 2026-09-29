"""Pseudo-random streams with explicit, injectable state.

Every stochastic function in :mod:`mcengine` accepts ``seed: int | None`` and/or
``rng: numpy.random.Generator``; this module turns that pair into a single generator
and derives statistically independent child streams via :class:`numpy.random.SeedSequence`
spawning (PCG64 by default, O'Neill 2014).
"""

from __future__ import annotations

import numpy as np

from mcengine._typing import FloatArray


def make_rng(
    seed: int | None = None, rng: np.random.Generator | None = None
) -> np.random.Generator:
    """Return a generator from either ``seed`` or ``rng`` (not both).

    Parameters
    ----------
    seed
        Integer seed; ``None`` draws fresh OS entropy.
    rng
        An existing generator, returned unchanged (its state is shared with the caller).
    """
    if rng is not None:
        if seed is not None:
            raise ValueError("pass either seed or rng, not both")
        if not isinstance(rng, np.random.Generator):
            raise ValueError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")
        return rng
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError(f"seed must be a non-negative integer or None, got {seed!r}")
    return np.random.default_rng(seed)


def spawn(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Derive ``n`` independent child generators from ``rng``."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return list(rng.spawn(n))


def standard_normals(rng: np.random.Generator, shape: tuple[int, ...]) -> FloatArray:
    """Draw i.i.d. standard normals of the given shape (C order, sequential in the stream).

    Consecutive calls consume the stream sequentially, so drawing ``(a, d)`` then
    ``(b, d)`` yields exactly the same numbers as one ``(a + b, d)`` draw. The chunked
    Monte Carlo engine relies on this to be independent of the chunk size.
    """
    return rng.standard_normal(shape)
