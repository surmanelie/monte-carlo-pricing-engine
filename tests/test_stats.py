from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mcengine.stats import StreamingMoments, confidence_interval, z_value


def test_matches_numpy_on_single_batch() -> None:
    x = np.random.default_rng(0).normal(3.0, 2.0, size=1000)
    acc = StreamingMoments()
    acc.update(x)
    assert acc.mean[0] == pytest.approx(x.mean(), rel=1e-14)
    assert acc.variance[0] == pytest.approx(x.var(ddof=1), rel=1e-12)
    assert acc.std_error() == pytest.approx(x.std(ddof=1) / np.sqrt(x.size), rel=1e-12)


@settings(max_examples=50, deadline=None)
@given(
    sizes=st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=8),
    seed=st.integers(min_value=0, max_value=2**31),
)
def test_chunked_equals_single_pass(sizes: list[int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    data = rng.normal(1e6, 1.0, size=(sum(sizes) + 1, 2))  # large offset: stability check
    acc = StreamingMoments(dim=2)
    start = 0
    for size in [*sizes, 1]:
        acc.update(data[start : start + size])
        start += size
    np.testing.assert_allclose(acc.mean, data.mean(axis=0), rtol=1e-13)
    np.testing.assert_allclose(acc.covariance, np.cov(data.T), rtol=1e-6, atol=1e-9)


def test_merge_two_accumulators() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=100), rng.normal(size=37)
    acc_a, acc_b = StreamingMoments(), StreamingMoments()
    acc_a.update(a)
    acc_b.update(b)
    acc_a.merge(acc_b)
    both = np.concatenate([a, b])
    assert acc_a.count == both.size
    assert acc_a.variance[0] == pytest.approx(both.var(ddof=1), rel=1e-12)


def test_empty_update_and_merge_are_noops() -> None:
    acc = StreamingMoments()
    acc.update(np.empty(0))
    acc.merge(StreamingMoments())
    assert acc.count == 0


def test_errors() -> None:
    with pytest.raises(ValueError, match="dim"):
        StreamingMoments(dim=0)
    acc = StreamingMoments(dim=2)
    with pytest.raises(ValueError, match="shape"):
        acc.update(np.ones(3))
    with pytest.raises(ValueError, match="no samples"):
        _ = acc.mean
    acc.update(np.ones((1, 2)))
    with pytest.raises(ValueError, match="two samples"):
        _ = acc.covariance
    with pytest.raises(ValueError, match="dimension"):
        acc.merge(StreamingMoments(dim=1))


def test_confidence_interval() -> None:
    assert z_value(0.95) == pytest.approx(1.959963984540054)
    low, high = confidence_interval(1.0, 0.1, 0.99)
    assert (high - low) / 2 == pytest.approx(0.2575829303548901)
    with pytest.raises(ValueError, match="level"):
        z_value(1.0)
    with pytest.raises(ValueError, match="std_error"):
        confidence_interval(0.0, -1.0)
