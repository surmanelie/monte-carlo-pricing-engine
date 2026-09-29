"""Closed-form Greeks against central finite differences of the closed-form price."""

from __future__ import annotations

import numpy as np
import pytest

from mcengine.engines.analytic import bs_digital_price, bs_price
from mcengine.greeks.analytic import bs_delta, bs_gamma, bs_greeks, bs_vega, digital_greeks

S, K, T, R, SIG, Q = 100.0, 95.0, 0.75, 0.04, 0.25, 0.02


def _fd(f, x: float, h: float) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    up, mid, dn = f(x + h), f(x), f(x - h)
    return (up - dn) / (2 * h), (up - 2 * mid + dn) / h**2


@pytest.mark.parametrize("kind", ["call", "put"])
def test_bs_greeks_match_finite_differences(kind: str) -> None:
    g = bs_greeks(S, K, T, R, SIG, Q, kind)
    delta, gamma = _fd(lambda s: float(bs_price(s, K, T, R, SIG, Q, kind)), S, 1e-2)
    vega, _ = _fd(lambda v: float(bs_price(S, K, T, R, v, Q, kind)), SIG, 1e-5)
    rho, _ = _fd(lambda r: float(bs_price(S, K, T, r, SIG, Q, kind)), R, 1e-6)
    dtheta, _ = _fd(lambda t: float(bs_price(S, K, t, R, SIG, Q, kind)), T, 1e-6)
    assert g.delta == pytest.approx(delta, rel=1e-7)
    assert g.gamma == pytest.approx(gamma, rel=1e-5)
    assert g.vega == pytest.approx(vega, rel=1e-7)
    assert g.rho == pytest.approx(rho, rel=1e-6)
    assert g.theta == pytest.approx(-dtheta, rel=1e-6)
    assert g.method == "bs-analytic"


def test_vectorised_greeks_match_scalar() -> None:
    strikes = np.array([90.0, 100.0, 110.0])
    for i, k in enumerate(strikes):
        g = bs_greeks(S, float(k), T, R, SIG, Q, "put")
        assert bs_delta(S, strikes, T, R, SIG, Q, "put")[i] == pytest.approx(g.delta)  # type: ignore[index]
        assert bs_gamma(S, strikes, T, R, SIG, Q)[i] == pytest.approx(g.gamma)  # type: ignore[index]
        assert bs_vega(S, strikes, T, R, SIG, Q)[i] == pytest.approx(g.vega)  # type: ignore[index]
    assert isinstance(bs_delta(S, K, T, R, SIG, Q), float)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_digital_greeks_match_finite_differences(kind: str) -> None:
    g = digital_greeks(S, K, T, R, SIG, Q, kind, payout=2.0)
    price = lambda s, v: 2.0 * float(bs_digital_price(s, K, T, R, v, Q, kind))  # noqa: E731
    delta, gamma = _fd(lambda s: price(s, SIG), S, 1e-2)
    vega, _ = _fd(lambda v: price(S, v), SIG, 1e-5)
    assert g.delta == pytest.approx(delta, rel=1e-6)
    assert g.gamma == pytest.approx(gamma, rel=1e-4)
    assert g.vega == pytest.approx(vega, rel=1e-6)
    assert g.theta is None


def test_invalid_option_type() -> None:
    with pytest.raises(ValueError, match="option_type"):
        bs_greeks(S, K, T, R, SIG, Q, "chooser")
    with pytest.raises(ValueError, match="option_type"):
        bs_delta(S, K, T, R, SIG, Q, "chooser")
    with pytest.raises(ValueError, match="option_type"):
        digital_greeks(S, K, T, R, SIG, Q, "chooser")
