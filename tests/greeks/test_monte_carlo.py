"""Monte Carlo Greeks vs closed forms, in standard-error units."""

from __future__ import annotations

import numpy as np
import pytest

from mcengine.greeks.analytic import bs_greeks, digital_greeks
from mcengine.greeks.monte_carlo import greek_samples, mc_greeks
from mcengine.models.gbm import GBM
from mcengine.models.merton import Merton
from mcengine.products.asian import AsianOption
from mcengine.products.european import DigitalOption, EuropeanOption
from mcengine.results import GreeksResult
from tests.conftest import TOL_SE

N = 200_000


def _check(result: GreeksResult, ref: GreeksResult, names: tuple[str, ...]) -> None:
    for name in names:
        est, exact = getattr(result, name), getattr(ref, name)
        se = result.std_errors[name]
        assert abs(est - exact) < TOL_SE * se, (result.method, name, est, exact, se)


@pytest.mark.parametrize("method", ["bump", "pathwise", "lr"])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_vanilla_greeks(bs_model: GBM, method: str, kind: str) -> None:
    ref = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, kind)
    res = mc_greeks(bs_model, EuropeanOption(100.0, 1.0, kind), method=method, n_paths=N, seed=11)
    assert res.method == f"mc-{method}"
    assert res.n_paths == N
    _check(res, ref, ("delta", "gamma", "vega"))


def test_digital_likelihood_ratio_works_and_pathwise_fails(bs_model: GBM) -> None:
    ref = digital_greeks(100.0, 100.0, 1.0, 0.05, 0.2)
    digital = DigitalOption(100.0, 1.0)
    lr = mc_greeks(bs_model, digital, method="lr", n_paths=N, seed=12)
    _check(lr, ref, ("delta", "gamma", "vega"))
    pw = mc_greeks(bs_model, digital, method="pathwise", n_paths=N, seed=12)
    assert pw.delta == 0.0
    assert pw.gamma == 0.0
    assert ref.delta is not None
    assert ref.gamma is not None
    assert abs(ref.delta) > 0.01  # the true delta is far from the pathwise answer
    assert abs(ref.gamma) > 1e-4


def test_digital_bump_gamma_is_much_noisier_than_lr(bs_model: GBM) -> None:
    digital = DigitalOption(100.0, 1.0)
    bump = mc_greeks(bs_model, digital, method="bump", n_paths=N, seed=13)
    lr = mc_greeks(bs_model, digital, method="lr", n_paths=N, seed=13)
    assert bump.std_errors["gamma"] > 10.0 * lr.std_errors["gamma"]


def test_pathwise_lr_gamma_has_lowest_variance(bs_model: GBM) -> None:
    call = EuropeanOption(100.0, 1.0)
    se = {
        m: mc_greeks(bs_model, call, method=m, n_paths=N, seed=14).std_errors
        for m in ("bump", "pathwise", "lr")
    }
    assert se["pathwise"]["gamma"] < se["bump"]["gamma"]
    assert se["pathwise"]["gamma"] < se["lr"]["gamma"]
    assert se["pathwise"]["delta"] < se["lr"]["delta"]


def test_samples_are_reproducible(bs_model: GBM) -> None:
    z = np.random.default_rng(0).standard_normal(10)
    a = greek_samples(bs_model, EuropeanOption(100.0, 1.0), z, "lr")
    b = greek_samples(bs_model, EuropeanOption(100.0, 1.0), z, "lr")
    np.testing.assert_array_equal(a["gamma"], b["gamma"])


def test_invalid(bs_model: GBM) -> None:
    with pytest.raises(ValueError, match="method"):
        mc_greeks(bs_model, EuropeanOption(100.0, 1.0), method="aad")
    with pytest.raises(ValueError, match="GBM"):
        mc_greeks(Merton(100.0, 0.05, 0.2, 1.0, 0.0, 0.1), EuropeanOption(100.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="European and digital"):
        mc_greeks(bs_model, AsianOption(100.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bump_spot"):
        mc_greeks(bs_model, EuropeanOption(100.0, 1.0), method="bump", bump_spot=0.0)
