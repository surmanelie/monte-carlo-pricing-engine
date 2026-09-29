"""CRR binomial tree: convergence to Black-Scholes, American/Bermudan properties."""

from __future__ import annotations

import numpy as np
import pytest

from mcengine.engines.analytic import bs_price
from mcengine.engines.tree import crr_exercise_boundary, crr_price, price_tree
from mcengine.models.gbm import GBM
from mcengine.products.american import AmericanOption
from mcengine.products.asian import AsianOption
from mcengine.products.european import EuropeanOption

ARGS = (36.0, 40.0, 1.0, 0.06, 0.2)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_tree_converges_to_black_scholes(kind: str) -> None:
    ref = float(bs_price(*ARGS, option_type=kind))
    errors = [abs(crr_price(*ARGS, 0.0, kind, n, "european") - ref) for n in (50, 400, 3200)]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 2e-4
    bbsr = crr_price(*ARGS, 0.0, kind, 2000, "european", smoothing=True, richardson=True)
    assert bbsr == pytest.approx(ref, abs=1e-5)


def test_american_put_properties() -> None:
    euro = crr_price(*ARGS, 0.0, "put", 2000, "european")
    american = crr_price(*ARGS, 0.0, "put", 2000, "american")
    bermudan = crr_price(*ARGS, 0.0, "put", 2000, "bermudan", np.arange(1, 51) / 50)
    assert euro < bermudan < american
    assert american >= 40.0 - 36.0  # at least the intrinsic value


def test_bbsr_is_stable_in_n() -> None:
    a = crr_price(*ARGS, 0.0, "put", 2000, "american", smoothing=True, richardson=True)
    b = crr_price(*ARGS, 0.0, "put", 4000, "american", smoothing=True, richardson=True)
    assert a == pytest.approx(b, abs=5e-5)


def test_american_call_without_dividends_equals_european() -> None:
    american = crr_price(40.0, 40.0, 1.0, 0.06, 0.2, 0.0, "call", 1000, "american")
    euro = crr_price(40.0, 40.0, 1.0, 0.06, 0.2, 0.0, "call", 1000, "european")
    assert american == pytest.approx(euro, abs=1e-12)
    with_div = crr_price(40.0, 40.0, 1.0, 0.06, 0.2, 0.08, "call", 1000, "american")
    assert with_div > crr_price(40.0, 40.0, 1.0, 0.06, 0.2, 0.08, "call", 1000, "european")


def test_exercise_boundary_of_put() -> None:
    times, boundary = crr_exercise_boundary(*ARGS, n_steps=500)
    valid = ~np.isnan(boundary)
    assert valid[20:].all()  # early steps have no node deep enough in the money
    assert np.all(boundary[valid][:-1] < 40.0)
    assert boundary[-1] == 40.0
    # the critical price rises towards the strike as maturity approaches
    assert boundary[450] > boundary[50]
    assert times[-1] == 1.0


def test_bermudan_boundary_only_on_exercise_dates() -> None:
    dates = np.arange(1, 11) / 10
    _, boundary = crr_exercise_boundary(*ARGS, n_steps=500, style="bermudan", exercise_times=dates)
    on_dates = np.rint(dates * 500).astype(int)
    off_dates = np.setdiff1d(np.arange(501), on_dates)
    assert np.isnan(boundary[off_dates]).all()
    assert not np.isnan(boundary[on_dates]).any()
    # fewer exercise rights -> lower continuation value -> larger exercise region
    _, american = crr_exercise_boundary(*ARGS, n_steps=500)
    assert np.all(boundary[on_dates][:-1] >= american[on_dates][:-1])
    with pytest.raises(ValueError, match="style"):
        crr_exercise_boundary(*ARGS, style="european")


def test_price_tree_wrapper(bs_model: GBM) -> None:
    euro = price_tree(bs_model, EuropeanOption(100.0, 1.0), n_steps=1000)
    assert euro.method == "crr-bbsr"
    assert euro.price == pytest.approx(float(bs_price(100, 100, 1, 0.05, 0.2)), abs=1e-5)
    american = price_tree(GBM(36.0, 0.06, 0.2), AmericanOption(40.0, 1.0, "put", 50), 1000)
    assert american.n_steps == 1000
    plain = price_tree(bs_model, EuropeanOption(100.0, 1.0), 100, smoothing=False, richardson=False)
    assert plain.method == "crr"
    with pytest.raises(ValueError, match="does not price"):
        price_tree(bs_model, AsianOption(100.0, 1.0))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"style": "asian"}, "style"),
        ({"style": "bermudan"}, "exercise_times"),
        ({"style": "bermudan", "exercise_times": np.array([0.333])}, "grid"),
        ({"richardson": True, "n_steps": 11}, "even"),
        ({"n_steps": 0}, "n_steps"),
        ({"sigma": 0.001, "n_steps": 2}, "probability"),
    ],
)
def test_invalid(kwargs: dict[str, object], match: str) -> None:
    base: dict[str, object] = {
        "s0": 36.0,
        "strike": 40.0,
        "maturity": 1.0,
        "r": 0.06,
        "sigma": 0.2,
        "n_steps": 100,
    }
    with pytest.raises(ValueError, match=match):
        crr_price(**(base | kwargs))  # type: ignore[arg-type]
