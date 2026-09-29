"""Heston calibration: synthetic recovery, noise, CSV loading, validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mcengine.calibration.heston_calibration import (
    PARAM_NAMES,
    VolSurface,
    calibrate_heston,
    load_option_chain_csv,
    model_vols,
    residuals,
    synthetic_surface,
)
from mcengine.engines.analytic import bs_price
from mcengine.models.heston import Heston

TRUE = Heston(100.0, 0.02, 0.04, 1.5, 0.05, 0.6, -0.7)


def test_noise_free_surface_is_recovered_exactly() -> None:
    surface = synthetic_surface(TRUE, maturities=(0.5, 1.0, 2.0))
    fit = calibrate_heston(surface, n_starts=2, seed=1)
    assert fit.success
    for name in PARAM_NAMES:
        assert fit.params[name] == pytest.approx(getattr(TRUE, name), rel=1e-6)
    assert fit.rmse_vol < 1e-10
    assert fit.n_starts == 2
    assert not fit.feller_satisfied
    assert fit.feller_ratio == pytest.approx(TRUE.feller_ratio, rel=1e-6)


def test_noisy_surface_fits_to_the_noise_level() -> None:
    noise = 0.002
    surface = synthetic_surface(TRUE, noise_vol=noise, seed=3)
    fit = calibrate_heston(surface, initial=(0.05, 1.0, 0.04, 0.5, -0.5))
    assert fit.rmse_vol < 1.2 * noise
    assert fit.params["rho"] == pytest.approx(-0.7, abs=0.05)
    assert fit.params["v0"] == pytest.approx(0.04, rel=0.05)
    assert fit.residuals.shape == (surface.size,)


def test_weights_enter_the_residuals() -> None:
    surface = synthetic_surface(TRUE, maturities=(1.0,))
    weighted = VolSurface(
        surface.s0, surface.r, surface.q, surface.strikes, surface.maturities,
        surface.vols + 0.01, np.full(surface.size, 4.0),
    )  # fmt: skip
    x = [getattr(TRUE, n) for n in PARAM_NAMES]
    np.testing.assert_allclose(residuals(x, weighted), -0.02, atol=1e-9)


def test_model_vols_match_black_scholes_limit() -> None:
    nearly_bs = Heston(100.0, 0.01, 0.04, 3.0, 0.04, 1e-3, 0.0)
    surface = synthetic_surface(nearly_bs, maturities=(1.0,))
    np.testing.assert_allclose(model_vols(nearly_bs, surface), 0.2, atol=1e-5)


def test_csv_with_implied_vols_and_prices(tmp_path: Path) -> None:
    vol_file = tmp_path / "vols.csv"
    vol_file.write_text("maturity,strike,implied_vol,weight\n1.0,90,0.25,2\n1.0,110,0.2,\n")
    surface = load_option_chain_csv(vol_file, s0=100.0, r=0.01)
    np.testing.assert_allclose(surface.vols, [0.25, 0.2])
    np.testing.assert_allclose(surface.weight_vector(), [2.0, 1.0])
    price = float(bs_price(100.0, 105.0, 0.5, 0.01, 0.3, option_type="put"))
    price_file = tmp_path / "prices.csv"
    price_file.write_text(f"maturity,strike,price,option_type\n0.5,105,{price},put\n")
    surface = load_option_chain_csv(price_file, s0=100.0, r=0.01)
    assert surface.vols[0] == pytest.approx(0.3, abs=1e-8)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("strike,implied_vol\n100,0.2\n", "maturity"),
        ("maturity,strike\n1,100\n", "implied_vol"),
        ("maturity,strike,implied_vol\n1,abc,0.2\n", "line 2"),
        ("maturity,strike,price,option_type\n1,100,500,call\n", "line 2"),
    ],
)
def test_csv_errors(tmp_path: Path, content: str, match: str) -> None:
    path = tmp_path / "chain.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        load_option_chain_csv(path, s0=100.0, r=0.0)


def test_surface_and_bounds_validation() -> None:
    one = np.ones(1)
    with pytest.raises(ValueError, match="equal size"):
        VolSurface(100.0, 0.0, 0.0, one, np.ones(2), one)
    with pytest.raises(ValueError, match="> 0"):
        VolSurface(100.0, 0.0, 0.0, -one, one, one)
    with pytest.raises(ValueError, match="finite"):
        VolSurface(100.0, 0.0, 0.0, one, one, np.array([np.nan]))
    with pytest.raises(ValueError, match="weights"):
        VolSurface(100.0, 0.0, 0.0, one, one, one, -one)
    surface = synthetic_surface(TRUE, maturities=(1.0,))
    with pytest.raises(ValueError, match="bounds"):
        calibrate_heston(surface, bounds=((0.0,) * 5, (0.0,) * 5))
    with pytest.raises(ValueError, match="noise_vol"):
        synthetic_surface(TRUE, noise_vol=-1.0)
