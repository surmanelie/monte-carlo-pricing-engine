r"""Calibration of the Heston model to an implied-volatility surface.

The five parameters :math:`\Theta = (v_0, \kappa, \theta, \xi, \rho)` minimise the weighted
sum of squared implied-volatility errors

.. math::

    \min_{\Theta \in B}\ \sum_i w_i\big(\sigma^{\mathrm{model}}_i(\Theta)
    - \sigma^{\mathrm{mkt}}_i\big)^2,

where :math:`\sigma^{\mathrm{model}}_i` inverts the Heston price (vectorised Gil-Pelaez
quadrature, one call per maturity) with the Black-Scholes implied-volatility solver.
Working in volatility units makes errors comparable across strikes and maturities; the
optional weights allow, e.g., bid-ask or vega weighting. The problem is solved with the
bounded trust-region reflective algorithm of ``scipy.optimize.least_squares``
(Branch, Coleman & Li, 1999) from several starting points, keeping the best fit.

The Feller condition :math:`2\kappa\theta \ge \xi^2` is *reported*, not imposed: market
calibrations routinely violate it, and the QE scheme handles that regime (Andersen,
2008).

Real market data can be loaded from a CSV file (see :func:`load_option_chain_csv` for the
schema); nothing is downloaded or scraped.

References
----------
Heston, S. L. (1993). *Review of Financial Studies*, 6(2), 327-343.
Gatheral, J. (2006). *The Volatility Surface*, Chapter 3. Wiley.
Branch, M. A., Coleman, T. F., & Li, Y. (1999). A subspace, interior, and conjugate
gradient method for large-scale bound-constrained minimization problems. *SIAM Journal
on Scientific Computing*, 21(1), 1-23.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from mcengine._typing import FloatArray
from mcengine._validation import require_int_at_least, require_non_negative, require_positive
from mcengine.engines.fourier import fourier_prices
from mcengine.models.heston import Heston
from mcengine.random.generators import make_rng
from mcengine.volatility.implied import implied_vol

PARAM_NAMES: tuple[str, ...] = ("v0", "kappa", "theta", "xi", "rho")
#: Default box constraints for (v0, kappa, theta, xi, rho).
DEFAULT_BOUNDS: tuple[tuple[float, ...], tuple[float, ...]] = (
    (1e-4, 0.05, 1e-4, 0.05, -0.99),
    (1.0, 10.0, 1.0, 3.0, 0.99),
)
#: Residual assigned to a quote whose model price cannot be inverted (volatility units).
PENALTY = 1.0


@dataclass(frozen=True)
class VolSurface:
    """Market implied volatilities, one entry per quote (flattened surface)."""

    s0: float
    r: float
    q: float
    strikes: FloatArray
    maturities: FloatArray
    vols: FloatArray
    weights: FloatArray = field(default_factory=lambda: np.empty(0))

    def __post_init__(self) -> None:
        require_positive("s0", self.s0)
        n = self.strikes.size
        if n == 0 or self.maturities.size != n or self.vols.size != n:
            raise ValueError("strikes, maturities and vols must be non-empty and of equal size")
        if np.any(self.strikes <= 0.0) or np.any(self.maturities <= 0.0):
            raise ValueError("strikes and maturities must be > 0")
        if np.any(~np.isfinite(self.vols)) or np.any(self.vols <= 0.0):
            raise ValueError("implied volatilities must be finite and > 0")
        if self.weights.size not in (0, n) or np.any(self.weights < 0.0):
            raise ValueError("weights must be empty or non-negative with one per quote")

    @property
    def size(self) -> int:
        """Number of quotes."""
        return int(self.strikes.size)

    def weight_vector(self) -> FloatArray:
        """Weights (all ones by default)."""
        return self.weights if self.weights.size else np.ones(self.size)


def model_vols(model: Heston, surface: VolSurface) -> FloatArray:
    """Heston implied volatilities at the quotes of ``surface`` (``nan`` if not invertible)."""
    out = np.full(surface.size, np.nan)
    for maturity in np.unique(surface.maturities):
        idx = np.flatnonzero(surface.maturities == maturity)
        k = surface.strikes[idx]
        # out-of-the-money options carry the information and invert most accurately
        calls = fourier_prices(model, k, float(maturity), "call")
        forward = surface.s0 * math.exp((surface.r - surface.q) * maturity)
        puts = (
            calls
            - surface.s0 * math.exp(-surface.q * maturity)
            + k * math.exp(-surface.r * maturity)
        )
        use_put = k < forward
        vols = np.where(
            use_put,
            np.asarray(implied_vol(puts, surface.s0, k, maturity, surface.r, surface.q, "put")),
            np.asarray(implied_vol(calls, surface.s0, k, maturity, surface.r, surface.q, "call")),
        )
        out[idx] = vols
    return out


def _model(x: Sequence[float], surface: VolSurface) -> Heston:
    v0, kappa, theta, xi, rho = (float(v) for v in x)
    return Heston(surface.s0, surface.r, v0, kappa, theta, xi, rho, surface.q)


def residuals(x: Sequence[float], surface: VolSurface) -> FloatArray:
    """Weighted implied-volatility residuals ``sqrt(w) (model - market)``."""
    diff = model_vols(_model(x, surface), surface) - surface.vols
    diff = np.where(np.isfinite(diff), diff, PENALTY)
    return np.asarray(np.sqrt(surface.weight_vector()) * diff, dtype=np.float64)


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of :func:`calibrate_heston`."""

    model: Heston
    residuals: FloatArray
    rmse_vol: float
    max_abs_error_vol: float
    success: bool
    n_evaluations: int
    n_starts: int
    elapsed_s: float
    message: str

    @property
    def params(self) -> dict[str, float]:
        """Calibrated parameters by name."""
        return {name: float(getattr(self.model, name)) for name in PARAM_NAMES}

    @property
    def feller_ratio(self) -> float:
        r""":math:`2\kappa\theta/\xi^2` of the calibrated model."""
        return self.model.feller_ratio

    @property
    def feller_satisfied(self) -> bool:
        """Whether the calibrated variance process stays strictly positive."""
        return self.feller_ratio >= 1.0


def _starting_points(
    n_starts: int, rng: np.random.Generator, surface: VolSurface
) -> list[FloatArray]:
    atm_var = float(np.median(surface.vols) ** 2)
    base = np.array([atm_var, 1.5, atm_var, 0.5, -0.5])
    points = [base]
    lo, hi = np.array(DEFAULT_BOUNDS[0]), np.array(DEFAULT_BOUNDS[1])
    for _ in range(n_starts - 1):
        guess = base * np.exp(rng.normal(0.0, 0.5, size=5))
        guess[4] = rng.uniform(-0.9, 0.3)
        points.append(np.clip(guess, lo * 1.01, hi * 0.99))
    return points


def calibrate_heston(
    surface: VolSurface,
    *,
    initial: Sequence[float] | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] = DEFAULT_BOUNDS,
    n_starts: int = 4,
    seed: int | None = 0,
    max_nfev: int = 200,
) -> CalibrationResult:
    """Fit ``(v0, kappa, theta, xi, rho)`` to ``surface`` by bounded least squares.

    Parameters
    ----------
    surface
        Market implied volatilities.
    initial
        Optional starting point; otherwise ``n_starts`` points are generated around an
        ATM-variance guess with the given ``seed``.
    bounds
        Lower and upper bounds in the order of :data:`PARAM_NAMES`.
    """
    started = time.perf_counter()
    require_int_at_least("n_starts", n_starts, 1)
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    if lo.shape != (5,) or hi.shape != (5,) or np.any(lo >= hi):
        raise ValueError("bounds must be two sequences of 5 values with lower < upper")
    rng = make_rng(seed)
    starts = (
        [np.clip(np.asarray(initial, dtype=float), lo, hi)]
        if initial is not None
        else _starting_points(n_starts, rng, surface)
    )
    best = None
    evaluations = 0
    for x0 in starts:
        fit = least_squares(
            residuals, np.clip(x0, lo, hi), bounds=(lo, hi), args=(surface,), method="trf",
            x_scale="jac", max_nfev=max_nfev, xtol=1e-12, ftol=1e-12, gtol=1e-12,
        )  # fmt: skip
        evaluations += int(fit.nfev)
        if best is None or fit.cost < best.cost:
            best = fit
    assert best is not None
    model = _model(best.x, surface)
    res = model_vols(model, surface) - surface.vols
    return CalibrationResult(
        model=model,
        residuals=np.asarray(res, dtype=np.float64),
        rmse_vol=float(np.sqrt(np.nanmean(res**2))),
        max_abs_error_vol=float(np.nanmax(np.abs(res))),
        success=bool(best.success),
        n_evaluations=evaluations,
        n_starts=len(starts),
        elapsed_s=time.perf_counter() - started,
        message=str(best.message),
    )


def synthetic_surface(
    model: Heston,
    moneyness: Sequence[float] = (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2),
    maturities: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
    noise_vol: float = 0.0,
    seed: int | None = None,
) -> VolSurface:
    """Implied-volatility surface generated by ``model`` plus Gaussian noise (vol units)."""
    require_non_negative("noise_vol", noise_vol)
    k_grid, t_grid = np.meshgrid(np.asarray(moneyness) * model.s0, np.asarray(maturities))
    strikes, times = k_grid.ravel(), t_grid.ravel()
    clean = VolSurface(model.s0, model.r, model.q, strikes, times, np.full(strikes.size, 0.2))
    vols = model_vols(model, clean)
    if noise_vol > 0.0:
        vols = vols + make_rng(seed).normal(0.0, noise_vol, size=vols.size)
    return VolSurface(model.s0, model.r, model.q, strikes, times, vols)


#: Columns of the option-chain CSV accepted by :func:`load_option_chain_csv`.
CSV_SCHEMA = """\
Required columns (header row, comma separated):
  maturity      time to expiry in years (> 0)
  strike        strike price (> 0)
and either
  implied_vol   Black-Scholes implied volatility (decimal, e.g. 0.215)
or both
  price         option mid price
  option_type   "call" or "put"
Optional column:
  weight        non-negative least-squares weight (default 1)
Spot, rate and dividend yield are passed as arguments."""


def load_option_chain_csv(path: str | Path, s0: float, r: float, q: float = 0.0) -> VolSurface:
    """Load a user-provided option chain (see :data:`CSV_SCHEMA`) as a :class:`VolSurface`.

    Prices are converted to implied volatilities; quotes that violate the no-arbitrage
    bounds raise a :class:`ValueError` naming the offending line.
    """
    strikes, maturities, vols, weights = [], [], [], []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not {"maturity", "strike"} <= fields:
            raise ValueError(f"CSV must contain 'maturity' and 'strike' columns.\n{CSV_SCHEMA}")
        has_vol = "implied_vol" in fields
        if not has_vol and not {"price", "option_type"} <= fields:
            raise ValueError(f"CSV needs 'implied_vol' or 'price' + 'option_type'.\n{CSV_SCHEMA}")
        for line, row in enumerate(reader, start=2):
            try:
                k, t = float(row["strike"]), float(row["maturity"])
                if has_vol:
                    vol = float(row["implied_vol"])
                else:
                    kind = row["option_type"].strip().lower()
                    vol = float(implied_vol(float(row["price"]), s0, k, t, r, q, kind, strict=True))
                weight = float(row["weight"]) if row.get("weight") not in (None, "") else 1.0
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid quote on line {line}: {exc}") from exc
            strikes.append(k)
            maturities.append(t)
            vols.append(vol)
            weights.append(weight)
    return VolSurface(
        s0, r, q, np.asarray(strikes), np.asarray(maturities), np.asarray(vols), np.asarray(weights)
    )
