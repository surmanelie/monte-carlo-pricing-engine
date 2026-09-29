r"""Longstaff-Schwartz least-squares Monte Carlo for Bermudan/American options.

**Regression pass** (training paths). Going backwards over the exercise dates
:math:`t_{M-1}, \dots, t_1`, the realised discounted cash flow :math:`Y` of the current
policy is regressed on basis functions of the spot, *using in-the-money paths only*:

.. math::

    \hat C_k(S) = \sum_{j} \hat\beta_{k,j}\,\psi_j(S/K),\qquad
    \hat\beta_k = \arg\min_\beta \sum_{i \in \mathrm{ITM}}
    \big(Y_i - \psi(S_i/K)^\top\beta\big)^2

(``numpy.linalg.lstsq``), and the policy exercises where :math:`h(S) \ge \hat C_k(S)`.
The basis is the constant plus weighted Laguerre polynomials
:math:`e^{-x/2}L_n(x)` (Longstaff & Schwartz's choice) or monomials :math:`x^n`.

**Pricing pass** (an *independent* set of paths). The frozen policy is applied
forward; the average discounted cash flow is an unbiased estimate of the value of a
feasible, hence sub-optimal, stopping rule, so the estimator is **low-biased**:
:math:`\mathbb E[\hat V] \le V`. Re-using the training paths instead mixes this low bias
with an upward "foresight" bias (the policy is fitted to the very paths it is evaluated
on); the in-sample value is reported in ``diagnostics["in_sample_price"]`` for
comparison. The low bias shrinks as the basis becomes richer and the number of training
paths grows (Clément, Lamberton & Protter, 2002).

References
----------
Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: a
simple least-squares approach. *Review of Financial Studies*, 14(1), 113-147.
Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, Section 8.6.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.special import eval_laguerre

from mcengine._typing import FloatArray
from mcengine._validation import require_choice, require_int_at_least
from mcengine.engines.monte_carlo import auto_chunk_size, chunk_sizes
from mcengine.models.base import Model
from mcengine.products.american import AmericanOption
from mcengine.random.generators import make_rng, spawn
from mcengine.results import PricingResult
from mcengine.stats import StreamingMoments, z_value

BASES: tuple[str, ...] = ("laguerre", "monomial")


def basis_functions(x: FloatArray, basis: str, degree: int) -> FloatArray:
    """Design matrix ``[1, psi_1(x), ..., psi_degree(x)]`` for moneyness ``x = S / K``.

    ``laguerre``: ``psi_n(x) = exp(-x/2) L_{n-1}(x)``; ``monomial``: ``psi_n(x) = x^n``.
    """
    require_choice("basis", basis, BASES)
    require_int_at_least("degree", degree, 1)
    cols = [np.ones_like(x)]
    if basis == "laguerre":
        weight = np.exp(-0.5 * x)
        cols += [weight * eval_laguerre(n, x) for n in range(degree)]
    else:
        cols += [x**n for n in range(1, degree + 1)]
    return np.column_stack(cols)


@dataclass(frozen=True)
class LSMFit:
    """Exercise policy fitted by the regression pass.

    ``coefficients[k]`` belongs to exercise date ``exercise_times[k]``; the row for the
    last date (maturity) is unused because exercise there is always optimal when in the
    money. ``fitted[k]`` is ``False`` where too few paths were in the money to regress;
    the policy never exercises early at such dates. ``support[k]`` is the range of
    in-the-money training spots, outside which the regression is an extrapolation.
    """

    exercise_times: FloatArray
    coefficients: FloatArray
    fitted: np.ndarray
    support: FloatArray
    basis: str
    degree: int
    strike: float
    in_sample_price: float
    n_paths: int

    def continuation(self, k: int, spot: FloatArray) -> FloatArray:
        """Estimated continuation value at date ``k`` (``+inf`` where no fit exists)."""
        if not self.fitted[k]:
            return np.full(np.shape(spot), np.inf)
        design = basis_functions(spot / self.strike, self.basis, self.degree)
        return np.asarray(design @ self.coefficients[k], dtype=np.float64)

    def exercise_boundary(self, product: AmericanOption, n_grid: int = 2000) -> FloatArray:
        """Critical price at each exercise date (``nan`` if the policy never exercises).

        For a put, the largest spot within the training support where immediate exercise
        beats the fitted continuation value; for a call, the smallest.
        """
        is_put = product.option_type == "put"
        out = np.full(self.exercise_times.size, np.nan)
        for k in range(self.exercise_times.size - 1):
            if not self.fitted[k]:
                continue
            grid = np.linspace(self.support[k, 0], self.support[k, 1], n_grid)
            stop = product.exercise_value(grid) >= self.continuation(k, grid)
            if np.any(stop):
                out[k] = grid[stop].max() if is_put else grid[stop].min()
        out[-1] = self.strike
        return out


def _simulate(
    model: Model, times: FloatArray, n_paths: int, rng: np.random.Generator
) -> FloatArray:
    return model.paths_from_normals(
        times, rng.standard_normal((n_paths, times.size - 1, model.n_factors))
    )


def _time_grid(product: AmericanOption, steps_per_date: int) -> tuple[FloatArray, np.ndarray]:
    n = product.n_exercise * steps_per_date
    times = np.linspace(0.0, product.maturity, n + 1)
    return times, np.arange(steps_per_date, n + 1, steps_per_date)


def fit_lsm(
    model: Model,
    product: AmericanOption,
    *,
    n_paths: int = 100_000,
    basis: str = "laguerre",
    degree: int = 3,
    steps_per_date: int = 1,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> LSMFit:
    """Regression pass: estimate the continuation-value coefficients on training paths."""
    require_int_at_least("n_paths", n_paths, 2)
    require_int_at_least("steps_per_date", steps_per_date, 1)
    basis_functions(np.ones(1), basis, degree)  # validates basis and degree
    gen = make_rng(seed, rng)
    times, idx = _time_grid(product, steps_per_date)
    paths = _simulate(model, times, n_paths, gen)
    ex_times = times[idx]
    spots = paths[:, idx]
    n_dates = ex_times.size
    coefficients = np.zeros((n_dates, degree + 1))
    fitted = np.zeros(n_dates, dtype=bool)
    support = np.full((n_dates, 2), np.nan)
    cash = product.exercise_value(spots[:, -1])
    for k in range(n_dates - 2, -1, -1):
        cash *= math.exp(-model.r * (ex_times[k + 1] - ex_times[k]))
        exercise = product.exercise_value(spots[:, k])
        itm = exercise > 0.0
        if np.count_nonzero(itm) <= 2 * (degree + 1):
            continue  # too few in-the-money paths: never exercise at this date
        itm_spots = spots[itm, k]
        design = basis_functions(itm_spots / product.strike, basis, degree)
        beta, *_ = np.linalg.lstsq(design, cash[itm], rcond=None)
        coefficients[k], fitted[k] = beta, True
        support[k] = itm_spots.min(), itm_spots.max()
        stop = exercise[itm] >= design @ beta
        itm_idx = np.flatnonzero(itm)[stop]
        cash[itm_idx] = exercise[itm_idx]
    in_sample = float(math.exp(-model.r * ex_times[0]) * cash.mean())
    return LSMFit(
        ex_times, coefficients, fitted, support, basis, degree, product.strike, in_sample, n_paths
    )


def _apply_policy(
    model: Model, product: AmericanOption, fit: LSMFit, paths: FloatArray, idx: np.ndarray
) -> FloatArray:
    """Discounted cash flow of the fitted policy on (independent) paths."""
    n = paths.shape[0]
    value = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    last = fit.exercise_times.size - 1
    for k in range(last + 1):
        spot = paths[:, idx[k]]
        exercise = product.exercise_value(spot)
        candidates = alive & (exercise > 0.0)
        if k < last and np.any(candidates):
            stop = np.zeros(n, dtype=bool)
            stop[candidates] = exercise[candidates] >= fit.continuation(k, spot[candidates])
        else:
            stop = candidates
        value[stop] = math.exp(-model.r * fit.exercise_times[k]) * exercise[stop]
        alive &= ~stop
    return value


def price_lsm(
    model: Model,
    product: AmericanOption,
    *,
    n_paths: int = 100_000,
    n_train: int | None = None,
    basis: str = "laguerre",
    degree: int = 3,
    steps_per_date: int = 1,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    chunk_size: int | None = None,
) -> PricingResult:
    """Low-biased Longstaff-Schwartz price with an independent pricing set.

    Parameters
    ----------
    n_paths
        Paths of the (chunked, constant-memory) pricing pass.
    n_train
        Training paths for the regression pass (default ``n_paths``).
    basis, degree
        Regression basis (``"laguerre"`` or ``"monomial"``) and number of non-constant terms.
    steps_per_date
        Simulation steps between exercise dates (1 suffices for exact models such as GBM).
    """
    started = time.perf_counter()
    if not isinstance(product, AmericanOption):
        raise ValueError("price_lsm prices AmericanOption products")
    require_int_at_least("n_paths", n_paths, 2)
    train_rng, price_rng = spawn(make_rng(seed, rng), 2)
    fit = fit_lsm(
        model,
        product,
        n_paths=n_paths if n_train is None else n_train,
        basis=basis,
        degree=degree,
        steps_per_date=steps_per_date,
        rng=train_rng,
    )
    times, idx = _time_grid(product, steps_per_date)
    chunk = auto_chunk_size(times.size, model.n_factors) if chunk_size is None else chunk_size
    require_int_at_least("chunk_size", chunk, 2)
    acc = StreamingMoments()
    for size in chunk_sizes(n_paths, chunk):
        paths = _simulate(model, times, size, price_rng)
        acc.update(_apply_policy(model, product, fit, paths, idx))
    price, se = float(acc.mean[0]), acc.std_error()
    half = z_value(0.95) * se
    return PricingResult(
        price=price,
        std_error=se,
        ci_low=price - half,
        ci_high=price + half,
        n_paths=n_paths,
        n_steps=times.size - 1,
        method="lsm",
        elapsed_s=time.perf_counter() - started,
        diagnostics={
            "in_sample_price": fit.in_sample_price,
            "n_train": float(fit.n_paths),
            "degree": float(degree),
        },
    )
