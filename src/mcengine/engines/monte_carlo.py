r"""Monte Carlo pricing engine with variance reduction.

The engine prices :math:`V_0 = e^{-rT}\,\mathbb E^{\mathbb Q}[h(S_{t_1},\dots,S_{t_m})]` by
averaging discounted payoffs over simulated paths, processed in chunks whose sample
moments are merged with the Chan-Golub-LeVeque update (:mod:`mcengine.stats`), so that
memory is constant in the number of paths.

Estimators (Glasserman 2003, Ch. 4):

``plain``
    :math:`\hat V = \frac1n\sum_i Y_i`, standard error :math:`s_Y/\sqrt n`.
``antithetic``
    pairs :math:`(Z, -Z)`; the estimator averages :math:`\tfrac12(Y(Z) + Y(-Z))` over
    :math:`n/2` independent pairs and its standard error is computed from the pair means.
``cv`` (control variate)
    :math:`\hat V_{cv} = \bar Y - \hat\beta(\bar X - \mathbb E X)` with the optimal
    coefficient :math:`\hat\beta = \widehat{\mathrm{Cov}}(Y,X)/\widehat{\mathrm{Var}}(X)`
    estimated on the same sample (bias :math:`O(1/n)`, Glasserman 2003, Section 4.1.2);
    variance :math:`\widehat{\mathrm{Var}}(Y - \hat\beta X)/n`. The default control for
    vanilla payoffs is the discounted terminal price :math:`e^{-rT}S_T` with known mean
    :math:`S_0 e^{-qT}`.
``is`` (importance sampling)
    the normals driving the price are drawn from :math:`N(c_k, 1)` instead of
    :math:`N(0, 1)`, with the constant-drift (Girsanov) shift
    :math:`c_k = \mu\sqrt{\Delta t_k/T}` that moves the terminal Brownian value by
    :math:`\mu\sqrt T`; each payoff is weighted by the likelihood ratio
    :math:`\exp(\sum_k -c_kZ_k + \tfrac12c_k^2)`. The default
    :math:`\mu = (\ln(K/S_0) - (r - q - \tfrac12\sigma^2)T)/(\sigma\sqrt T)` centres the
    terminal price at the strike, which is effective for out-of-the-money payoffs
    (Glasserman 2003, Section 4.6.1).
``qmc`` (randomised quasi-Monte Carlo)
    ``n_scrambles`` independently scrambled Sobol sequences of ``n_paths / n_scrambles``
    points (a power of two), with Brownian-bridge path construction
    (:mod:`mcengine.random.qmc`); the standard error is the standard deviation of the
    scramble estimates divided by :math:`\sqrt{R}`.

Reproducibility: all normals come from one generator stream consumed sequentially (and
Sobol points are drawn sequentially), so the estimate does not depend on the chunk size
(up to floating-point summation order).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np

from mcengine._typing import FloatArray
from mcengine._validation import require_choice, require_int_at_least
from mcengine.engines.analytic import geometric_asian_price
from mcengine.models.base import Model
from mcengine.models.gbm import GBM
from mcengine.models.heston import Heston
from mcengine.models.merton import Merton
from mcengine.products.american import AmericanOption
from mcengine.products.asian import AsianOption
from mcengine.products.barrier import BarrierOption
from mcengine.products.base import Product, time_indices
from mcengine.products.european import DigitalOption, EuropeanOption
from mcengine.random.generators import make_rng, spawn
from mcengine.random.qmc import SobolNormals, is_power_of_two
from mcengine.results import PricingResult
from mcengine.stats import StreamingMoments, z_value

Method = Literal["plain", "antithetic", "cv", "is", "qmc"]
METHODS: tuple[str, ...] = ("plain", "antithetic", "cv", "is", "qmc")
METHOD_LABELS: dict[str, str] = {
    "plain": "mc-plain",
    "antithetic": "mc-antithetic",
    "cv": "mc-cv",
    "is": "mc-is",
    "qmc": "qmc-sobol-bb",
}

#: Approximate number of float64 values held in memory per chunk (~64 MB of paths).
_CHUNK_BUDGET = 2**23
_MAX_CHUNK = 2**16


@dataclass(frozen=True)
class ControlVariate:
    """A control variate: discounted values ``func(paths, times)`` with known mean ``mean``."""

    name: str
    mean: float
    func: Callable[[FloatArray, FloatArray], FloatArray]


def underlying_control(model: Model, maturity: float) -> ControlVariate:
    """Discounted terminal price :math:`e^{-rT}S_T`, whose mean is :math:`S_0e^{-qT}`."""
    disc = math.exp(-model.r * maturity)

    def func(paths: FloatArray, times: FloatArray) -> FloatArray:
        return disc * paths[:, time_indices(times, np.array([maturity]))[0]]

    return ControlVariate("discounted-terminal", model.s0 * math.exp(-model.q * maturity), func)


def geometric_asian_control(model: GBM, product: AsianOption) -> ControlVariate:
    """Discounted geometric-average payoff with its Kemna-Vorst mean (Kemna & Vorst, 1990)."""
    geometric = product.with_average("geometric")
    disc = math.exp(-model.r * product.maturity)
    mean = geometric_asian_price(
        model.s0,
        product.strike,
        product.monitoring_times(),
        model.r,
        model.sigma,
        model.q,
        product.option_type,
    )

    def func(paths: FloatArray, times: FloatArray) -> FloatArray:
        return disc * geometric.payoff(paths, times)

    return ControlVariate("geometric-asian", mean, func)


def default_control(model: Model, product: Product) -> ControlVariate:
    """Standard control variate for ``product`` under ``model``.

    Arithmetic Asian options under GBM use the geometric Asian (Kemna-Vorst); every other
    case uses the discounted terminal price.
    """
    if isinstance(model, GBM) and isinstance(product, AsianOption):
        return geometric_asian_control(model, product)
    return underlying_control(model, product.maturity)


def reference_vol(model: Model) -> float:
    """A representative volatility of the log-price (used by the default IS shift)."""
    if isinstance(model, GBM):
        return model.sigma
    if isinstance(model, Merton):
        return math.sqrt(model.sigma**2 + model.lam * (model.mu_j**2 + model.delta_j**2))
    if isinstance(model, Heston):
        return math.sqrt(max(model.v0, model.theta))
    raise ValueError(f"no reference volatility for {type(model).__name__}; pass is_shift")


def default_is_shift(model: Model, product: Product) -> float:
    """Shift that centres the terminal price at the strike (median of ``S_T`` at ``K``)."""
    if not isinstance(product, EuropeanOption | DigitalOption | AsianOption | BarrierOption):
        raise ValueError(f"no default importance-sampling shift for {type(product).__name__}")
    sigma = reference_vol(model)
    t = product.maturity
    drift = (model.r - model.q - 0.5 * sigma**2) * t
    return (math.log(product.strike / model.s0) - drift) / (sigma * math.sqrt(t))


def build_time_grid(model: Model, product: Product, n_steps: int | None = None) -> FloatArray:
    """Simulation grid containing ``0`` and every monitoring date of ``product``.

    With ``n_steps=None`` the grid is exactly the monitoring dates (valid only for models
    with exact simulation). Otherwise it is ``linspace(0, T, n_steps + 1)``, which must
    contain the monitoring dates.
    """
    monitoring = np.asarray(product.monitoring_times(), dtype=np.float64)
    if n_steps is None:
        if not model.exact_simulation:
            raise ValueError(
                f"model {model.name!r} needs an explicit n_steps (time discretisation)"
            )
        return np.concatenate(([0.0], monitoring))
    require_int_at_least("n_steps", n_steps, 1)
    grid = np.linspace(0.0, product.maturity, n_steps + 1)
    time_indices(grid, monitoring, atol=1e-9 * product.maturity)
    return grid


def chunk_sizes(n_paths: int, chunk_size: int) -> Iterator[int]:
    """Yield chunk lengths summing to ``n_paths``."""
    remaining = n_paths
    while remaining > 0:
        size = min(chunk_size, remaining)
        yield size
        remaining -= size


def auto_chunk_size(n_times: int, n_factors: int) -> int:
    """Largest even chunk such that the normals and paths fit the memory budget."""
    size = _CHUNK_BUDGET // max(1, n_times * (n_factors + 1))
    return int(max(2, min(_MAX_CHUNK, size - size % 2)))


def simulate_discounted_payoffs(
    model: Model,
    product: Product,
    n_paths: int,
    *,
    n_steps: int | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Return the ``n_paths`` discounted payoffs (plain sampling, held in memory).

    Useful for convergence plots; use :func:`price_mc` for large runs.
    """
    require_int_at_least("n_paths", n_paths, 1)
    gen = make_rng(seed, rng)
    times = build_time_grid(model, product, n_steps)
    z = gen.standard_normal((n_paths, times.size - 1, model.n_factors))
    disc = math.exp(-model.r * product.maturity)
    return disc * product.payoff(model.paths_from_normals(times, z), times)


def price_mc(
    model: Model,
    product: Product,
    *,
    n_paths: int = 100_000,
    method: str = "plain",
    n_steps: int | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    chunk_size: int | None = None,
    control: ControlVariate | None = None,
    is_shift: float | None = None,
    n_scrambles: int = 16,
    bridge: bool = True,
) -> PricingResult:
    """Price ``product`` under ``model`` by Monte Carlo simulation.

    Parameters
    ----------
    model, product
        Asset model and payoff.
    n_paths
        Total number of simulated paths (``>= 2``; even for ``antithetic``; for ``qmc``
        ``n_scrambles`` times a power of two).
    method
        ``"plain"``, ``"antithetic"``, ``"cv"``, ``"is"`` or ``"qmc"``.
    n_steps
        Number of uniform time steps; ``None`` simulates on the monitoring dates only.
    seed, rng
        Randomness (pass at most one).
    chunk_size
        Paths per chunk; ``None`` picks a size that keeps memory around 64 MB.
    control
        Control variate for ``method="cv"`` (default: :func:`default_control`).
    is_shift
        Terminal drift shift ``mu`` for ``method="is"`` (default: :func:`default_is_shift`).
    n_scrambles, bridge
        Number of independent Sobol scramblings and Brownian-bridge construction for
        ``method="qmc"``.

    Returns
    -------
    PricingResult
        Price, standard error and 95 % confidence interval. ``diagnostics`` holds
        ``beta`` and ``vr_factor`` (variance ratio plain / controlled) for ``cv``, the
        shift and ``vr_factor`` for ``is``, and ``n_scrambles`` for ``qmc``.
    """
    started = time.perf_counter()
    require_choice("method", method, METHODS)
    require_int_at_least("n_paths", n_paths, 2)
    if isinstance(product, AmericanOption):
        raise ValueError("early-exercise products need price_lsm (Longstaff-Schwartz)")
    if method == "antithetic" and n_paths % 2:
        raise ValueError(f"antithetic sampling needs an even n_paths, got {n_paths}")
    gen = make_rng(seed, rng)
    times = build_time_grid(model, product, n_steps)
    n_t, n_f = times.size - 1, model.n_factors
    chunk = auto_chunk_size(times.size, n_f) if chunk_size is None else chunk_size
    require_int_at_least("chunk_size", chunk, 2)
    if method == "antithetic" and chunk % 2:
        raise ValueError("chunk_size must be even for antithetic sampling")
    disc = math.exp(-model.r * product.maturity)

    def discounted(z: FloatArray) -> tuple[FloatArray, FloatArray]:
        paths = model.paths_from_normals(times, z)
        return disc * product.payoff(paths, times), paths

    label = METHOD_LABELS[method]
    diagnostics: dict[str, float] = {}
    if method == "cv":
        cv = default_control(model, product) if control is None else control
        acc = StreamingMoments(dim=2)
        for size in chunk_sizes(n_paths, chunk):
            y, paths = discounted(gen.standard_normal((size, n_t, n_f)))
            acc.update(np.column_stack((y, cv.func(paths, times))))
        price, std_error, diagnostics = _control_variate_estimate(acc, cv.mean)
    elif method == "is":
        mu = default_is_shift(model, product) if is_shift is None else float(is_shift)
        shift = mu * np.sqrt(np.diff(times) / product.maturity)
        factor = model.price_factor
        acc = StreamingMoments(dim=2)
        for size in chunk_sizes(n_paths, chunk):
            z = gen.standard_normal((size, n_t, n_f))
            z[:, :, factor] += shift
            weight = np.exp(-(z[:, :, factor] @ shift) + 0.5 * float(shift @ shift))
            y = discounted(z)[0]
            acc.update(np.column_stack((y * weight, y * y * weight)))
        price, std_error = float(acc.mean[0]), acc.std_error()
        plain_var = float(acc.mean[1]) - price**2
        var = float(acc.variance[0])
        diagnostics = {"is_shift": mu, "vr_factor": plain_var / var if var > 0.0 else math.inf}
    elif method == "qmc":
        price, std_error = _qmc_estimate(
            discounted, times, n_f, n_paths, n_scrambles, chunk, bridge, gen
        )
        label = "qmc-sobol-bb" if bridge else "qmc-sobol"
        diagnostics = {"n_scrambles": float(n_scrambles)}
    else:
        acc = StreamingMoments(dim=1)
        for size in chunk_sizes(n_paths, chunk):
            if method == "antithetic":
                zh = gen.standard_normal((size // 2, n_t, n_f))
                y_plus, _ = discounted(zh)
                y_minus, _ = discounted(-zh)
                acc.update(0.5 * (y_plus + y_minus))
            else:
                acc.update(discounted(gen.standard_normal((size, n_t, n_f)))[0])
        price = float(acc.mean[0])
        std_error = acc.std_error()
    return _result(price, std_error, n_paths, n_t, label, started, diagnostics)


def _control_variate_estimate(
    acc: StreamingMoments, control_mean: float
) -> tuple[float, float, dict[str, float]]:
    cov, mean = acc.covariance, acc.mean
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0.0 else 0.0
    price = float(mean[0] - beta * (mean[1] - control_mean))
    var = max(float(cov[0, 0] - 2.0 * beta * cov[0, 1] + beta**2 * cov[1, 1]), 0.0)
    corr = (
        float(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1]))
        if cov[0, 0] > 0.0 and cov[1, 1] > 0.0
        else 0.0
    )
    diagnostics = {
        "beta": beta,
        "vr_factor": float(cov[0, 0] / var) if var > 0.0 else math.inf,
        "corr": corr,
    }
    return price, math.sqrt(var / acc.count), diagnostics


def _qmc_estimate(
    discounted: Callable[[FloatArray], tuple[FloatArray, FloatArray]],
    times: FloatArray,
    n_factors: int,
    n_paths: int,
    n_scrambles: int,
    chunk: int,
    bridge: bool,
    gen: np.random.Generator,
) -> tuple[float, float]:
    require_int_at_least("n_scrambles", n_scrambles, 2)
    per = n_paths // n_scrambles
    if per * n_scrambles != n_paths or not is_power_of_two(per):
        raise ValueError(
            f"qmc needs n_paths = n_scrambles x 2^m, got {n_paths} with {n_scrambles} scrambles"
        )
    chunk_q = min(per, 1 << (chunk.bit_length() - 1))  # largest power of two <= chunk
    estimates = np.empty(n_scrambles)
    for r, child in enumerate(spawn(gen, n_scrambles)):
        sampler = SobolNormals(times, n_factors, child, bridge)
        acc = StreamingMoments()
        for size in chunk_sizes(per, chunk_q):
            acc.update(discounted(sampler.draw(size))[0])
        estimates[r] = acc.mean[0]
    return float(estimates.mean()), float(estimates.std(ddof=1) / math.sqrt(n_scrambles))


def qmc_path_count(n_paths: int, n_scrambles: int = 16) -> int:
    """Largest valid QMC path count ``n_scrambles * 2^m`` not above ``n_paths``."""
    per = max(1, n_paths // n_scrambles)
    return n_scrambles * (1 << (per.bit_length() - 1))


def _result(
    price: float,
    std_error: float,
    n_paths: int,
    n_steps: int,
    method: str,
    started: float,
    diagnostics: dict[str, float],
) -> PricingResult:
    half = z_value(0.95) * std_error
    return PricingResult(
        price=price,
        std_error=std_error,
        ci_low=price - half,
        ci_high=price + half,
        n_paths=n_paths,
        n_steps=n_steps,
        method=method,
        elapsed_s=time.perf_counter() - started,
        diagnostics=diagnostics,
    )
