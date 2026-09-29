r"""Discrete delta hedging of a short European option.

A trader sells one option at the Black-Scholes price :math:`V_0 = \mathrm{BS}(S_0,
\sigma_h)` and hedges it with :math:`\Delta_i = \partial_S\mathrm{BS}(S_{t_i}, T - t_i,
\sigma_h)` shares, rebalanced at :math:`N` equally spaced dates
:math:`t_i = iT/N,\ i = 0..N-1`. The self-financing cash account accrues at the rate
:math:`r`, receives the dividend yield on the shares held, and pays proportional
transaction costs :math:`c\,|\Delta_i - \Delta_{i-1}|\,S_{t_i}` (including the initial
purchase and the final liquidation). The discounted terminal profit and loss is

.. math::

    \mathrm{PnL} = e^{-rT}\Big(B_T + \Delta_{N-1}S_T(1 - c) - f(S_T)\Big).

Under GBM with :math:`\sigma_h = \sigma` and no costs, the P&L is a pure discretisation
error with mean close to zero and standard deviation decaying like :math:`N^{-1/2}`
(Boyle & Emanuel, 1980; Bertsimas, Kogan & Lo, 2000). Under Heston the Black-Scholes
hedge is misspecified: the P&L keeps a volatility-risk component that does not vanish as
:math:`N \to \infty`.

All rebalancing frequencies are evaluated on the **same** simulated paths (a fine grid
whose number of steps is a multiple of every :math:`N`), so the comparison across
:math:`N` uses common random numbers.

References
----------
Boyle, P. P., & Emanuel, D. (1980). Discretely adjusted option hedges. *Journal of
Financial Economics*, 8(3), 259-282.
Bertsimas, D., Kogan, L., & Lo, A. W. (2000). When is time continuous? *Journal of
Financial Economics*, 55(2), 173-204.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mcengine._typing import FloatArray
from mcengine._validation import (
    require_choice,
    require_int_at_least,
    require_non_negative,
    require_positive,
)
from mcengine.engines.analytic import bs_price
from mcengine.greeks.analytic import bs_delta
from mcengine.models.base import Model
from mcengine.products.base import OPTION_TYPES, vanilla_payoff
from mcengine.random.generators import make_rng


@dataclass(frozen=True)
class HedgeResult:
    """Discounted P&L of the short-option hedge for one rebalancing frequency."""

    n_rebalance: int
    pnl: FloatArray
    premium: float
    cost_rate: float

    @property
    def mean(self) -> float:
        """Mean discounted P&L."""
        return float(self.pnl.mean())

    @property
    def std(self) -> float:
        """Standard deviation of the discounted P&L."""
        return float(self.pnl.std(ddof=1))

    @property
    def std_error(self) -> float:
        """Standard error of :attr:`mean`."""
        return self.std / math.sqrt(self.pnl.size)

    def relative_std(self) -> float:
        """Standard deviation as a fraction of the premium received."""
        return self.std / self.premium


def hedge_pnl(
    paths: FloatArray,
    times: FloatArray,
    strike: float,
    r: float,
    q: float,
    hedge_sigma: float,
    n_rebalance: int,
    option_type: str = "call",
    cost_rate: float = 0.0,
) -> HedgeResult:
    """P&L of hedging on ``n_rebalance`` dates along pre-simulated ``paths``.

    ``times`` is the simulation grid (``times[0] = 0``, ``times[-1] = T``); its number of
    steps must be a multiple of ``n_rebalance``.
    """
    require_choice("option_type", option_type, OPTION_TYPES)
    require_int_at_least("n_rebalance", n_rebalance, 1)
    require_positive("hedge_sigma", hedge_sigma)
    require_non_negative("cost_rate", cost_rate)
    n_steps = times.size - 1
    if n_steps % n_rebalance:
        raise ValueError(f"{n_steps} simulation steps are not a multiple of {n_rebalance}")
    stride = n_steps // n_rebalance
    maturity = float(times[-1])
    s0 = float(paths[0, 0])
    premium = float(bs_price(s0, strike, maturity, r, hedge_sigma, q, option_type))

    idx = np.arange(0, n_steps, stride)
    delta_prev = np.zeros(paths.shape[0])
    cash = np.full(paths.shape[0], premium)
    t_prev = 0.0
    for j, i in enumerate(idx):
        t, spot = float(times[i]), paths[:, i]
        if j > 0:
            dt = t - t_prev
            # interest on cash, dividends on the shares held over the period
            cash = cash * math.exp(r * dt) + delta_prev * spot * math.expm1(q * dt)
        delta = np.asarray(bs_delta(spot, strike, maturity - t, r, hedge_sigma, q, option_type))
        trade = delta - delta_prev
        cash -= trade * spot + cost_rate * np.abs(trade) * spot
        delta_prev, t_prev = delta, t
    s_t = paths[:, -1]
    dt = maturity - t_prev
    cash = cash * math.exp(r * dt) + delta_prev * s_t * math.expm1(q * dt)
    liquidation = delta_prev * s_t - cost_rate * np.abs(delta_prev) * s_t
    pnl = math.exp(-r * maturity) * (cash + liquidation - vanilla_payoff(s_t, strike, option_type))
    return HedgeResult(n_rebalance, np.asarray(pnl, dtype=np.float64), premium, cost_rate)


def hedging_experiment(
    model: Model,
    strike: float,
    maturity: float,
    n_rebalances: Sequence[int],
    *,
    hedge_sigma: float,
    option_type: str = "call",
    cost_rate: float = 0.0,
    n_paths: int = 20_000,
    n_steps: int = 252,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[HedgeResult]:
    """Hedge on each frequency in ``n_rebalances`` using the same ``n_paths`` paths.

    Paths are simulated once on ``n_steps`` uniform steps, which must be a multiple of
    every rebalancing frequency (252 is divisible by 1, 2, 3, 4, 6, 7, 9, 12, 14, 18, 21,
    28, 36, 42, 63, 84, 126, 252).
    """
    require_positive("strike", strike)
    require_positive("maturity", maturity)
    require_int_at_least("n_paths", n_paths, 2)
    require_int_at_least("n_steps", n_steps, 1)
    times = np.linspace(0.0, maturity, n_steps + 1)
    gen = make_rng(seed, rng)
    paths = model.paths_from_normals(
        times, gen.standard_normal((n_paths, n_steps, model.n_factors))
    )
    return [
        hedge_pnl(paths, times, strike, model.r, model.q, hedge_sigma, n, option_type, cost_rate)
        for n in n_rebalances
    ]


def fit_std_slope(results: Sequence[HedgeResult]) -> float:
    """Least-squares slope of ``log(std)`` against ``log(N)``."""
    n = np.log([res.n_rebalance for res in results])
    s = np.log([res.std for res in results])
    return float(np.polyfit(n, s, 1)[0])
