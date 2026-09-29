"""Build and execute the example notebooks in notebooks/ (outputs come from real runs).

Usage: ``python scripts/build_notebooks.py``
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

SETUP = """\
import numpy as np
import matplotlib.pyplot as plt
from mcengine.reporting.style import apply_style, PALETTE
apply_style()
%matplotlib inline"""

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_walkthrough.ipynb": [
        (
            "md",
            "# mcengine walkthrough\n\nPricing a European call with every estimator, "
            "then path-dependent and American options, each next to an independent "
            "reference.",
        ),
        ("code", SETUP),
        (
            "code",
            """\
from mcengine import GBM, EuropeanOption, price_mc, price_analytic
model = GBM(s0=100.0, r=0.05, sigma=0.2)
call = EuropeanOption(strike=100.0, maturity=1.0)
ref = price_analytic(model, call).price
for method in ("plain", "antithetic", "cv", "is", "qmc"):
    n = 16 * 2**13 if method == "qmc" else 100_000
    res = price_mc(model, call, n_paths=n, method=method, seed=7)
    print(f"{res.method:14s} {res.price:.5f} +/- {res.std_error:.5f}   "
          f"error {res.error_in_se(ref):+.2f} SE")
print(f"Black-Scholes  {ref:.5f}")""",
        ),
        ("md", "## Convergence of the plain estimator"),
        (
            "code",
            """\
from mcengine.engines.monte_carlo import simulate_discounted_payoffs
y = simulate_discounted_payoffs(model, call, 200_000, seed=1)
k = np.arange(1, y.size + 1)
mean = np.cumsum(y) / k
se = np.sqrt(np.maximum(np.cumsum(y**2) / k - mean**2, 0) / k)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.fill_between(k, mean - 1.96 * se, mean + 1.96 * se, alpha=0.2, color=PALETTE[0])
ax.plot(k, mean, color=PALETTE[0], label="running estimate")
ax.axhline(ref, color="k", ls="--", label="Black-Scholes")
ax.set_xscale("log"); ax.set_ylim(ref - 2, ref + 2)
ax.set_xlabel("paths"); ax.set_ylabel("price"); ax.legend();""",
        ),
        ("md", "## Arithmetic Asian: geometric control variate vs recursive convolution"),
        (
            "code",
            """\
from mcengine.products.asian import AsianOption
from mcengine.engines.convolution import asian_arithmetic_price
asian = AsianOption(100.0, 1.0, n_fixings=12)
ref_asian = asian_arithmetic_price(model, asian)
for method in ("plain", "cv"):
    res = price_mc(model, asian, n_paths=100_000, method=method, seed=3)
    print(res, f"error {res.error_in_se(ref_asian):+.2f} SE", res.diagnostics)
print("recursive convolution:", round(ref_asian, 6))""",
        ),
        ("md", "## Barrier: Brownian-bridge estimator vs Reiner-Rubinstein"),
        (
            "code",
            """\
from mcengine.products.barrier import BarrierOption
dao = BarrierOption(100.0, 1.0, 90.0, "down-and-out", "call", n_monitoring=50,
                    correction="bridge", correction_sigma=0.2)
res = price_mc(model, dao, n_paths=100_000, seed=5)
ref_b = price_analytic(model, dao).price
print(res, f"reference {ref_b:.5f}, error {res.error_in_se(ref_b):+.2f} SE")""",
        ),
        ("md", "## American put: Longstaff-Schwartz vs Bermudan CRR tree"),
        (
            "code",
            """\
from mcengine.products.american import AmericanOption
from mcengine.engines.lsm import price_lsm
from mcengine.engines.tree import price_tree
put = AmericanOption(40.0, 1.0, "put", n_exercise=50)
ls_model = GBM(36.0, 0.06, 0.2)
lsm = price_lsm(ls_model, put, n_paths=100_000, seed=11)
tree = price_tree(ls_model, put, n_steps=5000)
print(lsm, "| tree:", round(tree.price, 5), f"| error {lsm.error_in_se(tree.price):+.2f} SE")
print("in-sample (upward-biased) value:", round(lsm.diagnostics["in_sample_price"], 5))""",
        ),
    ],
    "02_heston_smile.ipynb": [
        ("md", "# Heston: QE simulation, Fourier pricing, smile and calibration"),
        ("code", SETUP),
        (
            "code",
            """\
from mcengine.models import Heston
from mcengine import EuropeanOption, price_mc
from mcengine.engines.fourier import gil_pelaez_price, carr_madan_prices, fourier_prices
from mcengine.volatility import implied_vol
h = Heston(s0=100.0, r=0.03, v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
print("Feller ratio:", round(h.feller_ratio, 3))
gp = gil_pelaez_price(h, 100.0, 1.0)
print("Gil-Pelaez:", gp, " Carr-Madan:", carr_madan_prices(h, [100.0], 1.0)[0])
for scheme in ("qe", "euler"):
    res = price_mc(h.with_scheme(scheme), EuropeanOption(100.0, 1.0), n_paths=100_000,
                   n_steps=50, seed=1)
    print(scheme, res, f"error {res.error_in_se(gp):+.2f} SE")""",
        ),
        ("md", "## Implied-volatility smile across maturities"),
        (
            "code",
            """\
m = np.linspace(0.7, 1.3, 41)
fig, ax = plt.subplots(figsize=(7, 3.8))
for i, t in enumerate((0.25, 0.5, 1.0, 2.0)):
    vols = implied_vol(fourier_prices(h, 100 * m, t), 100.0, 100 * m, t, h.r)
    ax.plot(m, vols, color=PALETTE[i], label=f"T = {t}")
ax.set_xlabel("K / S0"); ax.set_ylabel("implied vol"); ax.legend();""",
        ),
        ("md", "## Calibration to a noisy synthetic surface"),
        (
            "code",
            """\
from mcengine.calibration import synthetic_surface, calibrate_heston
true = Heston(100.0, 0.02, 0.04, 1.5, 0.05, 0.6, -0.7)
surface = synthetic_surface(true, noise_vol=0.001, seed=1)
fit = calibrate_heston(surface, seed=0)
for name, value in fit.params.items():
    print(f"{name:6s} true {getattr(true, name):+.4f}   fitted {value:+.4f}")
print(f"RMSE {fit.rmse_vol * 1e4:.2f} bp, Feller ratio {fit.feller_ratio:.3f}")""",
        ),
    ],
    "03_greeks_hedging.ipynb": [
        ("md", "# Greeks estimators and discrete delta hedging"),
        ("code", SETUP),
        (
            "code",
            """\
from mcengine import GBM
from mcengine.products.european import EuropeanOption, DigitalOption
from mcengine.greeks.analytic import bs_greeks, digital_greeks
from mcengine.greeks.monte_carlo import mc_greeks
model = GBM(100.0, 0.05, 0.2)
cases = [(EuropeanOption(100.0, 1.0), bs_greeks(100, 100, 1, 0.05, 0.2)),
         (DigitalOption(100.0, 1.0), digital_greeks(100, 100, 1, 0.05, 0.2))]
for product, ref in cases:
    print(product.label)
    for method in ("bump", "pathwise", "lr"):
        g = mc_greeks(model, product, method=method, n_paths=200_000, seed=2)
        print(f"  {method:9s}" + "".join(
            f"  {k} {getattr(g, k):+.5f} (SE {g.std_errors[k]:.1e}, exact {getattr(ref, k):+.5f})"
            for k in ("delta", "gamma")))""",
        ),
        (
            "md",
            "The pathwise estimator returns exactly zero for the digital option: the "
            "payoff is flat almost everywhere, so differentiating inside the expectation "
            "misses the discontinuity. The likelihood-ratio estimator does not need a "
            "differentiable payoff.",
        ),
        ("md", "## Hedging error vs rebalancing frequency"),
        (
            "code",
            """\
from mcengine.hedging import hedging_experiment, fit_std_slope
from mcengine.models import Heston
freqs = [1, 2, 4, 12, 21, 42, 126, 252]
gbm = hedging_experiment(model, 100.0, 1.0, freqs, hedge_sigma=0.2, n_paths=20_000, seed=3)
heston = hedging_experiment(Heston(100.0, 0.05, 0.04, 2.0, 0.04, 0.5, -0.7), 100.0, 1.0,
                            freqs, hedge_sigma=0.2, n_paths=20_000, seed=3)
fig, ax = plt.subplots(figsize=(7, 3.8))
for i, (name, res) in enumerate((("GBM", gbm), ("Heston (misspecified)", heston))):
    ax.loglog(freqs, [r.std for r in res], "o-", color=PALETTE[i],
              label=f"{name}: slope {fit_std_slope(res):.2f}")
ax.set_xlabel("rebalancing dates N"); ax.set_ylabel("std of P&L"); ax.legend();""",
        ),
    ],
}


def build(name: str, cells: list[tuple[str, str]]) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.cells = [
        nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
        for kind, src in cells
    ]
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(OUT)}}
    )
    nbformat.write(nb, OUT / name)
    print("executed", name)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        build(name, cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
