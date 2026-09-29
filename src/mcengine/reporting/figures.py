"""Figure generation. Every figure is produced by running the library.

Each ``fig_*`` function writes one PNG into ``outdir`` and returns the key numbers it
measured (e.g. fitted convergence slopes). :func:`make_all` stores those numbers in
``results/figures.json`` so that the README and docs quote measured values only.
``quick=True`` shrinks every experiment for smoke tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from mcengine.engines.analytic import bs_price
from mcengine.engines.monte_carlo import price_mc, simulate_discounted_payoffs
from mcengine.models.gbm import GBM
from mcengine.products.european import EuropeanOption
from mcengine.reporting.style import INK, METHOD_COLORS, new_figure, save_figure
from mcengine.stats import z_value

FigureFunc = Callable[[Path, bool], dict[str, float]]

_BS = GBM(s0=100.0, r=0.05, sigma=0.2)
_CALL = EuropeanOption(100.0, 1.0, "call")
_BS_CALL = float(bs_price(100.0, 100.0, 1.0, 0.05, 0.2))


def fig_convergence(outdir: Path, quick: bool = False) -> dict[str, float]:
    """Running MC estimate of the ATM call with its 95 % confidence band."""
    n = 10_000 if quick else 1_000_000
    y = simulate_discounted_payoffs(_BS, _CALL, n, seed=20240101)
    k = np.arange(1, n + 1)
    mean = np.cumsum(y) / k
    var = np.maximum(np.cumsum(y**2) / k - mean**2, 0.0) * k / np.maximum(k - 1, 1)
    half = z_value(0.95) * np.sqrt(var / k)
    idx = np.unique(np.geomspace(100, n, 400).astype(int)) - 1

    fig, ax = new_figure()
    color = METHOD_COLORS["mc-plain"]
    ax.fill_between(
        k[idx],
        (mean - half)[idx],
        (mean + half)[idx],
        color=color,
        alpha=0.18,
        linewidth=0,
        label="95 % confidence band",
    )
    ax.plot(k[idx], mean[idx], color=color, label="Monte Carlo estimate (plain)")
    ax.axhline(_BS_CALL, color=INK, linestyle="--", linewidth=1.2, label="Black-Scholes price")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths $n$")
    ax.set_ylabel("Call price")
    ax.set_title("Convergence of the Monte Carlo price (European call, $S_0 = K = 100$, $T = 1$)")
    ax.legend(loc="upper right")
    save_figure(fig, outdir / "convergence.png")
    return {"n": float(n), "final_estimate": float(mean[-1]), "final_half_width": float(half[-1])}


def fig_error_vs_n(outdir: Path, quick: bool = False) -> dict[str, float]:
    """Root-mean-square error vs ``n`` over independent seeds, log-log with fitted slopes."""
    n_seeds = 4 if quick else 20
    exponents = range(8, 12) if quick else range(8, 19)
    ns = np.array([2**e for e in exponents])
    fig, ax = new_figure()
    out: dict[str, float] = {"n_seeds": float(n_seeds)}
    for offset, method in enumerate(("plain", "antithetic", "cv")):
        label = f"mc-{method}"
        rmse = np.empty(ns.size)
        for i, n in enumerate(ns):
            errs = [
                price_mc(
                    _BS, _CALL, n_paths=int(n), method=method, seed=1000 * offset + 97 * i + s
                ).price
                - _BS_CALL
                for s in range(n_seeds)
            ]
            rmse[i] = np.sqrt(np.mean(np.square(errs)))
        slope, intercept = np.polyfit(np.log(ns), np.log(rmse), 1)
        out[f"slope_{method}"] = float(slope)
        color = METHOD_COLORS[label]
        ax.plot(ns, rmse, "o", color=color, markersize=6)
        ax.plot(
            ns,
            np.exp(intercept) * ns**slope,
            color=color,
            label=f"{method} (fitted slope {slope:.3f})",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of simulated paths $n$")
    ax.set_ylabel(f"RMSE over {n_seeds} seeds")
    ax.set_title(r"Error vs number of paths: $O(n^{-1/2})$ convergence (European call)")
    ax.legend()
    save_figure(fig, outdir / "error_vs_n.png")
    return out


FIGURES: dict[str, FigureFunc] = {
    "convergence": fig_convergence,
    "error_vs_n": fig_error_vs_n,
}


def make_all(
    outdir: Path,
    results_dir: Path | None = None,
    quick: bool = False,
    only: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, dict[str, float]]:
    """Generate every figure (or the subset ``only``) and store measured numbers."""
    names = list(FIGURES) if only is None else only
    unknown = set(names) - set(FIGURES)
    if unknown:
        raise ValueError(f"unknown figures: {sorted(unknown)}")
    numbers: dict[str, dict[str, float]] = {}
    if results_dir is not None and (results_dir / "figures.json").exists():
        numbers = json.loads((results_dir / "figures.json").read_text(encoding="utf-8"))
    for name in names:
        if progress is not None:
            progress(name)
        numbers[name] = FIGURES[name](outdir, quick)
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "figures.json").write_text(
            json.dumps(numbers, indent=2, sort_keys=True), encoding="utf-8"
        )
    return numbers
