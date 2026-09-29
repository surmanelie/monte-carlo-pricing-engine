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
from typing import Any

import numpy as np

from mcengine.engines.analytic import barrier_price, bs_price
from mcengine.engines.fourier import fourier_prices, gil_pelaez_price
from mcengine.engines.lsm import fit_lsm
from mcengine.engines.monte_carlo import price_mc, simulate_discounted_payoffs
from mcengine.engines.tree import crr_exercise_boundary
from mcengine.models.base import Model
from mcengine.models.gbm import GBM
from mcengine.models.heston import Heston
from mcengine.products.american import AmericanOption
from mcengine.products.barrier import BarrierOption
from mcengine.products.european import EuropeanOption
from mcengine.reporting.style import (
    INK,
    INK_SECONDARY,
    METHOD_COLORS,
    PALETTE,
    new_figure,
    save_figure,
)
from mcengine.reporting.validation import HESTON, MERTON
from mcengine.stats import z_value
from mcengine.volatility.implied import implied_vol

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


def fig_barrier_monitoring(outdir: Path, quick: bool = False) -> dict[str, float]:
    """Down-and-out call vs number of monitoring dates: raw, BGK, Brownian bridge, continuous."""
    n_paths = 20_000 if quick else 200_000
    dates = [4, 16] if quick else [4, 8, 16, 32, 64, 128, 256, 512]
    k, h, sigma = 100.0, 95.0, _BS.sigma
    continuous = barrier_price(_BS.s0, k, h, 1.0, _BS.r, sigma)
    fig, ax = new_figure()
    out: dict[str, float] = {"continuous": continuous, "n_paths": float(n_paths)}
    styles = {
        "none": ("Discrete monitoring (raw)", PALETTE[1], "o"),
        "bgk": ("BGK continuity correction", PALETTE[2], "s"),
        "bridge": ("Brownian-bridge estimator", PALETTE[0], "D"),
    }
    for offset, (correction, (label, color, marker)) in enumerate(styles.items()):
        prices, half = [], []
        for i, m in enumerate(dates):
            product = BarrierOption(
                k,
                1.0,
                h,
                "down-and-out",
                "call",
                m,
                correction,
                None if correction == "none" else sigma,
            )
            res = price_mc(_BS, product, n_paths=n_paths, method="cv", seed=5000 + 100 * offset + i)
            assert res.std_error is not None
            prices.append(res.price)
            half.append(z_value(0.95) * res.std_error)
        ax.errorbar(
            dates,
            prices,
            yerr=half,
            color=color,
            marker=marker,
            capsize=3,
            label=label,
            markersize=6,
        )
        out[f"{correction}_m{dates[0]}"] = prices[0]
        out[f"{correction}_m{dates[-1]}"] = prices[-1]
    ax.axhline(
        continuous, color=INK, linestyle="--", linewidth=1.2, label="Continuous (Reiner-Rubinstein)"
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of monitoring dates $m$ (log scale)")
    ax.set_ylabel("Option price")
    ax.set_title(f"Down-and-out call ($K = {k:g}$, $H = {h:g}$) vs monitoring frequency")
    ax.legend()
    save_figure(fig, outdir / "barrier_monitoring.png")
    return out


def fig_exercise_boundary(outdir: Path, quick: bool = False) -> dict[str, float]:
    """American put early-exercise boundary: Longstaff-Schwartz policy vs CRR tree."""
    model = GBM(36.0, 0.06, 0.2)
    product = AmericanOption(40.0, 1.0, "put", 50)
    n_steps = 500 if quick else 5000
    n_train = 20_000 if quick else 200_000
    args = (36.0, 40.0, 1.0, 0.06, 0.2)
    times, american = crr_exercise_boundary(*args, n_steps=n_steps)
    dates = product.monitoring_times()
    _, bermudan_all = crr_exercise_boundary(
        *args, n_steps=n_steps, style="bermudan", exercise_times=dates
    )
    bermudan = bermudan_all[np.rint(dates / 1.0 * n_steps).astype(int)]
    fig, ax = new_figure()
    ax.plot(
        times,
        american,
        color=INK_SECONDARY,
        linewidth=1.0,
        label=f"CRR tree, American ($N = {n_steps}$)",
    )
    ax.plot(
        dates,
        bermudan,
        color=INK,
        linewidth=1.5,
        marker="_",
        markersize=8,
        label="CRR tree, Bermudan (50 dates)",
    )
    out: dict[str, float] = {"n_train": float(n_train)}
    for offset, degree in enumerate((3, 6)):
        fit = fit_lsm(model, product, n_paths=n_train, degree=degree, seed=77 + offset)
        boundary = fit.exercise_boundary(product)
        ax.plot(
            dates,
            boundary,
            "o",
            color=PALETTE[offset],
            markersize=5,
            label=f"LSM, Laguerre degree {degree} ({n_train:,} paths)",
        )
        valid = ~np.isnan(boundary[:-1]) & ~np.isnan(bermudan[:-1])
        out[f"mean_abs_gap_deg{degree}"] = float(
            np.mean(np.abs(boundary[:-1][valid] - bermudan[:-1][valid]))
        )
    ax.axhline(40.0, color=INK_SECONDARY, linestyle=":", linewidth=1.0, label="Strike $K = 40$")
    ax.set_xlabel("Time $t$ (years)")
    ax.set_ylabel("Critical stock price $S^*(t)$")
    ax.set_title("Early-exercise boundary of the American put ($S_0 = 36$, $\\sigma = 0.2$)")
    ax.set_ylim(top=42.5)
    ax.legend(loc="upper left", ncols=2, fontsize=8)
    save_figure(fig, outdir / "exercise_boundary.png")
    return out


def _smile_panel(
    ax: Any, model: Model, title: str, mc_steps: int | None, n_paths: int, seed: int
) -> dict[str, float]:
    """Fourier implied-vol curves per maturity plus MC points (95 % CI) at T = 1."""
    moneyness = np.linspace(0.7, 1.3, 41)
    strikes = model.s0 * moneyness
    out: dict[str, float] = {}
    maturities = (0.25, 0.5, 1.0, 2.0)
    for i, maturity in enumerate(maturities):
        prices = fourier_prices(model, strikes, maturity)
        vols = np.asarray(implied_vol(prices, model.s0, strikes, maturity, model.r, model.q))
        ax.plot(moneyness, vols, color=PALETTE[i], label=f"$T = {maturity:g}$ (Fourier)")
        out[f"atm_vol_T{maturity:g}"] = float(np.interp(1.0, moneyness, vols))
        out[f"skew_T{maturity:g}"] = float(vols[0] - vols[-1])
    mc_moneyness = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    mc_vol, mc_lo, mc_hi = [], [], []
    for j, m in enumerate(mc_moneyness):
        product = EuropeanOption(model.s0 * m, 1.0, "put" if m < 1.0 else "call")
        res = price_mc(model, product, n_paths=n_paths, n_steps=mc_steps, seed=seed + j)
        assert res.ci_low is not None
        assert res.ci_high is not None
        iv = implied_vol(
            [res.price, res.ci_low, res.ci_high],
            model.s0,
            product.strike,
            1.0,
            model.r,
            model.q,
            product.option_type,
        )
        v = np.asarray(iv)
        mc_vol.append(v[0])
        mc_lo.append(v[0] - v[1])
        mc_hi.append(v[2] - v[0])
    ax.errorbar(
        mc_moneyness,
        mc_vol,
        yerr=[mc_lo, mc_hi],
        fmt="o",
        color=INK,
        markersize=5,
        capsize=3,
        label=f"Monte Carlo, $T = 1$ (95 % CI, {n_paths:,} paths)",
    )
    ax.set_xlabel("Moneyness $K / S_0$")
    ax.set_ylabel("Black-Scholes implied volatility")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return out


def fig_smiles(outdir: Path, quick: bool = False) -> dict[str, float]:
    """Implied-volatility smiles/skews of the Heston and Merton models."""
    n_paths = 20_000 if quick else 200_000
    fig, axes = new_figure(1, 2, width=11.0, height=4.4)
    out = {}
    heston = _smile_panel(axes[0], HESTON, "Heston (QE, $\\Delta t = 1/50$)", 50, n_paths, 900)
    merton = _smile_panel(axes[1], MERTON, "Merton jump-diffusion", None, n_paths, 950)
    out.update({f"heston_{k}": v for k, v in heston.items()})
    out.update({f"merton_{k}": v for k, v in merton.items()})
    save_figure(fig, outdir / "smiles.png")
    return out


def fig_heston_scheme_bias(outdir: Path, quick: bool = False) -> dict[str, float]:
    """QE vs full-truncation Euler bias vs time step in a Feller-violating stress case."""
    model = Heston(100.0, 0.0, 0.04, 0.5, 0.04, 1.0, -0.9)
    maturity = 10.0
    product = EuropeanOption(100.0, maturity)
    ref = gil_pelaez_price(model, 100.0, maturity)
    n_paths = 10_000 if quick else 400_000
    steps_per_year = [1, 4] if quick else [1, 2, 4, 8, 16, 32]
    dts = 1.0 / np.array(steps_per_year, dtype=float)
    fig, ax = new_figure()
    out: dict[str, float] = {"reference": ref, "feller_ratio": model.feller_ratio}
    for offset, (scheme, label) in enumerate(
        (("qe", "Andersen QE"), ("euler", "Full-truncation Euler"))
    ):
        errors, half = [], []
        for i, spy in enumerate(steps_per_year):
            res = price_mc(
                model.with_scheme(scheme),
                product,
                n_paths=n_paths,
                n_steps=int(spy * maturity),
                seed=7000 + 100 * offset + i,
            )
            assert res.std_error is not None
            errors.append(res.price - ref)
            half.append(z_value(0.95) * res.std_error)
            out[f"{scheme}_bias_dt{1 / spy:g}"] = res.price - ref
            out[f"{scheme}_se_dt{1 / spy:g}"] = res.std_error
        ax.errorbar(
            dts,
            errors,
            yerr=half,
            color=PALETTE[offset],
            marker="os"[offset],
            capsize=3,
            markersize=6,
            label=label,
        )
    ax.axhline(0.0, color=INK, linestyle="--", linewidth=1.2, label="Fourier reference")
    ax.set_xscale("log", base=2)
    ax.invert_xaxis()
    ax.set_xlabel("Time step $\\Delta t$ (years, log scale)")
    ax.set_ylabel("Price error vs Fourier (95 % CI)")
    ax.set_title(
        f"Heston ATM call, $T = 10$, Feller ratio {model.feller_ratio:.2f}: discretisation bias"
    )
    ax.legend()
    save_figure(fig, outdir / "heston_scheme_bias.png")
    return out


FIGURES: dict[str, FigureFunc] = {
    "convergence": fig_convergence,
    "error_vs_n": fig_error_vs_n,
    "barrier_monitoring": fig_barrier_monitoring,
    "exercise_boundary": fig_exercise_boundary,
    "smiles": fig_smiles,
    "heston_scheme_bias": fig_heston_scheme_bias,
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
