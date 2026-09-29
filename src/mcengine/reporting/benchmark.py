"""Benchmarks: throughput per model/backend and error vs wall-clock time per method.

* :func:`throughput` times complete pricings (simulation + payoff + statistics) and
  reports paths per second and path-steps per second, best of several repeats.
* :func:`error_vs_time` measures, for each estimator, the root-mean-square error over
  independent seeds against the reference price as a function of wall-clock time.
* :func:`efficiency` compares estimators by the product ``variance x time`` (the
  inverse of the work-normalised precision, Glasserman 2003, Section 1.1.3), relative
  to plain Monte Carlo.

All numbers depend on the machine; :func:`machine_info` records it next to the results.
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np

from mcengine import __version__
from mcengine._numba import HAS_NUMBA
from mcengine.engines.analytic import bs_price
from mcengine.engines.convolution import asian_arithmetic_price
from mcengine.engines.lsm import price_lsm
from mcengine.engines.monte_carlo import price_mc, qmc_path_count
from mcengine.models.base import Model
from mcengine.models.gbm import GBM
from mcengine.models.heston import Heston
from mcengine.models.merton import Merton
from mcengine.products.american import AmericanOption
from mcengine.products.asian import AsianOption
from mcengine.products.base import Product
from mcengine.products.european import EuropeanOption
from mcengine.reporting.style import INK_SECONDARY, METHOD_COLORS, new_figure, save_figure
from mcengine.results import PricingResult

BENCH_METHODS: tuple[str, ...] = ("plain", "antithetic", "cv", "qmc", "is")
_LABELS = {
    "plain": "mc-plain",
    "antithetic": "mc-antithetic",
    "cv": "mc-cv",
    "qmc": "qmc-sobol-bb",
    "is": "mc-is",
}


def machine_info() -> dict[str, str]:
    """Interpreter, platform and library versions for the benchmark report."""
    import scipy

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": str(os.cpu_count()),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": _numba_version(),
        "mcengine": __version__,
    }


def _numba_version() -> str:
    if not HAS_NUMBA:  # pragma: no cover - depends on the environment
        return "not installed"
    import numba

    return str(numba.__version__)


def _best_time(func: Callable[[], object], repeats: int) -> float:
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


@dataclass(frozen=True)
class ThroughputRow:
    """Timing of one configuration."""

    model: str
    engine: str
    backend: str
    n_steps: int
    n_paths: int
    seconds: float
    paths_per_second: float
    path_steps_per_second: float


def throughput(quick: bool = False, repeats: int = 3) -> list[ThroughputRow]:
    """Paths per second for each model / scheme / backend (European call, T = 1)."""
    scale = 0.05 if quick else 1.0
    call = EuropeanOption(100.0, 1.0)
    heston = Heston(100.0, 0.03, 0.04, 2.0, 0.04, 0.5, -0.7)
    configs: list[tuple[str, str, str, int, int, Callable[[int], object]]] = []

    def mc(model: Model, steps: int | None) -> Callable[[int], object]:
        return lambda n: price_mc(model, call, n_paths=n, n_steps=steps, seed=1)

    gbm = GBM(100.0, 0.05, 0.2)
    configs.append(("GBM (exact)", "mc-plain", "numpy", 1, 1_000_000, mc(gbm, None)))
    configs.append(("GBM (exact)", "mc-plain", "numpy", 252, 100_000, mc(gbm, 252)))
    configs.append(("Heston QE", "mc-plain", "numpy", 252, 100_000, mc(heston, 252)))
    if HAS_NUMBA:  # pragma: no branch - depends on the environment
        fast = Heston(100.0, 0.03, 0.04, 2.0, 0.04, 0.5, -0.7, backend="numba")
        price_mc(fast, call, n_paths=64, n_steps=4, seed=0)  # JIT compilation outside timing
        configs.append(("Heston QE", "mc-plain", "numba", 252, 100_000, mc(fast, 252)))
    euler = heston.with_scheme("euler")
    configs.append(("Heston Euler FT", "mc-plain", "numpy", 252, 100_000, mc(euler, 252)))
    merton = Merton(100.0, 0.05, 0.2, 1.0, -0.1, 0.15)
    configs.append(("Merton (exact)", "mc-plain", "numpy", 252, 100_000, mc(merton, 252)))
    put = AmericanOption(40.0, 1.0, "put", 50)
    lsm_model = GBM(36.0, 0.06, 0.2)

    def lsm(backend: str) -> Callable[[int], object]:
        return lambda n: price_lsm(lsm_model, put, n_paths=n, seed=1, backend=backend)

    backends = ("numpy", "numba") if HAS_NUMBA else ("numpy",)
    for backend in backends:
        if backend == "numba":
            price_lsm(lsm_model, put, n_paths=64, seed=0, backend="numba")
        configs.append(("GBM, American put", "lsm", backend, 50, 100_000, lsm(backend)))
    rows = []
    for model, engine, backend, steps, n, run in configs:
        n_paths = max(1000, int(n * scale))
        seconds = _best_time(partial(run, n_paths), repeats)
        rate = n_paths / seconds
        rows.append(
            ThroughputRow(model, engine, backend, steps, n_paths, seconds, rate, rate * steps)
        )
    return rows


@dataclass(frozen=True)
class ErrorTimePoint:
    """RMSE over seeds and mean wall-clock time of one (method, n) pair."""

    problem: str
    method: str
    n_paths: int
    rmse: float
    seconds: float


def _problems() -> dict[str, tuple[Model, Product, float]]:
    gbm = GBM(100.0, 0.05, 0.2)
    otm = EuropeanOption(140.0, 1.0)
    asian = AsianOption(100.0, 1.0, 12)
    return {
        "European call K=140": (gbm, otm, float(bs_price(100.0, 140.0, 1.0, 0.05, 0.2))),
        "Arithmetic Asian call K=100, m=12": (gbm, asian, asian_arithmetic_price(gbm, asian)),
    }


def _run(method: str, model: Model, product: Product, n: int, seed: int) -> PricingResult:
    n_paths = qmc_path_count(n) if method == "qmc" else n
    return price_mc(model, product, n_paths=n_paths, method=method, seed=seed)


def error_vs_time(quick: bool = False, n_seeds: int = 20) -> list[ErrorTimePoint]:
    """RMSE vs time for every method on each benchmark problem."""
    seeds = 3 if quick else n_seeds
    exponents = range(10, 13) if quick else range(12, 21)
    points = []
    for problem, (model, product, ref) in _problems().items():
        for m_idx, method in enumerate(BENCH_METHODS):
            _run(method, model, product, 1024, 0)  # warm-up (caches, imports)
            for e in exponents:
                n = 2**e
                errors, times = [], []
                for s in range(seeds):
                    res = _run(method, model, product, n, 10_000 * m_idx + 100 * e + s)
                    errors.append(res.price - ref)
                    times.append(res.elapsed_s)
                points.append(
                    ErrorTimePoint(
                        problem,
                        _LABELS[method],
                        qmc_path_count(n) if method == "qmc" else n,
                        float(np.sqrt(np.mean(np.square(errors)))),
                        float(np.mean(times)),
                    )
                )
    return points


@dataclass(frozen=True)
class EfficiencyRow:
    """Variance x time of one estimator relative to plain Monte Carlo."""

    problem: str
    method: str
    n_paths: int
    std_error: float
    seconds: float
    variance_reduction: float
    efficiency_gain: float


def efficiency(quick: bool = False) -> list[EfficiencyRow]:
    """Efficiency gain ``(se_plain^2 t_plain) / (se^2 t)`` at a fixed path budget."""
    n = 2**14 if quick else 2**20
    rows = []
    for problem, (model, product, _) in _problems().items():
        base: tuple[float, float] | None = None
        for method in BENCH_METHODS:
            _run(method, model, product, 1024, 0)
            res = _run(method, model, product, n, 123)
            assert res.std_error is not None
            var_t = res.std_error**2 * res.elapsed_s
            if base is None:
                base = (res.std_error**2, var_t)
            rows.append(
                EfficiencyRow(
                    problem,
                    res.method,
                    res.n_paths or n,
                    res.std_error,
                    res.elapsed_s,
                    base[0] / res.std_error**2,
                    base[1] / var_t,
                )
            )
    return rows


def plot_error_vs_time(points: list[ErrorTimePoint], outdir: Path) -> Path:
    """The key efficiency figure: RMSE against wall-clock time, one panel per problem."""
    problems = list(dict.fromkeys(p.problem for p in points))
    fig, axes = new_figure(1, len(problems), width=11.0, height=4.4)
    for ax, problem in zip(np.atleast_1d(axes), problems, strict=True):
        for method in _LABELS.values():
            pts = [p for p in points if p.problem == problem and p.method == method]
            t = np.array([p.seconds for p in pts])
            e = np.array([p.rmse for p in pts])
            ax.plot(t, e, "o-", color=METHOD_COLORS[method], label=method, markersize=5)
        t_all = np.array([p.seconds for p in points if p.problem == problem])
        e_plain = [p for p in points if p.problem == problem and p.method == "mc-plain"]
        t0, e0 = e_plain[0].seconds, e_plain[0].rmse
        grid = np.geomspace(t_all.min(), t_all.max(), 10)
        ax.plot(grid, e0 * (grid / t0) ** -0.5, color=INK_SECONDARY, linestyle=":", linewidth=1.0,
                label="slope $-1/2$")  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wall-clock time per estimate (s, log scale)")
        ax.set_ylabel("RMSE vs reference (log scale)")
        ax.set_title(problem)
        ax.legend(fontsize=8)
    return save_figure(fig, outdir / "error_vs_time.png")


def throughput_markdown(rows: list[ThroughputRow]) -> str:
    """Markdown table of :func:`throughput`."""
    lines = [
        "| Model | Engine | Backend | Steps | Paths | Time (s) | Paths / s | Path-steps / s |",
        "|---|---|---|--:|--:|--:|--:|--:|",
    ]
    lines += [
        f"| {r.model} | `{r.engine}` | {r.backend} | {r.n_steps} | {r.n_paths:,} | "
        f"{r.seconds:.3f} | {r.paths_per_second:,.0f} | {r.path_steps_per_second:,.0f} |"
        for r in rows
    ]
    return "\n".join(lines) + "\n"


def efficiency_markdown(rows: list[EfficiencyRow]) -> str:
    """Markdown table of :func:`efficiency`."""
    lines = [
        "| Problem | Method | Paths | Std error | Time (s) | Variance reduction "
        "| Efficiency gain (var x time) |",
        "|---|---|--:|--:|--:|--:|--:|",
    ]
    lines += [
        f"| {r.problem} | `{r.method}` | {r.n_paths:,} | {r.std_error:.2e} | {r.seconds:.3f} | "
        f"{r.variance_reduction:,.1f} | {r.efficiency_gain:,.1f} |"
        for r in rows
    ]
    return "\n".join(lines) + "\n"


def run_all(figures_dir: Path, results_dir: Path | None, quick: bool = False) -> dict[str, object]:
    """Run every benchmark, draw the figure and write ``benchmark.{md,json}``."""
    info = machine_info()
    tp = throughput(quick)
    ev = error_vs_time(quick)
    ef = efficiency(quick)
    plot_error_vs_time(ev, figures_dir)
    report: dict[str, object] = {
        "machine": info,
        "throughput": [asdict(r) for r in tp],
        "error_vs_time": [asdict(p) for p in ev],
        "efficiency": [asdict(r) for r in ef],
    }
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        md = (
            "Machine: "
            + ", ".join(f"{k} {v}" for k, v in info.items())
            + "\n\n### Throughput\n\n"
            + throughput_markdown(tp)
            + "\n### Efficiency at a fixed path budget\n\n"
            + efficiency_markdown(ef)
        )
        (results_dir / "benchmark.md").write_text(md, encoding="utf-8")
        (results_dir / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
