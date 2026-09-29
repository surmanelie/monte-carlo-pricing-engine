"""Consistent Matplotlib style for every figure of the project.

White background, 150 dpi, recessive grid, and a fixed categorical order (colour
follows the method, never its rank). The hues are a colour-vision-deficiency-checked
categorical palette; text stays in neutral ink.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
from cycler import cycler

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

#: Categorical palette in fixed order: blue, orange, aqua, yellow, magenta, green, violet, red.
PALETTE: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
DPI = 150

#: Stable colour per estimator / curve so that every figure uses the same encoding.
METHOD_COLORS: dict[str, str] = {
    "mc-plain": PALETTE[0],
    "mc-antithetic": PALETTE[1],
    "mc-cv": PALETTE[2],
    "qmc-sobol-bb": PALETTE[6],
    "mc-is": PALETTE[4],
}

_RC: dict[str, Any] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": INK_SECONDARY,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": cycler(color=list(PALETTE)),
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def apply_style() -> None:
    """Install the project style in Matplotlib's global ``rcParams``."""
    params: Any = mpl.rcParams
    params.update(_RC)


def new_figure(nrows: int = 1, ncols: int = 1, width: float = 7.0, height: float = 4.2) -> Any:
    """Create a styled figure; returns ``(fig, axes)`` like :func:`plt.subplots`."""
    apply_style()
    return plt.subplots(nrows, ncols, figsize=(width, height), constrained_layout=True)


def save_figure(fig: Figure, path: Path) -> Path:
    """Save ``fig`` as a 150-dpi PNG on white and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return path
