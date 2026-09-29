from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcengine.reporting.figures import FIGURES, make_all


def test_quick_figures(tmp_path: Path) -> None:
    names: list[str] = []
    numbers = make_all(
        tmp_path / "fig", results_dir=tmp_path / "res", quick=True, progress=names.append
    )
    assert names == list(FIGURES)
    for name in FIGURES:
        assert (tmp_path / "fig" / f"{name}.png").stat().st_size > 10_000
    stored = json.loads((tmp_path / "res" / "figures.json").read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(numbers))
    # re-running a subset keeps the numbers of the other figures
    make_all(tmp_path / "fig", results_dir=tmp_path / "res", quick=True, only=["convergence"])
    stored = json.loads((tmp_path / "res" / "figures.json").read_text(encoding="utf-8"))
    assert set(stored) == set(FIGURES)


def test_unknown_figure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown"):
        make_all(tmp_path, only=["nope"])
