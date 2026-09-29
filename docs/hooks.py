"""MkDocs hook: copy the generated figures into the documentation before each build.

The figures are produced by ``mcengine figures`` / ``mcengine benchmark`` into
``figures/``; the docs embed them from ``docs/figures/`` (ignored by git), and the tables
in ``results/`` are included with ``pymdownx.snippets``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def on_pre_build(config: Any, **kwargs: Any) -> None:
    target = ROOT / "docs" / "figures"
    target.mkdir(exist_ok=True)
    for png in (ROOT / "figures").glob("*.png"):
        shutil.copy2(png, target / png.name)
