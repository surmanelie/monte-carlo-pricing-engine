"""Regenerate every figure into figures/ and the measured numbers into results/figures.json.

Usage: ``python scripts/make_figures.py [--quick] [--only NAME ...]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mcengine.reporting.figures import FIGURES, make_all

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small experiments (smoke run)")
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args(argv)
    make_all(
        args.out,
        results_dir=None if args.quick else ROOT / "results",
        quick=args.quick,
        only=args.only,
        progress=lambda name: print(f"figure: {name}", file=sys.stderr),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
