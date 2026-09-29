from __future__ import annotations

import json
from pathlib import Path

from mcengine.reporting.benchmark import (
    BENCH_METHODS,
    efficiency_markdown,
    machine_info,
    run_all,
    throughput_markdown,
)


def test_quick_benchmark_writes_reports(tmp_path: Path) -> None:
    report = run_all(tmp_path / "fig", tmp_path / "res", quick=True)
    assert (tmp_path / "fig" / "error_vs_time.png").stat().st_size > 10_000
    stored = json.loads((tmp_path / "res" / "benchmark.json").read_text(encoding="utf-8"))
    assert stored["machine"]["mcengine"]
    assert len(stored["efficiency"]) == 2 * len(BENCH_METHODS)
    methods = {p["method"] for p in stored["error_vs_time"]}
    assert methods == {"mc-plain", "mc-antithetic", "mc-cv", "qmc-sobol-bb", "mc-is"}
    md = (tmp_path / "res" / "benchmark.md").read_text(encoding="utf-8")
    assert "Paths / s" in md
    assert "Efficiency gain" in md
    assert report["throughput"]


def test_markdown_helpers_handle_empty_input() -> None:
    assert throughput_markdown([]).count("\n") == 2
    assert efficiency_markdown([]).count("\n") == 2
    assert "python" in machine_info()
