import json

import pytest

from benchmarks.run_benchmark import main, prf


def test_prf():
    assert prf(8, 2, 2) == pytest.approx((0.8, 0.8, 0.8))
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_quick_benchmark_writes_reports(tmp_path):
    assert main(["--quick", "--out", str(tmp_path)]) == 0
    md = (tmp_path / "benchmark_results.md").read_text()
    assert "Session-level detection" in md and "Inference latency" in md and "Budget check" in md
    js = json.loads((tmp_path / "benchmark_results.json").read_text())
    c = js["agg"]["fused"]["sess"]
    assert c["tp"] + c["fn"] == 15 * 5
    assert c["tp"] / (c["tp"] + c["fn"]) > 0.7
