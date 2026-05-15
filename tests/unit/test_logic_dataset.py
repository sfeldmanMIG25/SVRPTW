"""SPEC-6-LOGIC-01 — dataset.build_pairs scaffolding tests.

No network — all committee/labeling paths are mocked.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from svrptw.logic.dataset import _label_from_scores, build_pairs


def test_label_from_scores_tie():
    # Within eps → 0.5
    assert _label_from_scores(0.50, 0.52, eps=0.05) == 0.5


def test_label_from_scores_a_wins():
    assert _label_from_scores(0.80, 0.30) == 1.0


def test_label_from_scores_b_wins():
    assert _label_from_scores(0.20, 0.70) == 0.0


def test_label_from_scores_none_returns_nan():
    import math
    assert math.isnan(_label_from_scores(None, 0.5))
    assert math.isnan(_label_from_scores(0.5, None))


def test_build_pairs_smoke(tmp_path: Path):
    """End-to-end smoke with mocked solvers + mocked committee.

    Validates the pair-selection + flushing logic without touching the
    network or running real solvers.
    """
    # Fake bench rows for 2 instances × 2 solvers each.
    bench = {
        "rows": [
            {"instance_id": "OSM-Manhattan-N050-I000", "solver": "portfolio@10",
             "operational_cost": 700.0, "wall_clock_seconds": 5.0,
             "n": 50, "capacity_overload": 0.0},
            {"instance_id": "OSM-Manhattan-N050-I000", "solver": "pyvrp@30",
             "operational_cost": 800.0, "wall_clock_seconds": 30.0,
             "n": 50, "capacity_overload": 0.0},
            {"instance_id": "OSM-Manhattan-N050-I001", "solver": "portfolio@10",
             "operational_cost": 750.0, "wall_clock_seconds": 5.0,
             "n": 50, "capacity_overload": 0.0},
            {"instance_id": "OSM-Manhattan-N050-I001", "solver": "pyvrp@30",
             "operational_cost": 770.0, "wall_clock_seconds": 30.0,
             "n": 50, "capacity_overload": 0.0},
        ]
    }
    bench_path = tmp_path / "bench.json"
    bench_path.write_text(json.dumps(bench), encoding="utf-8")
    out_path = tmp_path / "pairs.json"

    # build_pairs requires {iid}.json to exist before calling load_instance.
    for iid in ("OSM-Manhattan-N050-I000", "OSM-Manhattan-N050-I001"):
        (tmp_path / f"{iid}.json").write_text("{}", encoding="utf-8")

    # Mock everything that hits I/O or network.
    def fake_load_instance(p):
        class _FakeInst:
            instance_id = Path(p).stem
        return _FakeInst()

    def fake_reconstruct(inst, row):
        from svrptw.solvers.common import Route, Solution
        # Return a tiny synthetic solution; only used for hashing + rendering.
        return Solution(
            instance_id=inst.instance_id, routes=[Route(customers=[1, 2, 3])],
            solver=row["solver"], wall_clock_seconds=row["wall_clock_seconds"],
            budget_seconds=10.0, feasible=True, metrics={"operational_cost": row["operational_cost"]},
        )

    def fake_render(inst, sol, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"FAKEPNG")

    # Committee returns higher score for the lower-cost row (portfolio)
    # via the pair-judging path (SPEC-6-LOGIC-02).
    def fake_committee_label_pair(self, inst, sol_a, sol_b, prompt, image_a, image_b):
        from svrptw.logic.committee import PairLabel
        sa = 0.7 if sol_a.solver.startswith("portfolio") else 0.4
        sb = 0.7 if sol_b.solver.startswith("portfolio") else 0.4
        return PairLabel(
            score_a=sa, score_b=sb, score_a_std=0.05, score_b_std=0.05,
            rationale="mock pair", authoritative=True, n_responders=3,
        )

    # The combined-judge path calls to_text(inst, sol) which needs a real
    # Instance. Patch _judge_pair_combined directly to skip it in this smoke.
    def fake_judge_pair(committee, inst, sol_a, sol_b, cache_dir, rng=None):
        sa = 0.7 if sol_a.solver.startswith("portfolio") else 0.4
        sb = 0.7 if sol_b.solver.startswith("portfolio") else 0.4
        return (sa, sb, True, "mock rationale")

    with patch("svrptw.logic.dataset.load_instance", side_effect=fake_load_instance), \
         patch("svrptw.logic.dataset._reconstruct_solution", side_effect=fake_reconstruct), \
         patch("svrptw.logic.dataset.render_solution", side_effect=fake_render), \
         patch("svrptw.logic.dataset._judge_pair_combined", side_effect=fake_judge_pair):
        pairs = build_pairs(
            bench_path=bench_path,
            out_path=out_path,
            instances_dir=tmp_path,        # any path; load_instance is mocked
            n_pairs=10,
            seed=0,
        )

    assert out_path.exists()
    assert len(pairs) == 2  # 2 instances × 1 pair each
    # In every pair where solver_a starts with "portfolio", label should be 1.0.
    for p in pairs:
        if p.solver_a.startswith("portfolio"):
            assert p.label == 1.0
        else:
            assert p.label == 0.0
