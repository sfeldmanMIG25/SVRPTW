"""Smoke tests for the bench reporting stack: leaderboard + pareto_plot."""
import json


def _fake_rows():
    return [
        {"solver": "greedy",       "instance_id": "X-N050-I000", "n": 50,
         "operational_cost": 800, "missed_deliveries": 0,
         "num_vehicles_used": 10, "wall_clock_seconds": 0.01},
        {"solver": "greedy",       "instance_id": "X-N050-I001", "n": 50,
         "operational_cost": 820, "missed_deliveries": 0,
         "num_vehicles_used": 10, "wall_clock_seconds": 0.01},
        {"solver": "ortools@1",    "instance_id": "X-N050-I000", "n": 50,
         "operational_cost": 750, "missed_deliveries": 0,
         "num_vehicles_used": 11, "wall_clock_seconds": 1.0},
        {"solver": "ortools@1",    "instance_id": "X-N050-I001", "n": 50,
         "operational_cost": 760, "missed_deliveries": 0,
         "num_vehicles_used": 11, "wall_clock_seconds": 1.0},
        {"solver": "auction_gart", "instance_id": "X-N050-I000", "n": 50,
         "operational_cost": 700, "missed_deliveries": 0,
         "num_vehicles_used": 9,  "wall_clock_seconds": 0.5},
        {"solver": "auction_gart", "instance_id": "X-N050-I001", "n": 50,
         "operational_cost": 710, "missed_deliveries": 0,
         "num_vehicles_used": 9,  "wall_clock_seconds": 0.5},
    ]


def test_leaderboard_pareto_annotation():
    from svrptw.bench.leaderboard import render
    md = render(_fake_rows())
    # auction_gart is strictly better on both axes than ortools@1, and
    # strictly better on cost than greedy (worse on time) — so the Pareto
    # frontier is {greedy, auction_gart}, NOT ortools@1.
    assert "`auction_gart`*" in md
    assert "`greedy`*" in md
    assert "`ortools@1`*" not in md          # dominated
    assert "Pareto" in md


def test_pareto_plot_writes_file(tmp_path):
    json_path = tmp_path / "rows.json"
    json_path.write_text(json.dumps({"rows": _fake_rows(), "config": {}}))
    out = tmp_path / "p.png"
    from svrptw.bench.pareto_plot import main
    rc = main([str(json_path), "--out", str(out)])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
