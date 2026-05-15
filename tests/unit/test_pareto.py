"""Smoke tests for the Pareto MOO stack."""
from svrptw.bench.pareto import (
    Objective,
    hypervolume,
    lex_then_hv_rank,
    nondominated,
    render_report,
)


def _rows():
    return [
        # solver, cost (min), wall (min), missed (min) - greedy is cheap+fast, auction beats cost
        {"solver": "greedy",       "n": 50, "operational_cost": 800.0,
         "wall_clock_seconds": 0.01, "missed_deliveries": 0,
         "instance_id": "X-I0", "tw_late_minutes": 0},
        {"solver": "auction_gart", "n": 50, "operational_cost": 715.0,
         "wall_clock_seconds": 0.50, "missed_deliveries": 0,
         "instance_id": "X-I0", "tw_late_minutes": 0},
        {"solver": "lkh3@1",       "n": 50, "operational_cost": 820.0,
         "wall_clock_seconds": 1.30, "missed_deliveries": 0,
         "instance_id": "X-I0", "tw_late_minutes": 0},
        # ortools dominated by all: higher cost and higher time
        {"solver": "ortools@10",   "n": 50, "operational_cost": 950.0,
         "wall_clock_seconds": 10.0, "missed_deliveries": 0,
         "instance_id": "X-I0", "tw_late_minutes": 0},
    ]


def test_nondominated_picks_pareto():
    rows = _rows()
    objs = [Objective("operational_cost"), Objective("wall_clock_seconds")]
    front = nondominated(rows, objs)
    solvers_on_front = {rows[i]["solver"] for i in front}
    assert "greedy" in solvers_on_front
    assert "auction_gart" in solvers_on_front
    assert "lkh3@1" not in solvers_on_front       # dominated by greedy (cheaper & faster)
    assert "ortools@10" not in solvers_on_front   # dominated by both


def test_hypervolume_positive():
    rows = _rows()
    objs = [Objective("operational_cost"), Objective("wall_clock_seconds")]
    hv = hypervolume(rows, objs)
    assert hv > 0


def test_lex_then_hv_rank_zero_constraints():
    rows = _rows()
    # All clean (missed=0, tw_late=0).  Top of ranking should be a Pareto-front member.
    objs = [Objective("operational_cost"), Objective("wall_clock_seconds")]
    ranked = lex_then_hv_rank(rows, hard_zero=["missed_deliveries", "tw_late_minutes"], objs=objs)
    top = rows[ranked[0]]["solver"]
    assert top in {"greedy", "auction_gart"}


def test_render_report_smoke():
    md = render_report(_rows())
    assert "Pareto report" in md
    assert "auction_gart" in md
    assert "Hypervolume" in md
