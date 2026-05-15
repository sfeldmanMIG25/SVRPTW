"""Soft-drop operator tests.

We construct a synthetic instance where one customer is intentionally far
from every other customer — its insertion cost should exceed
hard_late_penalty so soft_drop accepts the drop.  Also verifies that on
a "tight, all-near-depot" instance soft_drop never drops anyone.
"""
import numpy as np

from svrptw.config import Settings
from svrptw.io import Customer, Depot, Instance
from svrptw.solvers.common import evaluate, soft_drop
from svrptw.solvers.common.solution import Route, Solution


def _make_with_outlier() -> Instance:
    """N=4 customers: 3 clustered near depot, 1 absurdly far."""
    n = 4
    pts = np.array([
        [0.0, 0.0],   # depot
        [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],   # close cluster
        [1000.0, 1000.0],                       # outlier
    ])
    T = np.sqrt(((pts[:, None] - pts[None, :]) ** 2).sum(-1))
    np.fill_diagonal(T, 0.0)
    D = T
    return Instance(
        instance_id="DROP-OUTLIER", city="Synth",
        num_customers=n, num_vehicles=2, vehicle_capacity=20,
        depot=Depot(node_id=0, x=0.0, y=0.0, ready=480, due=10_000),
        customers=[
            Customer(id=i + 1, node_id=i + 1, x=pts[i + 1, 0], y=pts[i + 1, 1],
                     demand=1, ready=480, due=10_000, service=5)
            for i in range(n)
        ],
        travel_time=T, travel_dist=D, asymmetry_score=0.0, seed=0,
    )


def test_outlier_is_dropped():
    inst = _make_with_outlier()
    settings = Settings()
    # Force the outlier into the route
    sol = Solution(
        instance_id=inst.instance_id,
        routes=[Route([1, 2, 3, 4])],   # 4 is the outlier
        solver="seed",
        wall_clock_seconds=0.0, budget_seconds=0.0, feasible=False,
    )
    sol.metrics = evaluate(inst, sol, settings)
    sol.feasible = bool(sol.metrics["feasible"])
    cost_before = sol.metrics["operational_cost"]
    out = soft_drop(inst, sol, settings, max_seconds=2.0, max_drops_pct=0.5)
    # The outlier should be gone — drop saves ~2000 minutes wage which is way more
    # than the 1000 miss penalty.
    assert 4 not in out.routes[0].customers, \
        f"outlier should be dropped; route is {out.routes[0].customers}"
    assert out.metrics["operational_cost"] < cost_before, \
        f"cost should drop: {cost_before:.1f} -> {out.metrics['operational_cost']:.1f}"


def test_no_drop_on_tight_instance():
    """If every customer is cheap to serve, soft_drop must drop nothing."""
    rng = np.random.default_rng(0)
    n = 6
    pts = rng.uniform(0, 10, size=(n + 1, 2))
    T = np.sqrt(((pts[:, None] - pts[None, :]) ** 2).sum(-1))
    np.fill_diagonal(T, 0.0)
    inst = Instance(
        instance_id="TIGHT", city="Synth",
        num_customers=n, num_vehicles=2, vehicle_capacity=20,
        depot=Depot(node_id=0, x=pts[0, 0], y=pts[0, 1], ready=480, due=960),
        customers=[
            Customer(id=i + 1, node_id=i + 1, x=pts[i + 1, 0], y=pts[i + 1, 1],
                     demand=2, ready=480, due=900, service=2)
            for i in range(n)
        ],
        travel_time=T, travel_dist=T * 0.5, asymmetry_score=0.0, seed=0,
    )
    settings = Settings()
    sol = Solution(
        instance_id=inst.instance_id,
        routes=[Route([1, 2, 3]), Route([4, 5, 6])],
        solver="seed",
        wall_clock_seconds=0.0, budget_seconds=0.0, feasible=False,
    )
    sol.metrics = evaluate(inst, sol, settings)
    sol.feasible = bool(sol.metrics["feasible"])
    out = soft_drop(inst, sol, settings, max_seconds=1.0)
    served_before = sum(len(r.customers) for r in sol.routes)
    served_after  = sum(len(r.customers) for r in out.routes)
    assert served_after == served_before, "no customers should be dropped on a cheap instance"
