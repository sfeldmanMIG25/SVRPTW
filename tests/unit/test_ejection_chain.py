"""Smoke test for the ejection chain operator: never regresses cost, never
breaks TW feasibility."""
import numpy as np

from svrptw.config import Settings
from svrptw.io import Customer, Depot, Instance
from svrptw.solvers.common import ejection_chain, evaluate
from svrptw.solvers.common.solution import Route, Solution


def _toy_instance() -> Instance:
    rng = np.random.default_rng(0)
    n = 8
    pts = rng.uniform(0.0, 50.0, size=(n + 1, 2))
    T = np.sqrt(((pts[:, None] - pts[None, :]) ** 2).sum(-1))
    np.fill_diagonal(T, 0.0)
    D = T * 0.5
    return Instance(
        instance_id="TOY", city="Synth",
        num_customers=n, num_vehicles=4, vehicle_capacity=20,
        depot=Depot(node_id=0, x=pts[0, 0], y=pts[0, 1], ready=480, due=960),
        customers=[
            Customer(id=i + 1, node_id=i + 1, x=pts[i + 1, 0], y=pts[i + 1, 1],
                     demand=2, ready=500, due=900, service=5)
            for i in range(n)
        ],
        travel_time=T, travel_dist=D, asymmetry_score=0.0, seed=0,
    )


def test_ejection_chain_no_regression():
    inst = _toy_instance()
    settings = Settings()
    # Start from a deliberately scattered 4-route solution
    routes = [Route([1, 4]), Route([2, 5]), Route([3, 6]), Route([7, 8])]
    sol = Solution(
        instance_id=inst.instance_id, routes=routes, solver="seed",
        wall_clock_seconds=0.0, budget_seconds=0.0, feasible=False,
    )
    sol.metrics = evaluate(inst, sol, settings)
    sol.feasible = bool(sol.metrics["feasible"])
    cost_before = sol.metrics["operational_cost"]

    out = ejection_chain(inst, sol, settings, max_chain_length=3, max_seconds=1.0)
    cost_after = out.metrics["operational_cost"]
    assert cost_after <= cost_before + 1e-6, (
        f"ejection_chain regressed cost: {cost_before:.2f} -> {cost_after:.2f}"
    )
    assert out.metrics["missed_deliveries"] == 0, "all customers must remain served"
    assert out.metrics["tw_late_minutes"] == 0, "no TW violations allowed"
