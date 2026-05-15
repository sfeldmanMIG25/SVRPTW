"""Greedy nearest-neighbor for asymmetric VRPTW.  Deterministic baseline."""
from __future__ import annotations

import time

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common import Route, Solution, evaluate


def solve(inst: Instance, settings: Settings) -> Solution:
    t0 = time.perf_counter()
    T = inst.travel_time
    cust_by_id = {c.id: c for c in inst.customers}
    unserved = set(c.id for c in inst.customers)
    demands = {c.id: c.demand for c in inst.customers}
    routes: list[Route] = []

    for _ in range(inst.num_vehicles):
        if not unserved:
            break
        r = Route()
        cur = 0
        clock = float(inst.depot.ready)
        load = 0
        while unserved:
            candidates = []
            for cid in unserved:
                if load + demands[cid] > inst.vehicle_capacity:
                    continue
                cust = cust_by_id[cid]
                arrive = clock + float(T[cur, cid])
                start = max(arrive, float(cust.ready))
                if start > cust.due:
                    continue
                if start + cust.service + float(T[cid, 0]) > inst.depot.due:
                    continue
                # rank by start - wage: prefer minimal travel + minimal late slack
                cost = float(T[cur, cid]) + max(0.0, cust.ready - arrive) * 0.5
                candidates.append((cost, cid))
            if not candidates:
                break
            candidates.sort()
            cid = candidates[0][1]
            cust = cust_by_id[cid]
            arrive = clock + float(T[cur, cid])
            clock = max(arrive, float(cust.ready)) + cust.service
            load += demands[cid]
            r.customers.append(cid)
            unserved.discard(cid)
            cur = cid
        routes.append(r)

    sol = Solution(
        instance_id=inst.instance_id,
        routes=routes,
        solver="greedy",
        wall_clock_seconds=time.perf_counter() - t0,
        budget_seconds=0.0,
        feasible=False,
    )
    sol.metrics = evaluate(inst, sol, settings)
    sol.feasible = bool(sol.metrics["feasible"])
    return sol


if __name__ == "__main__":
    import argparse
    import json

    from svrptw.io import load_instance
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve(inst, Settings())
    print(json.dumps({"solver": sol.solver, "metrics": sol.metrics,
                      "wall_clock_s": sol.wall_clock_seconds}, indent=2))
