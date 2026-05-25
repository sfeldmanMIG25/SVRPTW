from __future__ import annotations
import time
import numpy as np
from svrptw.config import Settings
from svrptw.council.proposal import OperatorContext
from svrptw.io import Instance
from svrptw.solvers.common import Route, Solution, evaluate
from svrptw.solvers.common.local_search import _cust_by_id, _route_arrival_and_close
def operator(solution: Solution, context: OperatorContext) -> Solution | None:
    inst = context.instance
    cust_by_id = _cust_by_id(inst)
    def get_centroid(r):
        coords = [inst.customers[c-1] for c in r.customers]
        return np.mean([(c.x, c.y) for c in coords], axis=0) if coords else (0, 0)
    route_centroids = [get_centroid(r) for r in solution.routes]
    best_sol, best_cost = None, solution.metrics['operational_cost']
    for ri, r in enumerate(solution.routes):
        if not r.customers or time.perf_counter() >= (time.perf_counter() + context.deadline_seconds): continue
        for c_idx, cid in enumerate(r.customers):
            c = inst.customers[cid-1]
            curr_ang = np.arctan2(c.y - route_centroids[ri][1], c.x - route_centroids[ri][0])
            for rj, r_target in enumerate(solution.routes):
                if ri == rj: continue
                target_ang = np.arctan2(c.y - route_centroids[rj][1], c.x - route_centroids[rj][0])
                if abs(curr_ang - target_ang) > 2.5:
                    new_routes = [Route(customers=list(r.customers)) for r in solution.routes]
                    cust = new_routes[ri].customers.pop(c_idx)
                    new_routes[rj].customers.append(cust)
                    ok, _ = _route_arrival_and_close(inst, new_routes[rj].customers)
                    if ok and sum(cust_by_id[c].demand for c in new_routes[rj].customers) <= inst.vehicle_capacity:
                        new_sol = Solution(inst.instance_id, new_routes, solution.solver, solution.wall_clock_seconds, solution.budget_seconds, False)
                        new_sol.metrics = evaluate(inst, new_sol, context.settings)
                        if new_sol.metrics['feasible'] and new_sol.metrics['operational_cost'] < best_cost:
                            best_cost, best_sol = new_sol.metrics['operational_cost'], new_sol
    return best_sol