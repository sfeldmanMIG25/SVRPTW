"""Shared local-search moves for asymmetric VRPTW.

All moves preserve TW feasibility (a move is rejected outright if it would
violate any visited customer's time window or push the route past depot.due).
"""
from __future__ import annotations

import time

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common.solution import Route, Solution, evaluate

# Per-instance cache for the customer lookup dict — the hot path
# `_route_arrival_and_close` is called ~10^6 times in a single solve and
# the dict comprehension was 65% of its runtime in profiling.
#
# Keyed on instance_id (string) so a GC'd Instance address-reuse can't
# return stale customers (judge-review finding).
_CUST_DICT_CACHE: dict[str, dict[int, object]] = {}


def _cust_by_id(inst: Instance) -> dict:
    key = inst.instance_id
    d = _CUST_DICT_CACHE.get(key)
    if d is None:
        d = {c.id: c for c in inst.customers}
        _CUST_DICT_CACHE[key] = d
    return d


def _route_arrival_and_close(inst: Instance, customers: list[int]) -> tuple[bool, float]:
    """Return (feasible, end_clock_at_depot).  Feasible = all TWs respected
    AND return to depot by depot.due."""
    T = inst.travel_time
    cust_by_id = _cust_by_id(inst)
    capacity = inst.vehicle_capacity
    depot_due = inst.depot.due
    clock = float(inst.depot.ready)
    cur = 0
    load = 0
    for cid in customers:
        c = cust_by_id[cid]
        load += c.demand
        if load > capacity:
            return False, 0.0
        arrive = clock + float(T[cur, cid])
        start  = max(arrive, float(c.ready))
        if start > c.due:
            return False, 0.0
        clock = start + c.service
        cur = cid
    end = clock + float(T[cur, 0])
    if end > depot_due:
        return False, 0.0
    return True, end


def two_opt_intra(inst: Instance, sol: Solution, settings: Settings,
                  max_seconds: float = 2.0) -> Solution:
    """Intra-route 2-opt: reverse a sub-segment of a single route.  Reject any
    move that breaks TW feasibility.  This is complementary to relocate
    (which moves customers between routes)."""
    deadline = time.perf_counter() + max_seconds
    best = sol
    best_cost = sol.metrics["operational_cost"]
    improved = True

    while improved and time.perf_counter() < deadline:
        improved = False
        for ri in range(len(best.routes)):
            route = best.routes[ri].customers
            if len(route) < 3:
                continue
            if time.perf_counter() >= deadline:
                break
            done = False
            for i in range(len(route) - 1):
                if done or time.perf_counter() >= deadline:
                    break
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    ok, _ = _route_arrival_and_close(inst, new_route)
                    if not ok:
                        continue
                    candidate = list(best.routes)
                    candidate[ri] = Route(customers=new_route)
                    cand = Solution(
                        instance_id=inst.instance_id, routes=candidate,
                        solver=best.solver,
                        wall_clock_seconds=best.wall_clock_seconds,
                        budget_seconds=best.budget_seconds, feasible=False,
                    )
                    cand.metrics = evaluate(inst, cand, settings)
                    cand.feasible = bool(cand.metrics["feasible"])
                    if cand.metrics["operational_cost"] < best_cost - 1e-6:
                        best = cand
                        best_cost = cand.metrics["operational_cost"]
                        improved = True
                        done = True
                        break
    return best


def three_opt_intra(inst: Instance, sol: Solution, settings: Settings,
                    max_seconds: float = 2.0) -> Solution:
    """Intra-route 3-opt: pick 3 edges and reconnect in any of the 7
    non-trivial ways (4 reverse-pattern + 3 segment-reorderings).  Reject
    any move that breaks TW feasibility.

    Complements two_opt_intra (which only handles segment-reversal).
    Each pass is O(N³); deadline-bounded.
    """
    deadline = time.perf_counter() + max_seconds
    best = sol
    best_cost = sol.metrics["operational_cost"]
    improved = True

    while improved and time.perf_counter() < deadline:
        improved = False
        for ri in range(len(best.routes)):
            if improved or time.perf_counter() >= deadline:
                break
            route = best.routes[ri].customers
            n = len(route)
            if n < 4:
                continue
            for i in range(n - 2):
                if improved or time.perf_counter() >= deadline:
                    break
                for j in range(i + 1, n - 1):
                    if improved:
                        break
                    for k in range(j + 1, n):
                        # Segments: A=route[:i], B=route[i:j+1], C=route[j+1:k+1], D=route[k+1:]
                        A = route[:i]
                        B = route[i:j + 1]
                        C = route[j + 1:k + 1]
                        D = route[k + 1:]
                        candidates = [
                            A + B[::-1] + C + D,        # reverse B
                            A + B + C[::-1] + D,        # reverse C
                            A + B[::-1] + C[::-1] + D,  # reverse B and C
                            A + C + B + D,              # swap B and C
                            A + C[::-1] + B + D,        # swap, reverse C
                            A + C + B[::-1] + D,        # swap, reverse B
                            A + C[::-1] + B[::-1] + D,  # swap both, reverse both
                        ]
                        for new_route in candidates:
                            if new_route == route:
                                continue
                            ok, _ = _route_arrival_and_close(inst, new_route)
                            if not ok:
                                continue
                            cand_routes = list(best.routes)
                            cand_routes[ri] = Route(customers=new_route)
                            cand = Solution(
                                instance_id=inst.instance_id, routes=cand_routes,
                                solver=best.solver,
                                wall_clock_seconds=best.wall_clock_seconds,
                                budget_seconds=best.budget_seconds, feasible=False,
                            )
                            cand.metrics = evaluate(inst, cand, settings)
                            cand.feasible = bool(cand.metrics["feasible"])
                            if cand.metrics["operational_cost"] < best_cost - 1e-6:
                                best = cand
                                best_cost = cand.metrics["operational_cost"]
                                improved = True
                                break
    return best


def cyclic_3_exchange(inst: Instance, sol: Solution, settings: Settings,
                      max_seconds: float = 2.0) -> Solution:
    """Cyclic 3-route customer exchange: pick three non-empty routes
    (R_a, R_b, R_c) and move a customer from R_a → R_b, R_b → R_c, R_c → R_a.
    Captures moves that no 2-route swap can find (Thompson & Orlin 1989).
    """
    deadline = time.perf_counter() + max_seconds
    best = sol
    best_cost = sol.metrics["operational_cost"]
    improved = True

    while improved and time.perf_counter() < deadline:
        improved = False
        non_empty = [i for i, r in enumerate(best.routes) if r.customers]
        if len(non_empty) < 3:
            break
        for ia_i in range(len(non_empty)):
            if improved or time.perf_counter() >= deadline:
                break
            for ib_i in range(len(non_empty)):
                if ia_i == ib_i:
                    continue
                for ic_i in range(len(non_empty)):
                    if ic_i in (ia_i, ib_i):
                        continue
                    if improved or time.perf_counter() >= deadline:
                        break
                    ra_idx, rb_idx, rc_idx = non_empty[ia_i], non_empty[ib_i], non_empty[ic_i]
                    ra, rb, rc = (
                        best.routes[ra_idx].customers,
                        best.routes[rb_idx].customers,
                        best.routes[rc_idx].customers,
                    )
                    # Try moving one customer from each.
                    for pa in range(len(ra)):
                        if improved:
                            break
                        for pb in range(len(rb)):
                            if improved:
                                break
                            for pc in range(len(rc)):
                                ca, cb, cc = ra[pa], rb[pb], rc[pc]
                                # Best-insertion of ca into rb (sans cb), cb into rc (sans cc),
                                # cc into ra (sans ca).
                                ra2 = ra[:pa] + ra[pa + 1:]
                                rb2 = rb[:pb] + rb[pb + 1:]
                                rc2 = rc[:pc] + rc[pc + 1:]

                                def _best_insert(seq, cid):
                                    best_d = float("inf")
                                    best_q = None
                                    for pos in range(len(seq) + 1):
                                        cand_route = seq[:pos] + [cid] + seq[pos:]
                                        ok, _ = _route_arrival_and_close(inst, cand_route)
                                        if not ok:
                                            continue
                                        before = _route_time(inst, seq)
                                        after = _route_time(inst, cand_route)
                                        d = after - before
                                        if d < best_d:
                                            best_d = d
                                            best_q = cand_route
                                    return best_q

                                rb3 = _best_insert(rb2, ca)
                                if rb3 is None:
                                    continue
                                rc3 = _best_insert(rc2, cb)
                                if rc3 is None:
                                    continue
                                ra3 = _best_insert(ra2, cc)
                                if ra3 is None:
                                    continue
                                cand_routes = [Route(list(r.customers)) for r in best.routes]
                                cand_routes[ra_idx] = Route(ra3)
                                cand_routes[rb_idx] = Route(rb3)
                                cand_routes[rc_idx] = Route(rc3)
                                cand = Solution(
                                    instance_id=inst.instance_id, routes=cand_routes,
                                    solver=best.solver,
                                    wall_clock_seconds=best.wall_clock_seconds,
                                    budget_seconds=best.budget_seconds, feasible=False,
                                )
                                cand.metrics = evaluate(inst, cand, settings)
                                cand.feasible = bool(cand.metrics["feasible"])
                                if cand.metrics["operational_cost"] < best_cost - 1e-6:
                                    best = cand
                                    best_cost = cand.metrics["operational_cost"]
                                    improved = True
                                    break
    return best


def soft_drop(inst: Instance, sol: Solution, settings: Settings,
              max_seconds: float = 2.0, max_drops_pct: float = 0.10) -> Solution:
    """Drop a customer if doing so lowers TOTAL operational cost (i.e. the
    routing savings from the drop exceed `hard_late_penalty`).

    Note: per-customer LOCAL marginal cost is typically <$15 while
    hard_late_penalty is $1000; the trick is that dropping a customer may
    cascade to vehicle removal or much shorter route — that only shows
    in the full evaluate(), not in the local delta.  So we rank customers
    by local marginal descending (cheap screen) and full-evaluate the top
    candidates.

    SPEC-3-SOFTDROP-01.  Replicates the LKH-3 trick that beats us on
    N=100 instances.
    """
    deadline = time.perf_counter() + max_seconds
    T = inst.travel_time
    D = inst.travel_dist
    wage = settings.economics.wage_per_minute
    cpm = settings.economics.cost_per_mile

    best = sol
    best_cost = sol.metrics["operational_cost"]
    n_served = sum(len(r.customers) for r in best.routes)
    max_drops = max(1, int(n_served * max_drops_pct))
    drops_so_far = 0
    improved = True

    while improved and time.perf_counter() < deadline and drops_so_far < max_drops:
        improved = False

        # Score every routed customer by its local marginal (descending).
        scored: list[tuple[float, int, int]] = []  # (marginal, route_idx, pos)
        for r_idx, route in enumerate(best.routes):
            seq = route.customers
            for pos, cid in enumerate(seq):
                prev_node = 0 if pos == 0 else seq[pos - 1]
                next_node = 0 if pos + 1 >= len(seq) else seq[pos + 1]
                dt = float(T[prev_node, cid]) + float(T[cid, next_node]) - float(T[prev_node, next_node])
                dd = float(D[prev_node, cid]) + float(D[cid, next_node]) - float(D[prev_node, next_node])
                svc = inst.customers[cid - 1].service
                marginal = wage * (dt + svc) + cpm * dd
                scored.append((marginal, r_idx, pos))
        scored.sort(reverse=True)

        # Try dropping the top candidates one at a time with full eval.
        # `max_trials` caps work per pass.
        max_trials = min(len(scored), max(8, n_served // 4))
        for _marginal, r_idx, pos in scored[:max_trials]:
            if time.perf_counter() >= deadline:
                break
            trial_routes = [Route(list(r.customers)) for r in best.routes]
            seq = trial_routes[r_idx].customers
            # Pos may now be stale after a previous accepted drop; refind cid.
            if pos >= len(seq):
                continue
            seq.pop(pos)
            cand = Solution(
                instance_id=inst.instance_id, routes=trial_routes,
                solver=best.solver,
                wall_clock_seconds=best.wall_clock_seconds,
                budget_seconds=best.budget_seconds, feasible=False,
            )
            cand.metrics = evaluate(inst, cand, settings)
            cand.feasible = bool(cand.metrics["feasible"])
            if cand.metrics["operational_cost"] < best_cost - 1e-6:
                best = cand
                best_cost = cand.metrics["operational_cost"]
                improved = True
                drops_so_far += 1
                break  # restart with fresh score
    return best


def sisr_destroy_repair(inst: Instance, sol: Solution, settings: Settings,
                        max_seconds: float = 3.0,
                        avg_strings: int = 10,
                        max_string_len: int = 10,
                        blink_p: float = 0.01,
                        seed: int = 0) -> Solution:
    """SISR (Slack Induction by String Removals).  SPEC-3-SISR-01.

    Ruin: remove `avg_strings` strings of length 1..max_string_len from
    routes adjacent to a random seed.  Repair: blink-greedy reinsertion.
    Accept only if total operational cost strictly drops.
    """
    import random
    rng = random.Random(seed)
    deadline = time.perf_counter() + max_seconds
    T = inst.travel_time
    best = sol
    best_cost = sol.metrics["operational_cost"]

    while time.perf_counter() < deadline:
        routes = [list(r.customers) for r in best.routes]
        all_routed = [(ri, pi, cid) for ri, r in enumerate(routes) for pi, cid in enumerate(r)]
        if len(all_routed) < 2:
            break
        # 1) Seed
        seed_ri, seed_pi, seed_cid = rng.choice(all_routed)
        seed_x = inst.customers[seed_cid - 1].x
        seed_y = inst.customers[seed_cid - 1].y

        # 2) Pick adjacent routes by closest-mean-customer to seed
        def route_dist_to_seed(r: list[int], sx: float = seed_x, sy: float = seed_y) -> float:
            if not r:
                return float("inf")
            return min(
                (inst.customers[c - 1].x - sx) ** 2 +
                (inst.customers[c - 1].y - sy) ** 2
                for c in r
            )
        route_order = sorted(range(len(routes)), key=lambda i: route_dist_to_seed(routes[i]))
        n_strings = min(avg_strings, len([i for i in route_order if routes[i]]))
        target_route_ids = route_order[:n_strings]

        # 3) Remove a string from each target route
        removed: list[int] = []
        for ri in target_route_ids:
            r = routes[ri]
            if not r:
                continue
            length = rng.randint(1, min(max_string_len, len(r)))
            start = rng.randint(0, len(r) - length)
            removed.extend(r[start:start + length])
            routes[ri] = r[:start] + r[start + length:]
        if not removed:
            continue

        # 4) Blink-greedy reinsertion (random order).  Granular neighbourhood
        # prunes which routes to even consider — a customer is inserted only
        # into routes containing at least one of its k-nearest neighbours.
        from svrptw.solvers.common.neighbors import get_neighbors
        gn = get_neighbors(inst, k=20)
        rng.shuffle(removed)
        reinsert_failed: list[int] = []
        for cid in removed:
            neigh = set(gn.neighbors_of(cid).tolist())
            best_cost_pos = float("inf")
            best_pos = None
            for ri, r in enumerate(routes):
                # Skip routes that share no neighbour with cid (and are non-empty).
                if r and not any(c in neigh for c in r):
                    continue
                for pos in range(len(r) + 1):
                    if blink_p > 0 and rng.random() < blink_p:
                        continue
                    cand = r[:pos] + [cid] + r[pos:]
                    ok, _ = _route_arrival_and_close(inst, cand)
                    if not ok:
                        continue
                    if pos == 0:
                        prev_node = 0
                    else:
                        prev_node = r[pos - 1]
                    next_node = r[pos] if pos < len(r) else 0
                    delta = (float(T[prev_node, cid]) + float(T[cid, next_node])
                             - float(T[prev_node, next_node]))
                    if delta < best_cost_pos:
                        best_cost_pos = delta
                        best_pos = (ri, pos)
            # If granular filter excluded all routes, fall back to a full scan
            # for this customer.  This guarantees no customer is locked out.
            if best_pos is None:
                for ri, r in enumerate(routes):
                    for pos in range(len(r) + 1):
                        cand = r[:pos] + [cid] + r[pos:]
                        ok, _ = _route_arrival_and_close(inst, cand)
                        if not ok:
                            continue
                        prev_node = 0 if pos == 0 else r[pos - 1]
                        next_node = r[pos] if pos < len(r) else 0
                        delta = (float(T[prev_node, cid]) + float(T[cid, next_node])
                                 - float(T[prev_node, next_node]))
                        if delta < best_cost_pos:
                            best_cost_pos = delta
                            best_pos = (ri, pos)
            if best_pos is None:
                reinsert_failed.append(cid)
                continue
            ri, pos = best_pos
            routes[ri] = routes[ri][:pos] + [cid] + routes[ri][pos:]
        if reinsert_failed:
            continue  # don't accept partial repairs

        cand = Solution(
            instance_id=inst.instance_id,
            routes=[Route(customers=r) for r in routes],
            solver=best.solver,
            wall_clock_seconds=best.wall_clock_seconds,
            budget_seconds=best.budget_seconds, feasible=False,
        )
        cand.metrics = evaluate(inst, cand, settings)
        cand.feasible = bool(cand.metrics["feasible"])
        if cand.feasible and cand.metrics["operational_cost"] < best_cost - 1e-6:
            best = cand
            best_cost = cand.metrics["operational_cost"]
    return best


def ejection_chain(inst: Instance, sol: Solution, settings: Settings,
                   max_chain_length: int = 3,
                   max_seconds: float = 5.0) -> Solution:
    """Glover-style k-cyclic ejection chain.  SPEC-3-EJECT-01.

    Tries chains of length 2..max_chain_length, accepting the first chain
    that strictly drops operational cost while preserving TW feasibility.
    """
    import random
    deadline = time.perf_counter() + max_seconds
    best = sol
    best_cost = sol.metrics["operational_cost"]
    rng = random.Random(0)

    while time.perf_counter() < deadline:
        routes = [list(r.customers) for r in best.routes]
        non_empty = [i for i, r in enumerate(routes) if r]
        if len(non_empty) < 2:
            break
        # Pick a source route + customer.  Heuristic: random non-empty route.
        src_idx = rng.choice(non_empty)
        if not routes[src_idx]:
            continue
        pos = rng.randrange(len(routes[src_idx]))
        c1 = routes[src_idx][pos]

        # Build chain.
        chain_routes = [list(r) for r in routes]
        chain_routes[src_idx].pop(pos)
        ejected = [(src_idx, c1)]   # (route_of_origin, customer)
        chain_ok = True

        for depth in range(max_chain_length):
            cur_route_idx, cur_cust = ejected[-1]
            # Find best insertion of cur_cust into a route OTHER than its
            # origin and not yet visited this chain.
            visited_routes = {r for r, _ in ejected}
            best_target = None
            best_delta = float("inf")
            for ri in range(len(chain_routes)):
                if ri in visited_routes:
                    continue
                base = chain_routes[ri]
                for ins in range(len(base) + 1):
                    new_route = base[:ins] + [cur_cust] + base[ins:]
                    ok, _ = _route_arrival_and_close(inst, new_route)
                    if not ok:
                        continue
                    delta = _route_time(inst, new_route) - _route_time(inst, base)
                    if delta < best_delta:
                        best_delta = delta
                        best_target = (ri, ins, new_route)
            if best_target is None:
                chain_ok = False
                break
            tgt_idx, ins, new_target = best_target
            # If target route was non-empty, eject one of its customers to
            # continue the chain — pick the one whose removal saves the most.
            if depth < max_chain_length - 1 and len(chain_routes[tgt_idx]) > 0:
                # Eject the candidate whose removal from new_target saves
                # the most travel time.
                candidates = []
                for k, cid in enumerate(new_target):
                    if cid == cur_cust:
                        continue
                    trial = new_target[:k] + new_target[k + 1:]
                    ok, _ = _route_arrival_and_close(inst, trial)
                    if not ok:
                        continue
                    save = _route_time(inst, new_target) - _route_time(inst, trial)
                    candidates.append((save, k, cid))
                if not candidates:
                    chain_routes[tgt_idx] = new_target
                    break
                candidates.sort(reverse=True)
                _, k, ejected_cust = candidates[0]
                chain_routes[tgt_idx] = new_target[:k] + new_target[k + 1:]
                ejected.append((tgt_idx, ejected_cust))
            else:
                chain_routes[tgt_idx] = new_target
                break

        if not chain_ok:
            # The last ejected customer couldn't be placed; try inserting it
            # back into its origin's route as a last-resort close.
            last_route_origin, last_cust = ejected[-1]
            base = chain_routes[last_route_origin]
            placed = False
            for ins in range(len(base) + 1):
                new_route = base[:ins] + [last_cust] + base[ins:]
                ok, _ = _route_arrival_and_close(inst, new_route)
                if ok:
                    chain_routes[last_route_origin] = new_route
                    placed = True
                    break
            if not placed:
                continue  # chain failed, try another seed

        cand = Solution(
            instance_id=inst.instance_id,
            routes=[Route(customers=r) for r in chain_routes],
            solver=best.solver,
            wall_clock_seconds=best.wall_clock_seconds,
            budget_seconds=best.budget_seconds, feasible=False,
        )
        cand.metrics = evaluate(inst, cand, settings)
        cand.feasible = bool(cand.metrics["feasible"])
        if cand.feasible and cand.metrics["operational_cost"] < best_cost - 1e-6:
            best = cand
            best_cost = cand.metrics["operational_cost"]
    return best


def swap_star(inst: Instance, sol: Solution, settings: Settings,
              max_seconds: float = 2.0) -> Solution:
    """SwapStar (Vidal 2022, HGS-DIMACS): for each route-pair (A, B) and
    customers u in A, v in B, swap them — but u goes to its BEST position
    in B and v goes to its BEST position in A (not necessarily the slot the
    other vacated).  This finds moves no simple swap can.

    Implementation note: O(|A|*|B|*max(|A|,|B|)) per route pair, capped by
    the deadline.  Strict-improvement acceptance.
    """
    deadline = time.perf_counter() + max_seconds
    best = sol
    best_cost = sol.metrics["operational_cost"]
    improved = True

    while improved and time.perf_counter() < deadline:
        improved = False
        for ri_a in range(len(best.routes)):
            if improved or time.perf_counter() >= deadline:
                break
            for ri_b in range(ri_a + 1, len(best.routes)):
                if improved or time.perf_counter() >= deadline:
                    break
                ra, rb = best.routes[ri_a].customers, best.routes[ri_b].customers
                if not ra or not rb:
                    continue
                for ia, u in enumerate(ra):
                    if improved or time.perf_counter() >= deadline:
                        break
                    for ib, v in enumerate(rb):
                        # Remove u from ra, v from rb.
                        ra_without_u = ra[:ia] + ra[ia + 1:]
                        rb_without_v = rb[:ib] + rb[ib + 1:]
                        # Best position for v in ra_without_u.
                        best_v_pos = None
                        best_v_delta = float("inf")
                        for pos in range(len(ra_without_u) + 1):
                            cand = ra_without_u[:pos] + [v] + ra_without_u[pos:]
                            ok, _ = _route_arrival_and_close(inst, cand)
                            if not ok:
                                continue
                            d = _route_time(inst, cand) - _route_time(inst, ra_without_u)
                            if d < best_v_delta:
                                best_v_delta = d
                                best_v_pos = cand
                        if best_v_pos is None:
                            continue
                        # Best position for u in rb_without_v.
                        best_u_pos = None
                        best_u_delta = float("inf")
                        for pos in range(len(rb_without_v) + 1):
                            cand = rb_without_v[:pos] + [u] + rb_without_v[pos:]
                            ok, _ = _route_arrival_and_close(inst, cand)
                            if not ok:
                                continue
                            d = _route_time(inst, cand) - _route_time(inst, rb_without_v)
                            if d < best_u_delta:
                                best_u_delta = d
                                best_u_pos = cand
                        if best_u_pos is None:
                            continue
                        # Accept by full evaluate (final answer wins).
                        cand_routes = list(best.routes)
                        cand_routes[ri_a] = Route(customers=best_v_pos)
                        cand_routes[ri_b] = Route(customers=best_u_pos)
                        cand = Solution(
                            instance_id=inst.instance_id, routes=cand_routes,
                            solver=best.solver,
                            wall_clock_seconds=best.wall_clock_seconds,
                            budget_seconds=best.budget_seconds, feasible=False,
                        )
                        cand.metrics = evaluate(inst, cand, settings)
                        cand.feasible = bool(cand.metrics["feasible"])
                        if cand.metrics["operational_cost"] < best_cost - 1e-6:
                            best = cand
                            best_cost = cand.metrics["operational_cost"]
                            improved = True
                            break
    return best


def vehicle_kill(inst: Instance, sol: Solution, settings: Settings,
                 max_seconds: float = 2.0,
                 fixed_vehicle_cost: float = 0.0) -> Solution:
    """Try to ELIMINATE each route by redistributing its customers, allowing
    DROPS for customers whose cheapest re-insertion cost exceeds the miss
    penalty.  This closes the N=100 LKH-3 gap where 'serve everyone' is
    over-eager — sometimes a route running for a single awkward customer
    is cheaper to kill outright.

    Per Track-1 drop-permission research (May 2026).
    """
    deadline = time.perf_counter() + max_seconds
    miss_pen = float(settings.economics.hard_late_penalty)
    wage = settings.economics.wage_per_minute

    best = sol
    best_cost = sol.metrics["operational_cost"]
    improved = True

    while improved and time.perf_counter() < deadline:
        improved = False
        # Try smallest non-empty routes first — most likely to be killable.
        ranked = sorted(
            [(i, len(r.customers)) for i, r in enumerate(best.routes) if r.customers],
            key=lambda x: x[1],
        )
        for src_idx, _ in ranked:
            if time.perf_counter() >= deadline:
                break
            src_seq = list(best.routes[src_idx].customers)
            other_routes = [list(r.customers) for i, r in enumerate(best.routes) if i != src_idx]

            # Cost of keeping the src route alive: its wage-time + fixed cost.
            ok, end_clock = _route_arrival_and_close(inst, src_seq)
            if not ok:
                continue
            keep_cost = wage * (end_clock - inst.depot.ready) + fixed_vehicle_cost

            # Try to redistribute src_seq into other_routes, allowing drops.
            scratch = [list(r) for r in other_routes]
            reinsert_cost = 0.0
            kept_drops: list[int] = []
            for cid in src_seq:
                best_pos = None
                best_delta = float("inf")
                for ri, base in enumerate(scratch):
                    for pos in range(len(base) + 1):
                        new_route = base[:pos] + [cid] + base[pos:]
                        ok2, _ = _route_arrival_and_close(inst, new_route)
                        if not ok2:
                            continue
                        delta = _route_time(inst, new_route) - _route_time(inst, base)
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = (ri, pos, new_route)
                if best_pos is None or best_delta * wage > miss_pen:
                    # Cheaper to drop than to insert anywhere feasibly.
                    kept_drops.append(cid)
                    reinsert_cost += miss_pen
                else:
                    ri, _pos, new_route = best_pos
                    scratch[ri] = new_route
                    reinsert_cost += wage * best_delta

            if reinsert_cost + 1e-6 < keep_cost:
                # Commit the kill.
                new_full = [None] * len(best.routes)
                new_full[src_idx] = Route(customers=[])
                j = 0
                for i in range(len(best.routes)):
                    if i == src_idx:
                        continue
                    new_full[i] = Route(customers=scratch[j])
                    j += 1
                cand = Solution(
                    instance_id=inst.instance_id, routes=new_full,
                    solver=best.solver,
                    wall_clock_seconds=best.wall_clock_seconds,
                    budget_seconds=best.budget_seconds, feasible=False,
                )
                cand.metrics = evaluate(inst, cand, settings)
                cand.feasible = bool(cand.metrics["feasible"])
                if cand.metrics["operational_cost"] < best_cost - 1e-6:
                    best = cand
                    best_cost = cand.metrics["operational_cost"]
                    improved = True
                    break  # rescore: route indices changed
    return best


def two_opt_star(inst: Instance, sol: Solution, settings: Settings,
                 max_seconds: float = 2.0) -> Solution:
    """Inter-route 2-opt*: swap the tails of two routes.  For routes
    A = [a1 ... ai | ai+1 ... am] and B = [b1 ... bj | bj+1 ... bn],
    produce A' = [a1 ... ai, bj+1 ... bn] and B' = [b1 ... bj, ai+1 ... am].
    The dominant cross-route operator in HGS-DIMACS / PyVRP.  Track-2 research:
    'biggest gap our solver has vs SOTA'.
    """
    deadline = time.perf_counter() + max_seconds
    best = sol
    best_cost = sol.metrics["operational_cost"]
    improved = True

    while improved and time.perf_counter() < deadline:
        improved = False
        routes = [list(r.customers) for r in best.routes]
        for ri_a in range(len(routes)):
            if time.perf_counter() >= deadline:
                break
            for ri_b in range(ri_a + 1, len(routes)):
                ra, rb = routes[ri_a], routes[ri_b]
                if not ra or not rb:
                    continue
                # Try every cut point in A and B.
                for i in range(len(ra) + 1):
                    for j in range(len(rb) + 1):
                        if i == len(ra) and j == len(rb):
                            continue  # no-op
                        if i == 0 and j == 0:
                            continue  # full swap = label swap, useless
                        new_a = ra[:i] + rb[j:]
                        new_b = rb[:j] + ra[i:]
                        ok_a, _ = _route_arrival_and_close(inst, new_a)
                        if not ok_a:
                            continue
                        ok_b, _ = _route_arrival_and_close(inst, new_b)
                        if not ok_b:
                            continue
                        cand_routes = list(routes)
                        cand_routes[ri_a] = new_a
                        cand_routes[ri_b] = new_b
                        cand = Solution(
                            instance_id=inst.instance_id,
                            routes=[Route(customers=r) for r in cand_routes],
                            solver=best.solver,
                            wall_clock_seconds=best.wall_clock_seconds,
                            budget_seconds=best.budget_seconds, feasible=False,
                        )
                        cand.metrics = evaluate(inst, cand, settings)
                        cand.feasible = bool(cand.metrics["feasible"])
                        if cand.metrics["operational_cost"] < best_cost - 1e-6:
                            best = cand
                            best_cost = cand.metrics["operational_cost"]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best


def merge_routes(inst: Instance, sol: Solution, settings: Settings,
                 max_seconds: float = 2.0) -> Solution:
    """Try to empty the shortest non-empty route by relocating ALL of its
    customers into other routes.  Accept the bundle move only if total
    operational cost drops AND every relocated customer is TW-feasible in
    its destination route.  This is the strongest move for cutting vehicle
    count when wage cost dominates."""
    deadline = time.perf_counter() + max_seconds
    best = sol
    best_cost = sol.metrics["operational_cost"]
    improved = True

    while improved and time.perf_counter() < deadline:
        improved = False
        # Sort non-empty routes by length ascending — try the cheapest to evacuate first.
        idx_by_len = sorted(
            [(i, len(r.customers)) for i, r in enumerate(best.routes) if r.customers],
            key=lambda x: x[1],
        )
        for src_idx, _ in idx_by_len:
            if time.perf_counter() >= deadline:
                break
            src_customers = list(best.routes[src_idx].customers)
            # Try greedy best-insertion of each customer into some other route.
            trial_routes = [list(r.customers) for r in best.routes]
            trial_routes[src_idx] = []
            placed_all = True
            for cid in src_customers:
                best_pos = None
                best_delta = float("inf")
                for ri in range(len(trial_routes)):
                    if ri == src_idx:
                        continue
                    base = trial_routes[ri]
                    for pos in range(len(base) + 1):
                        new_route = base[:pos] + [cid] + base[pos:]
                        ok, _ = _route_arrival_and_close(inst, new_route)
                        if not ok:
                            continue
                        # delta cost: difference in route's travel time before vs after
                        # is a cheap proxy; final acceptance uses full evaluate().
                        before = _route_time(inst, base)
                        after  = _route_time(inst, new_route)
                        delta = after - before
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = (ri, pos, new_route)
                if best_pos is None:
                    placed_all = False
                    break
                ri, _pos, new_route = best_pos
                trial_routes[ri] = new_route
            if not placed_all:
                continue
            cand = Solution(
                instance_id=inst.instance_id,
                routes=[Route(customers=r) for r in trial_routes],
                solver=best.solver,
                wall_clock_seconds=best.wall_clock_seconds,
                budget_seconds=best.budget_seconds, feasible=False,
            )
            cand.metrics = evaluate(inst, cand, settings)
            cand.feasible = bool(cand.metrics["feasible"])
            if cand.metrics["operational_cost"] < best_cost - 1e-6:
                best = cand
                best_cost = cand.metrics["operational_cost"]
                improved = True
                break
    return best


def _route_time(inst: Instance, customers: list[int]) -> float:
    if not customers:
        return 0.0
    T = inst.travel_time
    cust_by_id = _cust_by_id(inst)
    clock = float(inst.depot.ready)
    cur = 0
    for cid in customers:
        c = cust_by_id[cid]
        arrive = clock + float(T[cur, cid])
        start  = max(arrive, float(c.ready))
        clock = start + c.service
        cur = cid
    return clock + float(T[cur, 0]) - inst.depot.ready


def relocate(inst: Instance, sol: Solution, settings: Settings,
             max_seconds: float = 5.0) -> Solution:
    """First-improvement relocate move: try moving each customer to every
    other position (in same or different route).  Accept the first move that
    lowers total operational cost AND preserves TW feasibility everywhere.

    O(N^2) per pass; capped by max_seconds.
    """
    # Granular filter is conditional on N: at small/medium N the filter
    # cuts useful moves (saw +1.5% N=200 regression).  Only enable for
    # N >= 350 where the O(K²·M²) cost actually bites.
    use_granular = inst.num_customers >= 350
    if use_granular:
        from svrptw.solvers.common.neighbors import get_neighbors
        gn = get_neighbors(inst, k=30)
    deadline = time.perf_counter() + max_seconds
    best = sol
    best_cost = sol.metrics["operational_cost"]
    improved = True

    while improved and time.perf_counter() < deadline:
        improved = False
        routes = [list(r.customers) for r in best.routes]
        for ri_from, r_from in enumerate(routes):
            if time.perf_counter() >= deadline:
                break
            for pos_from in range(len(r_from)):
                cid = r_from[pos_from]
                if use_granular:
                    cid_neigh = set(gn.neighbors_of(cid).tolist())
                for ri_to in range(len(routes)):
                    r_to = routes[ri_to]
                    if use_granular and ri_to != ri_from and r_to and not any(c in cid_neigh for c in r_to):
                        continue
                    for pos_to in range(len(r_to) + 1):
                        if ri_from == ri_to and (pos_to == pos_from or pos_to == pos_from + 1):
                            continue  # no-op move
                        # Construct candidate routes
                        new_from = r_from[:pos_from] + r_from[pos_from + 1:]
                        if ri_from == ri_to:
                            # Adjust insertion index after removal.
                            adj = pos_to if pos_to < pos_from else pos_to - 1
                            new_to = new_from[:adj] + [cid] + new_from[adj:]
                            new_from = new_to
                            ok_from, _ = _route_arrival_and_close(inst, new_from)
                            if not ok_from:
                                continue
                            candidate_routes = list(routes)
                            candidate_routes[ri_from] = new_from
                        else:
                            new_to = r_to[:pos_to] + [cid] + r_to[pos_to:]
                            ok_from, _ = _route_arrival_and_close(inst, new_from)
                            ok_to,   _ = _route_arrival_and_close(inst, new_to)
                            if not (ok_from and ok_to):
                                continue
                            candidate_routes = list(routes)
                            candidate_routes[ri_from] = new_from
                            candidate_routes[ri_to]   = new_to
                        cand = Solution(
                            instance_id=inst.instance_id,
                            routes=[Route(customers=r) for r in candidate_routes],
                            solver=best.solver,
                            wall_clock_seconds=best.wall_clock_seconds,
                            budget_seconds=best.budget_seconds,
                            feasible=False,
                        )
                        cand.metrics = evaluate(inst, cand, settings)
                        cand.feasible = bool(cand.metrics["feasible"])
                        if cand.metrics["operational_cost"] < best_cost - 1e-6:
                            best = cand
                            best_cost = cand.metrics["operational_cost"]
                            improved = True
                            break
                    if improved or time.perf_counter() >= deadline:
                        break
                if improved or time.perf_counter() >= deadline:
                    break
            if improved or time.perf_counter() >= deadline:
                break
    return best
