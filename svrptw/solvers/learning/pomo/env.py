"""VRPTW environment for the POMO rollout.

A batched, vectorized step() returning state, reward, done.  Stays on
GPU once instantiated; the only host transfer is the final tour decoded
into Solutions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from svrptw.io import Instance


@dataclass
class VRPTWState:
    visited: torch.Tensor      # (B, K, N+1) bool — N+1 includes the depot
    cur_node: torch.Tensor     # (B, K) long
    cur_time: torch.Tensor     # (B, K) float — current clock
    cur_load: torch.Tensor     # (B, K) float
    tours: list[list[list[int]]]   # per (b, k), list of routes (list of cust ids)


class VRPTWEnv:
    """Batched VRPTW environment.

    Currently a scaffold — the step() and mask() methods are the contract
    surface; the training loop in train.py uses them to drive the POMO
    rollout.  The forward feasibility math mirrors `local_search._route_arrival_and_close`
    so the trained policy matches the offline evaluator exactly.
    """

    def __init__(self, instance: Instance, n_starts: int = 32,
                 device: str | torch.device = "cpu"):
        self.inst = instance
        self.n_starts = min(n_starts, instance.num_customers)
        self.device = torch.device(device)
        self.n_total = instance.num_customers + 1   # +depot
        # Node features: (x, y, demand, ready, due, service)
        feats = np.zeros((self.n_total, 6), dtype=np.float32)
        d = instance.depot
        feats[0] = [d.x, d.y, 0.0, d.ready, d.due, 0.0]
        for c in instance.customers:
            feats[c.id] = [c.x, c.y, c.demand, c.ready, c.due, c.service]
        # Normalize coords
        coord_min = feats[:, :2].min(axis=0)
        coord_max = feats[:, :2].max(axis=0)
        feats[:, :2] = (feats[:, :2] - coord_min) / np.maximum(coord_max - coord_min, 1e-6)
        # Normalize demand
        feats[:, 2] = feats[:, 2] / max(1.0, float(instance.vehicle_capacity))
        # Normalize times by day length
        day_len = max(1.0, float(d.due - d.ready))
        feats[:, 3:6] = (feats[:, 3:6] - d.ready) / day_len
        self.node_feats = torch.from_numpy(feats).to(self.device)
        # Edge features: (T[i,j], T[j,i]) normalized
        T = np.asarray(instance.travel_time, dtype=np.float32)
        scale = max(1.0, float(T.max()))
        T_norm = T / scale
        edge = np.stack([T_norm, T_norm.T], axis=-1)
        self.edge_feats = torch.from_numpy(edge).to(self.device)
        self.T = torch.from_numpy(T).to(self.device)
        # Precompute per-node tensor attributes — eliminates the K×N
        # python-side .item() loop in action_mask().
        # Index 0 = depot, 1..N = customers.
        N = self.n_total
        demand = np.zeros(N, dtype=np.float32)
        ready = np.zeros(N, dtype=np.float32)
        due = np.zeros(N, dtype=np.float32)
        service = np.zeros(N, dtype=np.float32)
        due[0] = float(d.due)
        ready[0] = float(d.ready)
        for c in instance.customers:
            demand[c.id] = float(c.demand)
            ready[c.id] = float(c.ready)
            due[c.id] = float(c.due)
            service[c.id] = float(c.service)
        self.demand = torch.from_numpy(demand).to(self.device)
        self.ready_t = torch.from_numpy(ready).to(self.device)
        self.due_t = torch.from_numpy(due).to(self.device)
        self.service_t = torch.from_numpy(service).to(self.device)
        self._cap = float(instance.vehicle_capacity)
        self._depot_due = float(d.due)

    def reset(self) -> VRPTWState:
        K = self.n_starts
        N = self.n_total
        visited = torch.zeros(1, K, N, dtype=torch.bool, device=self.device)
        # Depot stays unvisited so the agent can return; we just mark it
        # invalid for selection until at least one customer is in the route.
        first_customers = torch.arange(1, K + 1, device=self.device)
        # Mark the K start customers as visited from step 0.
        for k, fc in enumerate(first_customers.tolist()):
            visited[0, k, fc] = True
        cur_node = first_customers.unsqueeze(0).clone()
        # cur_time = travel from depot + service for first customer
        cur_time = torch.zeros(1, K, device=self.device, dtype=torch.float32)
        cur_load = torch.zeros(1, K, device=self.device, dtype=torch.float32)
        for k, fc in enumerate(first_customers.tolist()):
            c = self.inst.customers[fc - 1]
            arrive = float(self.T[0, fc])
            start = max(arrive, float(c.ready))
            cur_time[0, k] = float(start + c.service)
            cur_load[0, k] = float(c.demand)
        tours = [[[int(c)] for c in first_customers.cpu().tolist()]]
        return VRPTWState(visited=visited, cur_node=cur_node,
                          cur_time=cur_time, cur_load=cur_load, tours=tours)

    def action_mask(self, state: VRPTWState) -> torch.Tensor:
        """Return (1, K, N+1) mask where True = invalid action.

        Fully vectorised — every step now O(K + N) tensor ops on GPU
        instead of K*N python-level .item() calls. ~10× faster at N=100.
        """
        N = self.n_total
        visited = state.visited[0]      # (K, N) bool
        cur = state.cur_node[0]         # (K,) long
        clk = state.cur_time[0]         # (K,)
        load = state.cur_load[0]        # (K,)

        # Travel time from each current node to every candidate j: (K, N).
        T_cur_j = self.T[cur]            # advanced indexing: (K, N)
        T_j_depot = self.T[:, 0]         # (N,) — return-to-depot time

        # Broadcast-compute over-capacity, TW-late, end-past-day, all (K, N) bools.
        over_cap = (load.unsqueeze(1) + self.demand.unsqueeze(0)) > self._cap
        arrive = clk.unsqueeze(1) + T_cur_j                # (K, N)
        start = torch.maximum(arrive, self.ready_t.unsqueeze(0))
        late = start > self.due_t.unsqueeze(0)
        end = start + self.service_t.unsqueeze(0) + T_j_depot.unsqueeze(0)
        past_day = end > self._depot_due

        mask = visited | over_cap | late | past_day        # (K, N) bool

        # Depot column (j=0): depot is invalid iff vehicle is AT depot
        # (cur == 0) AND any other customer is still feasible.
        at_depot = (cur == 0)                              # (K,)
        # Recompute "depot column" without the visited-mask-on-depot constraint
        # so the depot itself is selectable. mask[:, 0] reflects visited[0]
        # (depot is normally False there) but we need explicit handling.
        # Any feasible customer remaining for this rollout? (skip depot col)
        any_cust_open = (~mask[:, 1:]).any(dim=1)          # (K,)
        depot_invalid = at_depot & any_cust_open
        mask[:, 0] = depot_invalid

        return mask.unsqueeze(0)

    def step(self, state: VRPTWState, action: torch.Tensor) -> VRPTWState:
        """Apply `action` (1, K) for one decode step.  Action == 0 means
        return to depot (closes current route, starts a new one).

        Vectorised: the only Python-side loop left is the tours list
        update (~K ops, no .item() calls in inner math).
        """
        # Caller may pass action on CPU (legacy train.py path); align.
        a = action[0].to(self.device)                # (K,) long
        cur = state.cur_node[0]                      # (K,)
        clk = state.cur_time[0]                      # (K,)
        load = state.cur_load[0]                     # (K,)

        is_depot = (a == 0)                          # (K,) bool

        # Customer-step math, computed unconditionally over all K (cheap),
        # then masked-merged with the depot-reset values.
        # Index per (k): T[cur[k], a[k]]
        idx_k = torch.arange(a.size(0), device=self.device)
        travel = self.T[cur, a]                       # (K,)
        arrive = clk + travel
        start = torch.maximum(arrive, self.ready_t[a])
        next_time_cust = start + self.service_t[a]
        next_load_cust = load + self.demand[a]

        zero = torch.zeros_like(clk)
        new_time = torch.where(is_depot, zero, next_time_cust)
        new_load = torch.where(is_depot, zero, next_load_cust)
        new_cur = torch.where(is_depot, torch.zeros_like(a), a)

        new_visited = state.visited.clone()
        # Mark visited for non-depot actions only.
        cust_idx = idx_k[~is_depot]
        if cust_idx.numel() > 0:
            new_visited[0, cust_idx, a[cust_idx]] = True

        # Tours list update — list-of-lists, can't vectorise but only ~K appends.
        new_tours = [list(state.tours[0])]
        a_cpu = a.cpu().tolist()
        for k in range(len(a_cpu)):
            new_tours[0][k].append(int(a_cpu[k]))

        return VRPTWState(visited=new_visited, cur_node=new_cur.unsqueeze(0),
                          cur_time=new_time.unsqueeze(0),
                          cur_load=new_load.unsqueeze(0), tours=new_tours)

    def is_done(self, state: VRPTWState) -> bool:
        """All non-depot nodes visited."""
        visited = state.visited[0, :, 1:]
        return bool(visited.all().item())

    def total_cost(self, state: VRPTWState, settings) -> list[float]:
        """Compute final operational cost per rollout via the canonical
        evaluator.  Returns a list of length K."""
        from svrptw.solvers.common import Route, Solution, evaluate
        costs: list[float] = []
        for k in range(self.n_starts):
            # Split tour at depot returns into per-vehicle routes.
            tour = state.tours[0][k]
            routes: list[list[int]] = []
            cur_route: list[int] = []
            for node in tour:
                if node == 0:
                    if cur_route:
                        routes.append(cur_route)
                        cur_route = []
                else:
                    cur_route.append(node)
            if cur_route:
                routes.append(cur_route)
            sol = Solution(
                instance_id=self.inst.instance_id,
                routes=[Route(customers=r) for r in routes],
                solver="pomo",
                wall_clock_seconds=0.0,
                budget_seconds=0.0,
                feasible=False,
            )
            sol.metrics = evaluate(self.inst, sol, settings)
            costs.append(float(sol.metrics["operational_cost"]))
        return costs
