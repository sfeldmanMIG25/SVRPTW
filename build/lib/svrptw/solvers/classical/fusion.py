"""Fusion solver — merge POMO warmstart + portfolio bandit fine-tune.

This is SPEC-9-FUSION-01 (to be written). Architecture:

  1. Use POMO to generate K diverse construction starts (~3 s).
  2. For each of the top-M (by initial cost) starts, run the portfolio
     bandit for ~`budget_per_start` seconds.
  3. Return the best solution across all M bandit runs.

Why this composition wins:
  - POMO sees the full neural prior over decoder choices and produces
    structurally diverse starts that pure construction heuristics
    (auction, greedy) miss. K=32 starts give the bandit a richer
    initial-state distribution than the single auction warm-up.
  - portfolio bandit excels at adaptive operator selection on a
    single trajectory; pairing it with diverse starts amortizes its
    plateau-stuck failure mode.
  - PyVRP's strength is its native LNS but it's monolithic — can't
    accept a warmstart from POMO without reformatting. Bandit can.

Performance expectation: at total budget=10s, M=3 bandit runs at
~3s each + 1s POMO. Should beat portfolio-alone at 10s on instances
where POMO's diversity gives a structurally-better start.

Risk: if POMO's checkpoint is missing, fall back gracefully.
"""
from __future__ import annotations

import time
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.classical import portfolio as pm
from svrptw.solvers.classical import greedy as greedy_mod
from svrptw.solvers.common import Route, Solution, evaluate
from svrptw.solvers.learning.pomo.infer import DEFAULT_CKPT


def _pomo_warmstart_K(inst: Instance, settings: Settings, K: int = 8) -> list[Solution]:
    """Return K constructed solutions from K POMO rollouts; falls back to
    [greedy] if checkpoint missing."""
    if not Path(DEFAULT_CKPT).exists():
        return [greedy_mod.solve(inst, settings)]
    try:
        # Inline the inference loop to extract ALL K tours, not just best.
        import torch
        from svrptw.solvers.learning.pomo.env import VRPTWEnv
        from svrptw.solvers.learning.pomo.infer import _load_model

        model, _, device = _load_model(DEFAULT_CKPT)
        K_eff = min(K, inst.num_customers)
        env = VRPTWEnv(inst, n_starts=K_eff, device=device)
        with torch.no_grad():
            nf = env.node_feats.unsqueeze(0).to(device)
            ef = env.edge_feats.unsqueeze(0).to(device)
            embeds = model.encode(nf, ef)
            state = env.reset()
            steps = 0
            max_steps = env.n_total * 2
            while not env.is_done(state) and steps < max_steps:
                cap_norm = (1.0 - state.cur_load / max(1.0, env.inst.vehicle_capacity))
                t_norm = state.cur_time / max(1.0, env.inst.depot.due - env.inst.depot.ready)
                scalars = torch.stack([cap_norm, t_norm], dim=-1).to(device)
                cur_idx = state.cur_node.to(device).squeeze(0)
                mask = env.action_mask(state).to(device).squeeze(0)
                logits = model.step_logits(embeds.expand(K_eff, -1, -1), cur_idx,
                                           scalars.squeeze(0), mask)
                probs = torch.softmax(logits, dim=-1).clamp_min(1e-12)
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)
                state = env.step(state, action.unsqueeze(0).cpu())
                steps += 1
        out: list[Solution] = []
        for k in range(K_eff):
            tour = state.tours[0][k]
            routes: list[list[int]] = []
            cur: list[int] = []
            for node in tour:
                if node == 0:
                    if cur:
                        routes.append(cur); cur = []
                else:
                    cur.append(node)
            if cur:
                routes.append(cur)
            s = Solution(
                instance_id=inst.instance_id,
                routes=[Route(customers=r) for r in routes],
                solver="pomo_warm", wall_clock_seconds=0.0,
                budget_seconds=0.0, feasible=False,
            )
            s.metrics = evaluate(inst, s, settings)
            s.feasible = bool(s.metrics["feasible"])
            out.append(s)
        return out
    except Exception:
        return [greedy_mod.solve(inst, settings)]


def solve(
    inst: Instance, settings: Settings,
    budget_seconds: float = 10.0,
    *,
    K_pomo: int = 8,
    M_topstarts: int = 3,
    pomo_overhead_seconds: float = 1.5,
) -> Solution:
    """Fusion solve. POMO produces K diverse starts; top-M (by cost)
    feed independent portfolio bandit runs sharing the remaining budget."""
    t0 = time.perf_counter()
    starts = _pomo_warmstart_K(inst, settings, K=K_pomo)
    if not starts:
        return pm.solve(inst, settings, budget_seconds=budget_seconds)

    # Pick top-M by cost (cheaper = better starts).
    starts.sort(key=lambda s: s.metrics["operational_cost"])
    starts = starts[:M_topstarts]

    remaining = budget_seconds - (time.perf_counter() - t0) - pomo_overhead_seconds
    per_start = max(0.5, remaining / max(1, len(starts)))

    best_sol = starts[0]
    for s in starts:
        # SPEC-9-FUSION-01: portfolio.solve(initial_solution=...) seeds
        # the bandit from the POMO start instead of auction_gart's
        # construction. Each top-M start fine-tunes independently.
        out = pm.solve(inst, settings, budget_seconds=per_start,
                       initial_solution=s)
        if out.metrics["operational_cost"] < best_sol.metrics["operational_cost"]:
            best_sol = out

    best_sol.solver = "fusion"
    best_sol.wall_clock_seconds = time.perf_counter() - t0
    best_sol.budget_seconds = budget_seconds
    return best_sol


if __name__ == "__main__":
    import argparse
    import json
    from svrptw.io import load_instance

    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--budget", type=float, default=10.0)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--M", type=int, default=3)
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve(inst, Settings(), budget_seconds=args.budget,
                K_pomo=args.K, M_topstarts=args.M)
    print(json.dumps({
        "solver": sol.solver,
        "operational_cost": sol.metrics["operational_cost"],
        "n_routes": int(sol.metrics["num_vehicles_used"]),
        "wall_clock_s": sol.wall_clock_seconds,
    }, indent=2))
