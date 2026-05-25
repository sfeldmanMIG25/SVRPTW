"""POMO + EAS-Emb (Hottung 2022 — Efficient Active Search, embedding-adaptive).

At inference time, freeze the base POMO model and add a learnable
per-instance embedding offset (an (N, d) parameter tensor initialized
to zero). Run K rollouts each iteration; REINFORCE-update the offset
toward rollouts with lower cost. After `n_iterations` steps, return
the best rollout seen.

Cost: ~ n_iterations × K rollouts × forward+backward per instance.
For N=50, K=32, n_iter=200, that's ~6400 rollouts per instance.
On a CPU this is ~30-60 s per instance; on a 3070 Ti ~3-5 s.

This is the "EAS-Emb" variant from Hottung et al. 2022 "Efficient
Active Search for Combinatorial Optimization Problems":
https://arxiv.org/abs/2106.05126
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common import Route, Solution, evaluate
from svrptw.solvers.learning.pomo.env import VRPTWEnv
from svrptw.solvers.learning.pomo.infer import DEFAULT_CKPT, _load_model


def _rollout_with_logp(
    model, embeds_adapted, env, state, K: int, device,
    *, greedy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One full episode for K rollouts. Returns (final_state.costs_tensor, sum_log_p)."""
    max_steps = env.n_total * 2
    log_p_sum = torch.zeros(K, device=device)
    steps = 0
    while not env.is_done(state) and steps < max_steps:
        cap_norm = (1.0 - state.cur_load / max(1.0, env.inst.vehicle_capacity))
        t_norm = state.cur_time / max(1.0, env.inst.depot.due - env.inst.depot.ready)
        scalars = torch.stack([cap_norm, t_norm], dim=-1).to(device)
        cur_idx = state.cur_node.to(device).squeeze(0)
        mask = env.action_mask(state).to(device).squeeze(0)
        logits = model.step_logits(embeds_adapted.expand(K, -1, -1), cur_idx,
                                   scalars.squeeze(0), mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        if greedy:
            action = logits.argmax(dim=-1)
        else:
            probs = log_probs.exp().clamp_min(1e-12)
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        log_p_sum = log_p_sum + log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        state = env.step(state, action.unsqueeze(0).cpu())
        steps += 1
    return state, log_p_sum


def solve_eas(
    inst: Instance, settings: Settings,
    *,
    checkpoint: str | Path | None = None,
    n_starts: int = 32,
    n_iterations: int = 200,
    lr: float = 1e-3,
    log_every: int = 50,
    seed: int = 0,
    max_wall_seconds: float | None = None,
) -> Solution:
    """Run EAS-Emb fine-tuning on `inst`, return the best Solution seen."""
    t0 = time.perf_counter()
    ckpt_path = Path(checkpoint) if checkpoint else DEFAULT_CKPT
    if not ckpt_path.exists():
        from svrptw.solvers.classical import greedy as greedy_mod
        sol = greedy_mod.solve(inst, settings)
        sol.solver = "pomo_eas(fallback=greedy)"
        sol.wall_clock_seconds = time.perf_counter() - t0
        return sol

    torch.manual_seed(seed)
    model, cfg, device = _load_model(ckpt_path)
    # Freeze the base model.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()  # batchnorm/dropout off

    K = min(n_starts, inst.num_customers)
    env = VRPTWEnv(inst, n_starts=K, device=device)

    # Encode once (frozen base, no grad).
    with torch.no_grad():
        nf = env.node_feats.unsqueeze(0).to(device)
        ef = env.edge_feats.unsqueeze(0).to(device)
        base_embeds = model.encode(nf, ef)  # (1, N, d)
    N_total = base_embeds.shape[1]
    d = base_embeds.shape[2]

    # The learnable adapter — the only trained parameter.
    adapter = nn.Parameter(torch.zeros(N_total, d, device=device))
    opt = torch.optim.Adam([adapter], lr=lr)

    best_cost = float("inf")
    best_state = None
    best_k = 0

    for it in range(n_iterations):
        # Wall-time guard for EAS-as-bandit-arm: stop before exceeding the
        # caller's deadline so the portfolio bandit's per-op budget holds.
        if max_wall_seconds is not None and (time.perf_counter() - t0) >= max_wall_seconds:
            break
        opt.zero_grad()
        embeds_adapted = base_embeds + adapter.unsqueeze(0)  # (1, N, d)
        state = env.reset()
        final_state, log_p_sum = _rollout_with_logp(
            model, embeds_adapted, env, state, K, device, greedy=False,
        )
        with torch.no_grad():
            costs = env.total_cost(final_state, settings)  # list[float] len=K
        costs_t = torch.tensor(costs, dtype=torch.float32, device=device)
        # Baseline = mean cost over K rollouts (variance reduction).
        adv = costs_t - costs_t.mean()
        # REINFORCE: minimize advantage * log_p (lower cost = better).
        loss = (adv * log_p_sum).mean()
        loss.backward()
        opt.step()

        # Track best rollout across all iterations.
        for k in range(K):
            if costs[k] < best_cost:
                best_cost = float(costs[k])
                best_state = final_state
                best_k = k

        if log_every > 0 and (it % log_every == 0):
            print(f"[eas] iter={it:>3}  K_mean={costs_t.mean().item():.1f}  "
                  f"K_min={costs_t.min().item():.1f}  best_so_far={best_cost:.1f}")

    # Convert best rollout's tour into a Solution.
    if best_state is None:
        # Should never happen, but fall back.
        from svrptw.solvers.classical import greedy as greedy_mod
        sol = greedy_mod.solve(inst, settings)
        sol.solver = "pomo_eas(fallback=greedy)"
        sol.wall_clock_seconds = time.perf_counter() - t0
        return sol

    tour = best_state.tours[0][best_k]
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
        instance_id=inst.instance_id,
        routes=[Route(customers=r) for r in routes],
        solver="pomo_eas",
        wall_clock_seconds=time.perf_counter() - t0,
        budget_seconds=0.0, feasible=False,
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
    p.add_argument("--ckpt", default=None)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve_eas(
        inst, Settings(),
        checkpoint=args.ckpt, n_starts=args.k,
        n_iterations=args.iters, lr=args.lr,
    )
    print(json.dumps({
        "solver": sol.solver,
        "operational_cost": sol.metrics["operational_cost"],
        "n_routes": int(sol.metrics["num_vehicles_used"]),
        "wall_clock_s": sol.wall_clock_seconds,
    }, indent=2))
