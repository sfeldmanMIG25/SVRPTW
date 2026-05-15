"""POMO inference: load a checkpoint, run K-start rollouts, return the
best Solution.  Wraps as `svrptw.solvers.learning.pomo.infer.solve()`.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common import Route, Solution, evaluate
from svrptw.solvers.learning.pomo.env import VRPTWEnv
from svrptw.solvers.learning.pomo.model import POMOConfig, POMOModel

DEFAULT_CKPT = Path("models/pomo_v3/pomo_N50_e80.pt")


_MODEL_CACHE: dict[str, tuple[POMOModel, POMOConfig, torch.device]] = {}


def _load_model(ckpt_path: Path) -> tuple[POMOModel, POMOConfig, torch.device]:
    key = str(ckpt_path.resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = POMOConfig(**payload["cfg"])
    model = POMOModel(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    _MODEL_CACHE[key] = (model, cfg, device)
    return model, cfg, device


def solve(inst: Instance, settings: Settings,
          checkpoint: str | Path | None = None,
          n_starts: int = 32, greedy_decode: bool = False) -> Solution:
    """Load checkpoint, run a K-start POMO rollout, return the best
    Solution by operational cost.  If `checkpoint` is None, falls back
    to the DEFAULT_CKPT path; if that's missing, returns the greedy
    baseline so the bench harness can keep going."""
    t0 = time.perf_counter()
    ckpt_path = Path(checkpoint) if checkpoint else DEFAULT_CKPT
    if not ckpt_path.exists():
        from svrptw.solvers.classical import greedy as greedy_mod
        out = greedy_mod.solve(inst, settings)
        out.solver = "pomo_v1(fallback=greedy)"
        out.wall_clock_seconds = time.perf_counter() - t0
        return out

    model, cfg, device = _load_model(ckpt_path)
    K = min(n_starts, inst.num_customers)
    env = VRPTWEnv(inst, n_starts=K, device=device)

    with torch.no_grad():
        nf = env.node_feats.unsqueeze(0).to(device)
        ef = env.edge_feats.unsqueeze(0).to(device)
        embeds = model.encode(nf, ef)
        state = env.reset()
        max_steps = env.n_total * 2
        steps = 0
        while not env.is_done(state) and steps < max_steps:
            cap_norm = (1.0 - state.cur_load / max(1.0, env.inst.vehicle_capacity))
            t_norm = state.cur_time / max(1.0, env.inst.depot.due - env.inst.depot.ready)
            scalars = torch.stack([cap_norm, t_norm], dim=-1).to(device)
            cur_idx = state.cur_node.to(device).squeeze(0)
            mask = env.action_mask(state).to(device).squeeze(0)
            logits = model.step_logits(embeds.expand(K, -1, -1), cur_idx,
                                       scalars.squeeze(0), mask)
            if greedy_decode:
                action = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs.clamp_min(1e-12), num_samples=1).squeeze(-1)
            state = env.step(state, action.unsqueeze(0).cpu())
            steps += 1

        costs = env.total_cost(state, settings)

    # Pick the best rollout.
    best_k = int(min(range(len(costs)), key=lambda i: costs[i]))
    tour = state.tours[0][best_k]
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
        solver="pomo_v1",
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
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve(inst, Settings(), checkpoint=args.ckpt, n_starts=args.k,
                greedy_decode=args.greedy)
    print(json.dumps({"solver": sol.solver, "metrics": sol.metrics,
                      "wall_clock_s": sol.wall_clock_seconds}, indent=2))
