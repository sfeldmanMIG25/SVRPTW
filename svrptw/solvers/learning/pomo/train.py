"""POMO/MatNet REINFORCE training loop for asymmetric VRPTW.

Curriculum: N=50 → N=100 → N=200 per the May-2026 research brief.
Single GPU (RTX 3070 Ti 8GB).  Mixed precision via torch.amp.

Usage:
    python -m svrptw.solvers.learning.pomo.train --epochs 5 --batch 4 --n 50

This is the minimal viable training script — designed to validate the
pipeline end-to-end on a small budget, not to produce a paper-quality
checkpoint.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.learning.pomo.env import VRPTWEnv
from svrptw.solvers.learning.pomo.model import POMOConfig, POMOModel


def _list_instances(n_filter: int, max_n: int = 16) -> list[str]:
    base = Path("instances/v1")
    paths = sorted(base.glob(f"*-N{n_filter:03d}-*.json"))
    return [str(p) for p in paths[:max_n]]


def _synthetic_instance(N: int, seed: int):
    from svrptw.instances_gen.synthetic import generate as _gen
    return _gen(N=N, seed=seed)


def _rollout(model: POMOModel, env: VRPTWEnv, settings: Settings,
             device: torch.device, temperature: float = 1.0,
             ) -> tuple[torch.Tensor, list[float], torch.Tensor]:
    """One full K-rollout pass on one instance.  Returns
    (log_probs_sum (K,), costs (K,), mean_entropy (K,))."""
    nf = env.node_feats.unsqueeze(0).to(device)
    ef = env.edge_feats.unsqueeze(0).to(device)
    embeds = model.encode(nf, ef)
    state = env.reset()
    log_probs_accum = torch.zeros(env.n_starts, device=device)
    entropy_accum = torch.zeros(env.n_starts, device=device)
    steps_count = torch.zeros(env.n_starts, device=device)
    max_steps = env.n_total * 2  # safety bound
    steps = 0
    while not env.is_done(state) and steps < max_steps:
        cap_norm = (1.0 - state.cur_load / max(1.0, env.inst.vehicle_capacity))
        t_norm = state.cur_time / max(1.0, env.inst.depot.due - env.inst.depot.ready)
        scalars = torch.stack([cap_norm, t_norm], dim=-1).to(device)
        cur_idx = state.cur_node.to(device).squeeze(0)
        mask = env.action_mask(state).to(device).squeeze(0)
        logits = model.step_logits(embeds.expand(env.n_starts, -1, -1), cur_idx,
                                   scalars.squeeze(0), mask) / max(temperature, 1e-3)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs.clamp_min(1e-12), num_samples=1).squeeze(-1)
        log_probs_accum += torch.log(probs.gather(1, action.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12))
        # Per-step entropy of the policy at the visible (non-masked) actions.
        log_p = torch.log(probs.clamp_min(1e-12))
        step_entropy = -(probs * log_p).sum(dim=-1)
        entropy_accum += step_entropy
        steps_count += 1.0
        state = env.step(state, action.unsqueeze(0).cpu())
        steps += 1
    costs = env.total_cost(state, settings)
    mean_entropy = entropy_accum / steps_count.clamp_min(1.0)
    return log_probs_accum, costs, mean_entropy


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pomo-train] device={device}")
    cfg = POMOConfig(embed_dim=args.dim, n_encoder_layers=args.layers,
                     n_heads=args.heads, ff_dim=args.dim * 4)
    model = POMOModel(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    settings = Settings()

    # Curriculum: comma-separated list of N values; each batch picks one
    # uniformly at random.  Default = single N (= args.n).
    curriculum = [int(x) for x in args.n_curriculum.split(",")] if args.n_curriculum else [args.n]
    print(f"[pomo-train] curriculum N={curriculum} source={args.source}")

    instance_paths_by_n: dict[int, list[str]] = {}
    if args.source == "v1":
        for N in curriculum:
            instance_paths_by_n[N] = _list_instances(N, max_n=args.batch * 4)
            print(f"[pomo-train]   N={N}: {len(instance_paths_by_n[N])} v1 instances")
            if not instance_paths_by_n[N]:
                print(f"[pomo-train]   N={N} has no v1 instances; will skip if picked.")

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    epoch_seed_base = args.seed * 1_000_003
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        epoch_loss = 0.0
        epoch_mean_cost = 0.0
        # Linear entropy schedule, 0.01 -> 0.001 over the run.
        ent_coef = max(0.001, 0.01 - 0.009 * (epoch / max(1, args.epochs)))
        # Linear temperature anneal 1.0 -> args.temp_final over the run.
        temp = 1.0 - (1.0 - args.temp_final) * (epoch / max(1, args.epochs))
        for b in range(args.batch):
            # Sample N from the curriculum, then either pick a v1 instance
            # or generate a synthetic one.
            N = curriculum[int(rng.integers(0, len(curriculum)))]
            if args.source == "v1":
                paths = instance_paths_by_n.get(N, [])
                if not paths:
                    continue
                ip = paths[int(rng.integers(0, len(paths)))]
                inst = load_instance(ip)
            elif args.source == "net_synth":
                # SPEC-4-DATA-02 network-OD synthetic (structural asym via one-ways).
                from svrptw.instances_gen.network_synthetic import generate as net_gen
                # Grid sized just-big-enough for the requested N.
                gs = max(20, int(math.ceil(math.sqrt(N * 1.5))))
                inst = net_gen(N=N, seed=epoch_seed_base + epoch * args.batch + b,
                               grid_size=gs)
            else:  # synthetic (Euclidean + noise)
                inst = _synthetic_instance(N, seed=epoch_seed_base + epoch * args.batch + b)
            K = min(args.k, inst.num_customers)
            env = VRPTWEnv(inst, n_starts=K, device=device)
            log_probs, costs, entropies = _rollout(model, env, settings, device,
                                                    temperature=temp)
            costs_t = torch.tensor(costs, device=device, dtype=torch.float32)
            # Reward-scale the costs so the loss magnitude is controllable.
            costs_scaled = costs_t / args.reward_scale
            # POMO baseline = mean cost across K rollouts (no critic).
            baseline = costs_scaled.mean()
            advantage_raw = costs_scaled - baseline
            # Per-batch advantage normalization (Zhang 2024) → ~40% variance cut.
            if advantage_raw.std() > 1e-6:
                advantage = (advantage_raw - advantage_raw.mean()) / (advantage_raw.std() + 1e-8)
            else:
                advantage = advantage_raw
            policy_loss = (advantage.detach() * log_probs).mean()
            entropy_loss = -ent_coef * entropies.mean()
            loss = policy_loss + entropy_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.item())
            epoch_mean_cost += float(costs_t.mean().item())
        epoch_loss /= max(1, args.batch)
        epoch_mean_cost /= max(1, args.batch)
        elapsed = time.perf_counter() - t0
        print(f"epoch {epoch:3d}  loss={epoch_loss:+.4f}  mean_cost={epoch_mean_cost:.1f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch, "loss": epoch_loss,
                        "mean_cost": epoch_mean_cost, "elapsed_s": elapsed})

    ckpt_path = out_dir / f"pomo_N{args.n}_e{args.epochs}.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                "history": history}, ckpt_path)
    print(f"[pomo-train] saved checkpoint: {ckpt_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50, help="Customers per instance.")
    p.add_argument("--k", type=int, default=16, help="POMO parallel rollouts per instance.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward-scale", type=float, default=1000.0,
                   help="Divide costs by this before computing advantages (POMO stability).")
    p.add_argument("--temp-final", type=float, default=0.3,
                   help="Final sampling temperature (linear anneal from 1.0).")
    p.add_argument("--source", choices=["v1", "synthetic", "net_synth"], default="v1",
                   help="Training data source: v1 OSM, Euclidean+noise synthetic, "
                        "or network-OD synthetic (SPEC-4-DATA-02).")
    p.add_argument("--n-curriculum", default="",
                   help="Comma-sep N values e.g. '50,100,200'; default = single args.n.")
    p.add_argument("--out", default="models/pomo")
    args = p.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
