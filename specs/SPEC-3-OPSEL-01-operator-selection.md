# SPEC-3-OPSEL-01 — Trained operator-selection policy

```
ID:            SPEC-3-OPSEL-01
Title:         RL-trained operator selection for the portfolio solver
Owner role:    ML Engineer
Status:        DRAFT
Inputs:        Episodes of (state, op, reward) from the bandit baseline
Outputs:       models/op_policy/v1.pt + svrptw.solvers.learning.op_policy
```

## Goal

Replace the UCB1 bandit in SPEC-3-PORTFOLIO-01 with a learned policy
that conditions operator selection on instance + solution state. The
bandit adapts within one solve; the policy carries learning *across*
instances so cold starts are good.

Hyper-heuristic literature (Burke et al. 2013, Pisinger & Ropke 2019)
shows trained selectors beat UCB by 5-15% on heterogeneous instance
distributions — exactly our case (8 cities × 4 sizes).

## State (32-dim float vector)

Per-instance (12 dims):
- normalized N, K, capacity, fleet utilization
- asymmetry score, mean depot distance, mean pairwise distance
- TW tightness (mean / std of `due − ready − service`)
- one-hot 4 instance-size bins

Per-solution (16 dims):
- normalized current cost (against greedy baseline)
- num routes, mean route length, std route length, min/max route length
- num missed, mean wait, max wait
- routes-evacuable proxy (count of routes with ≤ 3 customers)
- cohesion proxy (mean intra-route distance / mean global distance)

Per-history (4 dims):
- time remaining, ops applied so far, last-improvement gap, plateaus

## Action: choose one of K operators

`K = 7` operators (relocate, or_opt_2, or_opt_3, swap_intra, swap_inter,
cross_exchange, ejection_chain). `merge_routes` and `perturb_destroy_repair`
are scheduled by separate logic (merge once-per-plateau, perturb on
restart).

## Architecture

Small MLP — 32 → 64 → 64 → 7. ReLU + LayerNorm. Softmax + temperature τ.
Total params ~7k. Fits the 8 GB GPU trivially. CPU inference < 1 ms.

## Training

- **Episodes:** one episode per (instance, seed) — randomly sampled from
  `instances/v1` with held-out cities.
- **Reward:** `−Δcost` per step + terminal `−final_cost / greedy_cost`
  to encourage long-horizon planning.
- **Algorithm:** REINFORCE with a moving-average baseline. PPO if variance
  is too high; the action space is small enough that REINFORCE should
  converge.
- **Budget:** 50k episodes; each episode is ~10 ops; ~2 hours single-GPU.

## Acceptance

- Trained policy beats the UCB bandit baseline by ≥ 5% on held-out cities
  (Cambridge, Pittsburgh) at the 10-second budget.
- Per-class operator selection probabilities are interpretable and
  differ by city topology (saved as a heatmap to
  `bench/figures/op_policy_per_class.pdf`).
- Inference latency ≤ 2 ms on CPU (so the selector is never the bottleneck).

## Non-goals

- Multi-agent / hierarchical policies. Single-agent flat selection.
- End-to-end neural solver. The policy chooses *which classical operator*
  to apply, not the move itself.

## Dependencies

- SPEC-3-PORTFOLIO-01 (the operators it chooses among).
- SPEC-3-EJECT-01 (one of the operators).
- v1 instance set frozen (needed for held-out splits).
