---
title: RL for neighborhood selection — design doc
project: SVRPTW
tags: [rl, bandit, design, phase-D]
updated: 2026-05-15
---

# RL for neighborhood selection — design doc (Phase D)

User asks (Human Reflection 10, item 3):
> "We want to have increasingly complex operators guided by a stochastic
> trained model that choose what neighborhoods to apply and when, with
> rewards for disposing of islands isolated stops and legs."

This doc is the concrete plan to replace / augment the LinUCB bandit
with a learned policy that takes the same inputs and adds the shaped
rewards the user wants.

## D1 — Audit of existing RL surface (done, 2026-05-15)

### State features (16 dims)

`svrptw/solvers/learning/state_features.py::featurize(inst, sol, greedy_cost, ops_applied, plateaus_so_far)` returns a 16-dim normalized vector:

| dim | name | normalization |
|-----|------|---------------|
| 0 | n_norm | N / 500 |
| 1 | k_used_frac | routes_used / num_vehicles |
| 2 | mean_route_len_norm | mean(route_lens) / 20 |
| 3 | std_route_len_norm | std(route_lens) / 20 |
| 4 | min_route_len_norm | min / 20 |
| 5 | max_route_len_norm | max / 20 |
| 6 | missed_frac | missed_deliveries / N |
| 7 | asym_score | OSM asymmetry (already 0..1) |
| 8 | cost_vs_greedy | current_cost / greedy_baseline |
| 9 | frac_short_routes | (routes with <=3 customers) / K_used |
| 10 | frac_empty_vehicles | (K - K_used) / K |
| 11 | tw_tightness | mean(due-ready) / day_length |
| 12 | depot_dist_mean_norm | T[0,1:].mean() / T.max() |
| 13 | time_used_frac | ops_applied / 50 |
| 14 | plateaus | plateaus_so_far / 10 |
| 15 | asym_x_n | asym * n / 500 (interaction term) |

**Already well-suited for RL.** No re-engineering needed. The vector
captures global instance shape (1, 7, 11, 12), current solution
quality (8, 9, 10, 6), and trajectory state (13, 14).

### Bandit surface (LinUCB)

`svrptw/solvers/learning/bandit.py::LinUCBBandit(ops, feature_dim=16, alpha=1.0, seed=0)`:
- `choose(context, eps=0.05)` → operator name (eps-greedy + tie-break random)
- `update(op, context, reward)` → linear posterior update
- `stats()` → per-arm pull counts + mean reward

Reward in pm.solve is `improvement_per_second` clipped to [-100, 200].

The interface is **policy-shaped** already — any class with `choose(context, eps)` and `update(op, context, reward)` can drop in.

### Operator pool (14 arms)

From `svrptw/solvers/classical/portfolio.py::_OPS`:
relocate, swap_star, two_opt_intra, two_opt_star, three_opt_intra,
merge_routes, sisr, ejection_chain, cyclic_3, vehicle_kill, soft_drop,
drop_route, destroy_island, drop_leg.
Plus optional council `extra_arms`. Action space = 14 (or larger).

### Training-data availability

`pm.solve` records `history: list[tuple[str, float]]` per call —
thin: (op_name, improvement). For offline RL we want
**(state, action, reward, next_state)** tuples. Two options:
1. Add a `LoggingBandit` wrapper around `LinUCBBandit` that also dumps
   full transitions to a JSONL file. Run benches with logging on for
   one day → dataset.
2. Re-derive transitions by re-running with a fixed seed and a
   replay-style ledger.

Option 1 is much simpler. Phase D3 starts there.

## D2 — Policy spec (next iteration)

### Action space choice

User said "operator + neighborhood". Two plausible expansions:

- **(operator, intensity)** — 14 ops × 3 intensities (light/med/heavy).
  Maps onto operator's `max_seconds` slot or, where applicable, a
  k-radius parameter (e.g. SISR's destruction size, ejection chain
  length). Action space = 42.

- **(operator, neighborhood-seed)** — 14 ops × {worst-util route, busiest
  route, isolated stop, depot-adjacent}. Action space = 56. Requires
  per-operator dispatch to know which seed makes sense.

**Recommend (operator, intensity)** for D2 — minimal API surface change,
maps cleanly to existing operator signatures via `max_seconds` and
optional intensity kwargs we add to ALNS-style operators.

### Policy shape

Small MLP, target ~50K parameters (cheap on CPU per inference):
- Input: 16-dim state + last-3 operator one-hots (3 × 14 = 42 dims) = **58 dims**
- Hidden: [128, 64], ReLU
- Output: action logits over (operator × intensity) = up to 42 outputs
- Sampling: Boltzmann softmax with temperature τ (annealed from 1.0 → 0.1)

### Training algorithm

- **Phase 1: behavior cloning** on bandit logs — the bandit's own
  successful picks form a reasonable expert policy. Loss = cross-entropy
  vs the bandit's chosen action, weighted by the realized reward.
- **Phase 2: REINFORCE / PPO** on simulated rollouts — solve fresh
  instances with the cloned policy, accumulate rewards (with Phase D4
  shaped terms), gradient-update.

Phase 1 is enough for D3 to be testable. Phase 2 is D5.

## D3 — Offline training scaffolding (next iteration)

Files to add (planned):
- `svrptw/solvers/learning/logging_bandit.py` — wraps LinUCBBandit; dumps
  (state, action, reward, plateaus, ops_applied) JSONL.
- `svrptw/solvers/learning/policy_mlp.py` — the MLP, plus a
  `MLPBanditPolicy` class with `choose(context, eps)` and `update(...)`
  methods so it drops into pm.solve unchanged.
- `bench/scripts/collect_bandit_logs.py` — runs solve_auto over a
  diverse instance set with the LoggingBandit; dumps transitions.
- `bench/scripts/train_policy_offline.py` — loads transitions, fits the
  MLP via behavior cloning.

## D4 — Reward shaping (per Human Reflection 10)

The user named three signals to reward:

| signal | proxy from solution | reward term |
|--------|---------------------|-------------|
| dispose of islands | count of (routes with N <= 2 customers) | + ·(prev - new) |
| remove isolated stops | count of customers in solo routes | + ·(prev - new) |
| shrink legs | sum of edge lengths above the route's median edge | + ·(prev - new) |

These compose with the existing improvement-per-second cost-reward.
Coefficients tuned via short hyperparameter sweep on a small instance set.

The shaped reward is computed in pm.solve right after `improvement = cost_before - cost_after`, added to the bandit's reward signal, and recorded in the LoggingBandit's transition.

## D5 — A/B integration

Add to `pm.solve`:
```python
def solve(..., bandit_kind: str = "linucb", policy_artifact: str | None = None):
    if bandit_kind == "linucb":
        bandit = LinUCBBandit(...)
    elif bandit_kind == "mlp":
        from svrptw.solvers.learning.policy_mlp import MLPBanditPolicy
        bandit = MLPBanditPolicy.load(policy_artifact, ops=list(ops_pool))
    ...
```

`solve_auto` exposes a `rl_neighborhood: bool = False` flag that maps to `bandit_kind="mlp"`.

A/B bench: same instance set, paired seeds, one run per kind, paired-delta summary. Lives at `bench/scripts/rl_vs_linucb_ab.py`.

## Cross-references
- [[01 - Progress Report]] — Phase D in the epic
- [[02 - Operators]] — pool of 14 + extra_arms
- [[10  - Human Reflection]] — item 3 ("stochastic trained model" + reward terms)
- `svrptw/solvers/learning/state_features.py` — input vector
- `svrptw/solvers/learning/bandit.py` — the LinUCB to replace
- `svrptw/solvers/classical/portfolio.py` — the call site to swap into
