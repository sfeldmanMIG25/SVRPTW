---
title: Construction Bypass — eliminate the PyVRP convergence dependency
project: SVRPTW
tags: [construction, research-direction, phase-E]
updated: 2026-05-15
---

# Phase E — Construction Bypass

## Hypothesis (user, 2026-05-15)

> "I am convinced we can still optimize construction to bypass the need for PyVRP convergence."

The architectural win this epic is `portfolio_pyvrp_warm` = PyVRP@8s construction → LinUCB bandit refinement. PyVRP's HGS-quality construction lands the bandit in a strong basin at large N where `auction_gart`'s 0.5s construction is convergence-stuck.

**Phase E asks:** can we build a Python-native construction that lands the bandit in a comparable basin without paying the 8-30s PyVRP cost? If yes, we get back 8-30s of bandit budget per solve AND eliminate the only third-party convergence dependency.

## Why this matters strategically

1. **Budget reclamation.** At 30s total budget, PyVRP eats 8s. Reclaiming that means the bandit has +36% more time for refinement. At 60s it's +13%; at 15s it's +53% (and currently the cap-on-PyVRP at 30% means PyVRP eats 4.5s — still a meaningful chunk).

2. **N=400+ scaling.** `scaled_cb(N)` already linear-scales PyVRP budget with N. A fast Python construction whose cost is closer to constant-time-to-10% relative gap would let the bandit dominate at large N.

3. **Drop PyVRP as a hard dep.** PyVRP is mature but heavy. Removing it as a *required* component (not as an option) shrinks the install surface and makes the solver embeddable.

4. **Composition layer is the win.** Per the pasted reflection earlier this epic: composition is where the architectural win lives. A fast construction is a NEW composition primitive, not a new operator.

## What we know about why PyVRP wins at construction

- Hybrid Genetic Search (HGS): population-based, periodic local search inside the GA. Diversifies starts in ways pure-greedy doesn't.
- Excellent TW-aware insertion. Solomon C-class hits often within 0.01% of optimum.
- Tight basin: PyVRP@8s typically within 1-3% of PyVRP@60s on the v1 OOD set.

What we have NOT tried:
- A multi-start regret-3 with diverse seeding rules.
- GART-biased Clarke-Wright savings.
- A hybrid: K parallel greedy starts → regret cleanup → quick LS.
- Replaying a frozen PyVRP basin per instance (cache + replay).

## Concrete plan

### Stage E1 — Baseline characterization (cheap, ~1 day bench)

Bench the *construction-only* output (operational_cost, n_routes, wall_time, missed) on a stratified instance set (16-32 instances spanning v1 N=100/200/500 + Solomon + Homberger N=200):

| construction | how to run |
|--------------|------------|
| greedy NN | `svrptw.solvers.classical.greedy.solve` |
| regret_k k=1 | `svrptw.solvers.classical.regret_k.solve(k=1)` |
| regret_k k=3 | k=3 |
| regret_k k=5 | k=5 |
| auction_gart @ 0.5s | `auction_gart.solve(budget=0.5)` |
| auction_gart @ 2s | `auction_gart.solve(budget=2.0)` |
| auction_gart @ 8s | budget=8.0 (matches PyVRP@8 budget) |
| **PyVRP@1s** | `pyvrp_solver.solve(budget=1.0)` |
| **PyVRP@2s** | budget=2.0 |
| PyVRP@4s | budget=4.0 |
| **PyVRP@8s** | budget=8.0 (current default) |
| PyVRP@16s | budget=16.0 |

Output: a leaderboard of (construction, mean_cost, mean_routes, mean_wall) per N. Plus a "basin quality" metric: `cost / cost_at_PyVRP@30s`. The PyVRP@1/2/4s rows tell us how much the 8s budget is buying — if 2s gets within 5% of 8s, our fast construction only needs to beat PyVRP@2s.

### Stage E2 — Fast composition prototype

Build `svrptw/solvers/classical/fast_construct.py`:

```python
def solve(inst, settings, budget_seconds: float = 2.0,
          n_starts: int = 4, ls_passes: int = 1, seed: int = 0) -> Solution:
    """Multi-start fast construction: parallel-seeded regret + light LS.

    1. Generate `n_starts` distinct greedy starts using diverse seeding
       (nearest-neighbor, savings, polar-angle sweep, GART-biased).
    2. For each start: regret-3 reinsertion of unrouted (~0.2s).
    3. For each start: 1 round of swap_star + 2-opt_intra (~0.3s).
    4. Return best by operational_cost.

    Target wall: <= budget_seconds. Default 2s gives 4 starts × 0.5s.
    """
```

Then `solve_auto_v2` flag in `portfolio_pyvrp_warm.py`:
- `construction: Literal["pyvrp", "fast_compose"] = "pyvrp"`
- When `"fast_compose"`: replace `pv.solve(...)` with `fast_construct.solve(...)`.
- Bandit phase unchanged.

### Stage E3 — A/B bench

Paired-seed A/B on the v1 OOD set + Homberger N=200/400:
- arm A: `solve_auto(construction="pyvrp", budget_seconds=B)`
- arm B: `solve_auto(construction="fast_compose", budget_seconds=B)`

Same total wall, fast_compose gives the bandit ~6s more budget.
Win criterion: arm B mean cost <= arm A mean cost on a 24+ instance subset, OR within $5/instance with strictly better wall.

If positive: ship as default for N>=200. If neutral: ship as opt-in. If negative: keep PyVRP and characterize *why* (the basin quality metric from Stage E1 will tell us).

### Stage E4 (stretch) — Drop PyVRP entirely

If E3 wins cleanly, mark PyVRP as opt-in only. The default path becomes pure-Python: `fast_construct → bandit`. This is the publishable headline.

## Cross-cutting: parallel RL pipeline

Stage E runs in parallel with the RL pipeline (Phase D3-D5 already implemented):
- `collect_bandit_logs.py` over a diverse set produces a transition corpus
- `train_policy_offline.py` fits the MLP on those transitions
- `rl_vs_linucb_ab.py` compares

Both can run simultaneously since they touch different layers (construction vs bandit policy). If both win, they compose: `fast_construct → MLP-policy bandit → shape_reward`.

## Acceptance + decision points

- **End of E1:** publish the baseline table. Decide whether the gap PyVRP exploits is small enough that fast_compose can plausibly close it.
- **End of E2:** smoke that fast_compose solves don't crash and produce feasible solutions on the full v1 OOD set.
- **End of E3:** publish the A/B numbers. Decide ship-as-default / ship-as-opt-in / shelve.

## Cross-references
- [[01 - Progress Report]] — epic snapshot
- [[02 - Operators]] — bandit pool (unchanged by Phase E)
- [[03 - Benchmarks]] — current PyVRP-warm baseline numbers
- [[10  - Human Reflection]] — Phase E aligns with item 5 ("performance writing built into the harness")
- [[11 - RL Roadmap]] — runs in parallel with Phase E
- [[12 - Architecture Review]] — current state of the build

## Files to add/edit (planned)
- `bench/scripts/construction_baseline.py` — Stage E1
- `svrptw/solvers/classical/fast_construct.py` — Stage E2
- `svrptw/solvers/classical/portfolio_pyvrp_warm.py` — add `construction` kwarg
- `bench/scripts/fast_construct_ab.py` — Stage E3
