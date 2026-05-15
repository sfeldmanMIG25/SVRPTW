# Operators tried

## Production-bandit native arms (13)

These ship in `svrptw/solvers/classical/portfolio.py::_OPS` and are the LinUCB arms the bandit chooses among at every step. All proven effective inside the bandit; none of them tested in isolation against PyVRP.

| Arm | Type | Reference |
|---|---|---|
| relocate | Single-customer move | classical |
| swap_star | Pairwise customer swap with adaptive neighborhood | Vidal et al. |
| two_opt_intra | 2-opt within a route | classical |
| two_opt_star | 2-opt across two routes (Potvin & Rousseau 1995) | classical |
| three_opt_intra | 3-opt within a route | classical |
| merge_routes | Combine two routes | classical |
| sisr | Slack-induced string-removal then regret-k reinsertion (Christiaens & Vanden Berghe 2020) | ALNS |
| ejection_chain | Multi-hop relocate | Glover |
| cyclic_3 | 3-route cyclic exchange | classical |
| vehicle_kill | Greedy-best route-deletion + reinsert | classical |
| soft_drop | Probabilistic drop-route operator | classical |
| drop_route | ε-greedy worst-util route removal + regret-1 reinsertion | SPEC-7-OPS-DESTROY-01 |
| destroy_island | Voronoi-clustered destroy + regret reinsertion | SPEC-7-OPS-DESTROY-01 |
| drop_leg | Worst-load-util edge removal + reinsert | SPEC-7-OPS-DESTROY-01 |

## Operators tested *outside* the bandit (as council seeds)

Tested via SPEC-8-COUNCIL-02 composability bench (paired-seed, 16-instance subset). All were **rejected at production budget** — the bandit's 13-arm pool already covers their value.

| Seed | Result | Why rejected |
|---|---|---|
| `tw_anchor_relocate` (seeds/) | mean +$3.68, hit 38% | Below 40% hit-rate floor — PyVRP's relocate covers this |
| `capacity_rebalance_swap` | not benched standalone | (covered by drop_route + reinsert) |
| `cross_route_2opt` (seeds/) | mean **-$2.14**, 0% hit | Strictly redundant with PyVRP's internal 2-opt* |
| `gart_guided_ruin` (seeds/) | mean +$8.34 (small subset bias), 38% hit | Doesn't beat noise floor under paired seeding |
| `fragment_relocate` (seeds/) | 0/5 hits on portfolio@10 | PyVRP's OR-opt already moves contiguous fragments |
| LLM-generated: spatial centroid swap (×3) | +$3-4 mean (8-inst bench) → +$0.13-0.49 on 16-inst | Selection bias on small bench; not real signal |
| LLM-generated: route-consolidation-squeeze | +$4.15 (small) → +$0.59 (wider) | Same selection-bias pattern |
| LLM-generated: spatial-temporal segment exchange | +$4.28 → +$0.53 | Same |

**Standing finding**: at production budget (portfolio@10s), the bandit's pool is operator-saturated. No simple-shot operator clears the paired-seed composability bar on a 16-instance bench. The LLM proposer ran 25 candidates total; **0 passed wider validation**.

## Architectural composition (DID work)

The single change that produced real value this session:

**`portfolio_pyvrp_warm`** = PyVRP@8s construction → portfolio LinUCB bandit @ remaining budget.

This is **not a new operator** — it's swapping the warmstart source. Vanilla portfolio warms with `auction_gart`'s construction; the variant warms with PyVRP's. At N≥100, this single swap flips a +$78 loss vs PyVRP@60s into a +$110 win. Tuned with `pyvrp_construction_budget=8` and `plateaus_to_stop=20` (default raised from 6 because the bandit refining a PyVRP start plateaus more easily).

## Variants tried (negative or partial results)

| Variant | Result |
|---|---|
| **POMO+EAS-Emb** (Hottung 2022) | Works (2.2% gain in 19s on greedy start) but base POMO 12% behind portfolio. EAS won't change leaderboard. |
| **EAS-as-bandit-arm** | Negative — EAS iter too coarse-grained for the bandit's per-op budget. |
| **Fusion solver** (POMO K-start × bandit fine-tune) | Negative — POMO warmstart structurally weaker than auction_gart; budget-split hurts. |
| **LKH-3 as warmstart** | Negative — $191 worse than auction, 12× slower. LKH-3's CVRPTW handling produces TW-violating tours that get penalized. |
| **Logic-axis (committee VLM grading)** | v0 LogicEnsemble separated greedy-vs-strong only (Δ 0.05); v1 with 30 strong-vs-strong labels regressed. Closed off under single-judge labelling. |
| **per_route_fixed_cost** | Works as a non-PyVRP-cost-extension lever, but on capacity-saturated v1 instances route count is pinned (11 forced by capacity). v2 (cap_buffer 2.5) unlocks it: portfolio drops 6.25 → 6.00 routes; PyVRP stays at 7.75. |
