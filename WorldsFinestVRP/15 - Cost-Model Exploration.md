---
title: Cost-model exploration — next research direction (iter-5l+)
project: SVRPTW
tags: [cost-model, phase-F, structural-advantage]
updated: 2026-05-15
---

# Phase F — Cost-model exploration

## Why this direction

Per the user's iter-0 reflection: "the structural advantage compounds with each
[new cost term], because PyVRP can't see them." `per_route_fixed_cost` is the
existing proof of concept. The iter-5f metrics suite operationalized 14 more
candidate terms — turning the best 2-3 into opt-in cost fields gives us
structurally-multiplicative wins.

## Candidate cost terms (from metrics suite)

| metric | direction | why PyVRP can't optimize it |
|---|---|---|
| `inter_route_crossings` | minimize | not in cost model; pure structural |
| `mean_detour_ratio` | minimize | hidden inside route cost; PyVRP optimizes wall length |
| `load_util_cv` | minimize | balance, not absolute |
| `convex_hull_overlap_count` | minimize | spatial cluster separation |
| `mean_tw_buffer_score` | maximize | rewards arrival-near-open, not just feasibility |
| `silhouette_like` | maximize | cluster separation quality |
| `late_frac` | strict 0 | already covered by hard penalty |
| `total_wait_min` | minimize | partially covered by wage·time |

## Top-3 picks for iter-5l prototype

1. **`crossings_penalty_per_pair`** ($/crossing): every inter-route segment crossing
   adds a fixed dollar penalty. Direct attack on the spaghetti problem the
   construction-bypass grid surfaced.
2. **`util_imbalance_penalty_coef`** ($·CV): weighted load_util_cv across routes.
   Penalizes "one route at 100%, three at 30%" arrangements that minimize total
   distance but stress one driver.
3. **`tw_buffer_bonus_coef`** ($·avg_buffer_score): pays the solver to arrive
   near TW-open without waiting AND with margin before TW-close.

Each is gated (default 0.0 = bit-identical legacy behavior).

## Implementation plan

### Stage F1 — Add three new fields to `svrptw.config.schema.Economics`
```python
# SPEC-F-COST-01 -- opt-in structural cost terms.
crossings_penalty_per_pair: float = 0.0       # $ per inter-route crossing
util_imbalance_penalty_coef: float = 0.0      # $ * CV(route_utils)
tw_buffer_bonus_coef: float = 0.0             # -$ * mean_tw_buffer_score (negative = reward)
```

### Stage F2 — Compute the terms inside `svrptw.solvers.common.solution.evaluate()`
- Already computes `wage*time + cost*dist + early_wait + missed*hard_late + per_route_fixed_cost`
- Add `+ crossings * crossings_penalty_per_pair` (use metrics.score_solution to get count)
- Add `+ util_imbalance_penalty_coef * load_util_cv`
- Add `- tw_buffer_bonus_coef * mean_tw_buffer_score`

### Stage F3 — A/B bench on Manhattan/Paris N=100/200
- arm A: `solve_auto` with default Settings (zero new terms)
- arm B: `solve_auto` with `crossings_penalty_per_pair=2.0`
- arm C: same with `util_imbalance_penalty_coef=20.0`
- arm D: same with `tw_buffer_bonus_coef=15.0`
- arm E: all three combined
- vs PyVRP (which optimizes only operational_cost without the new terms)

Win condition: arm E (combined) wins more on the unified objective at the SAME
operational cost — i.e. we don't pay more, we just get structurally better
solutions. PyVRP can't see the new terms so its solutions look worse on them.

### Stage F4 — Pick the 1-2 terms that survive the bench, add as defaults

## Files to add (planned)
- `svrptw/config/schema.py` — extend Economics
- `svrptw/solvers/common/solution.py` — extend evaluate()
- `bench/scripts/cost_term_ablation.py` — Stage F3 bench
- `tests/unit/test_cost_terms_smoke.py` — bit-identity at default + correct activation

## Cross-references
- [[01 - Progress Report]]
- [[10  - Human Reflection]] — user's vision
- [[12 - Architecture Review]]
- [[Sessions/2026-05-15_phase_a_closed]] — Phase A done; this is the unblock
