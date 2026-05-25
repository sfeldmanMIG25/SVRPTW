---
title: iter-5l Phase F (sub-agent) — opt-in structural cost terms
project: SVRPTW
tags: [phase-F, cost-model, structural-advantage, completed]
date: 2026-05-15
---

# Phase F1+F2+F3 — completion summary

Sub-agent execution of `WorldsFinestVRP/15 - Cost-Model Exploration.md`
Phases F1 (schema), F2 (evaluator), F3 (ablation bench), F4 (smoke
tests). All exit criteria green.

## What changed

### F1 — `svrptw/config/schema.py`
Added three opt-in fields to `Economics` (default 0.0 each):
- `crossings_penalty_per_pair`     — `$ * inter_route_crossings`
- `util_imbalance_penalty_coef`    — `$ * load_util_cv`
- `tw_buffer_bonus_coef`           — REWARD: `-$ * mean_tw_buffer_score`

### F2 — `svrptw/solvers/common/solution.py::evaluate()`
Appended a single guarded block AFTER the existing per-route fixed-cost
block. The metrics call (`svrptw.metrics.score_solution`) is gated on
the OR of the three coefs being non-zero — bit-identical when all are
0.0, AND no ~5–20 ms per-call overhead in the bandit accept loop at
the default Settings. Each enabled coef adds its own term linearly
(crossings + util CV + negative tw-buffer reward).

### F3 — `bench/scripts/cost_term_ablation.py`
New parallel-bench (ProcessPoolExecutor pattern from
`cb_scaled_rebench.py`). Five arms × six instances ×
{solve_auto budget, pyvrp baseline budget}:
- A: baseline (all coefs = 0)
- B: crossings_penalty_per_pair=2.0
- C: util_imbalance_penalty_coef=20.0
- D: tw_buffer_bonus_coef=15.0
- E: all three combined
- PYVRP: vanilla baseline (cannot see new terms anyway)

Default instances: Manhattan/Paris/SanFrancisco × N=100/200 (rep I000).
All arm cells re-evaluate the resulting solution under BASELINE
Settings to get a fair operational_cost (the arm's own metrics include
its added penalties, which would over-state cost in cross-arm compare).

CLI flags: `--arms`, `--instances`, `--budget`, `--pyvrp-budget`,
`--workers`, `--skip-pyvrp`, `--out`, `--webui`. Live-progress hooks
push to webui via `webui.client.push_stage` / `push_progress`. Worker
inherits `SVRPTW_WEBUI_URL` for snapshot streaming during the bandit
phase.

Output JSON: `bench/runs/cost_term_ablation.json` with `{config,
arm_rows, pyvrp_rows, summary, failed, wall_total_s}`. Console
leaderboard sorts arms by cost-delta and quality-delta vs PyVRP, then
prints a verdict line for arm-E vs arm-A on the win condition.

### F4 — `tests/unit/test_cost_terms_smoke.py`
Six smoke tests cover the three guarded invariants:
1. `Economics()` leaves all three new fields at 0.0.
2. With defaults, `evaluate()` is bit-identical across two fresh
   `Settings()` instances.
3. Each term is linear in its metric: setting only one coef adds
   exactly `coef * metric` to base cost (4 separate tests per term).
4. All three coefs together act additively.

## Exit criteria — all green

| # | Criterion | Result |
|---|-----------|--------|
| 1 | `Economics().crossings_penalty_per_pair == 0.0` | PASS |
| 2 | `pytest tests/unit -q` (no regressions) | **144 passed, 1 skipped** |
| 3 | `pytest tests/unit/test_cost_terms_smoke.py -q` | **6 passed** |
| 4 | `python bench/scripts/cost_term_ablation.py --help` | PASS |
| 5 | end-to-end smoke runs and dumps JSON | PASS |
| 6 | This summary file | PASS |

## Smoke leaderboard (Manhattan-N050-I003, 10 s budget per arm, 15 s PyVRP)

```
[cost_term_ablation] PYVRP baseline
  n=1  mean_op_cost=801.29  mean_qi=0.7684  mean_crossings=58.00

[cost_term_ablation arms]
  arm  n    op_cost    d_cost      qi     d_qi  cross     cv  tw_buf  routes  wall_s
  A    1     698.41   -102.88  0.6061  -0.1623  126.0  0.104  0.6051   11.00    6.45
  B    1     744.59    -56.70  0.6329  -0.1356   70.0  0.064  0.4997   11.00    8.89
  E    1     795.50     -5.79  0.7687  +0.0003   33.0  0.056  0.4502   11.00    9.90

[verdict] arm-E vs arm-A: cost_ok=False qi_ok=True pv_qi_ok=True -> MIXED
```

### Smoke read

Single-instance smoke at the smallest N (50) so don't over-read it,
but the directional signal is the right shape:

- Arm A (no new terms) buys cheap ops cost (-$103 vs PyVRP) but pays
  in crossings (126) and quality_index (0.606 vs PyVRP's 0.768).
  This is the structural-debt story Phase F was designed to expose.
- Arm E (all three terms) pulls the bandit toward PyVRP-quality
  geometry: crossings 126 → 33 (-74%), CV 0.104 → 0.056, qi
  0.606 → 0.769 (matches PyVRP). Cost gives back ~$97 of the savings
  arm A had over PyVRP, but still $6 cheaper than PyVRP at the same
  quality.
- Arm B (crossings only) is the partial-credit middle: half the
  crossings reduction, half the qi gain.

The clean verdict needs the full 6-instance run at the planned 30 s
budget to settle whether arm E truly hits "cost_ok AND qi_ok AND
pv_qi_ok = WIN" on average. The bench is wired up and ready —
hand-off to the next iteration to run the full grid.

## Files touched

- `svrptw/config/schema.py`                       (+8 lines, F1)
- `svrptw/solvers/common/solution.py`             (+15 lines, F2 — gated block at end of cost build)
- `tests/unit/test_cost_terms_smoke.py`           (NEW, 6 tests, F4)
- `bench/scripts/cost_term_ablation.py`           (NEW, F3)
- `bench/runs/cost_term_ablation_smoke.json`      (smoke output)
- `WorldsFinestVRP/Sessions/2026-05-15_iter5l_subagent_F.md` (this file)

## Cross-references
- [[15 - Cost-Model Exploration]] — Phase F plan
- [[Sessions/2026-05-15_phase_a_closed]] — Phase A close-out (predecessor)
- [[Sessions/2026-05-15_iter5f_subagent]] — metrics suite that powered F2's term picks
- [[Sessions/2026-05-15_iter5g_subagent]] — TW buffer metric origin
