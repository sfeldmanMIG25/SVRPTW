# SPEC-0-BENCH-01 — Baseline lock on v1

```
ID:            SPEC-0-BENCH-01
Title:         Frozen baseline metrics for all comparison solvers on instances/v1
Owner role:    Bench Engineer
Status:        FROZEN
Inputs:        instances/v1/ (160 instances, asymmetric road-network), all solvers
Outputs:       bench/baselines/v1.json + bench/baselines/v1.csv
```

## Behavior

Run every comparison solver on every v1 instance under the deterministic benchmark protocol (§10 of the SDD plan, amended per ADR-0001 for non-stochastic default). Lock the results to `bench/baselines/v1.json`. This file becomes the immutable reference against which every future PR is regression-checked.

### Solvers included in v1 baseline

| Solver | Variant | Budget points |
|---|---|---|
| Greedy NN | asymmetric | n/a (deterministic) |
| OR-Tools CP-SAT | first-solution + GUIDED_LOCAL_SEARCH | 1 s, 10 s, 60 s |
| LKH-3 | VRPTW formulation | 1 s, 10 s, 60 s |
| DQN (current) | trained on v1 train split | inference only |
| Bayesian Auction (current) | best_solver_params_2.json | n/a |
| MCTS hybrid (current) | 1000-iteration budget | n/a |

Each solver, each instance, each budget point → one row.

### Metrics per row

```jsonc
{
  "solver": "ortools",
  "budget_seconds": 10.0,
  "instance_id": "OSM-Austin-N100-I003",
  "operational_cost": 1812.44,
  "total_distance_miles": 137.21,
  "total_time_minutes": 622.0,
  "num_vehicles_used": 23,
  "missed_deliveries": 0,
  "feasible": true,
  "wall_clock_seconds": 9.987,
  "solver_git_sha": "abc1234",
  "hardware": {"cpu": "i9-...", "gpu": "RTX 3070 Ti Laptop", "passmark_st": 3400},
  "seed": 271828
}
```

Operational cost = wage·time + transit·distance + hard_late_penalty · missed (per `Economics` block in `Settings`).

## Invariants

- v1.json is **frozen**. Future PRs may not modify it. New baselines write to v2.json with a new manifest.
- Per-instance seed = `hash(instance_id) mod 2^31`, fixed.
- Stochastic simulation is OFF by default (ADR-0001 D4). A separate `bench/baselines/v1_stochastic.json` is optional, generated on-demand, not gated.
- Hardware metadata recorded for every row so future re-runs are comparable.

## Acceptance

- `python -m svrptw.bench run --solvers all --instances v1 --out bench/baselines/v1.json` finishes and writes the file.
- `python -m svrptw.bench report --baseline v1` prints a markdown summary per solver per N-bin.
- File is regenerable end-to-end from a clean clone in < 8 hours on the workstation.
- CI's smoke regression check (`SPEC-0-CI-01`) reads from this file.

## Non-goals

- Beating any solver. This is measurement, not improvement.
- Tuning hyperparameters. Use the current defaults / committed Optuna outputs.

## Dependencies

- SPEC-0-CFG-01 (Settings)
- SPEC-0-INST-01 (instances/v1)
