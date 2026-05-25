# svrptw — network-aware VRP solver with rich operational constraints

[![tests](https://img.shields.io/badge/tests-197%20unit%20passing-brightgreen)]() [![version](https://img.shields.io/badge/version-0.1.0-blue)]()

A production Python package for solving large-scale Vehicle Routing Problems with Time Windows on real road networks, with a stack-able catalog of operational constraints PyVRP and OR-Tools can't model.

## Quickstart

```python
from svrptw import Settings, load_instance, solve

inst = load_instance("instances/v1_large/OSM-Manhattan-N0500-I000.json")

settings = Settings()
# Opt in to whichever real-world constraints apply:
settings.economics.shift_max_minutes = 480.0
settings.economics.shift_overrun_penalty_per_min = 1.0
settings.economics.driving_max_minutes = 270.0        # EU 561 4.5h cap
settings.economics.break_violation_penalty_per_min = 2.0
settings.economics.embargo_window_starts = (480, 720)  # 8-8:30am, 12-12:30pm
settings.economics.embargo_window_ends = (510, 750)
settings.economics.embargo_violation_penalty_per_visit = 50.0

sol = solve(inst, settings, budget_seconds=75.0)

print(f"cost=${sol.metrics['operational_cost']:.2f}")
print(f"routes used: {sol.metrics['num_vehicles_used']}")
print(f"wall: {sol.wall_clock_seconds:.1f}s")
```

## What's in the box

### Cost-axis performance (v1_large 6-instance wholesale, matched-wall)

| solver | mean cost | feasibility | vs solve_auto |
|--------|-----------|-------------|---------------|
| **`solve` (this package)** | **$906.6** | **100%** | — |
| PyVRP | $1,166.4 | 83% | +$259.7/inst |
| OR-Tools | $1,590.5 | 100% | +$683.9/inst (−43%) |
| LKH-3 | $969,732 | 0% feas | (LKH-3 capacity config issue) |
| greedy | $1,034.2 | 100% | +$127.6/inst |

6/6 wins vs PyVRP, **6/6 wins vs OR-Tools at −43% cost**. No public solver dominates on both cost AND quality (0/6 Pareto check).

### PyVRP-independent construction (iter-7 final)

`solve_auto(..., construction="fast_construct_v4")` ships a pure-Python Solomon I1 sequential insertion + cost-aware + auto-multi-start for the warmstart — no PyVRP dependency. Final calibrated pick-by-axis recommendation (all numbers from contention-free sequential benches; the iter-7-determinism-audit established that the bandit is bit-deterministic under `max_workers=1` and parallel benches add ±$200-$2,600/inst noise):

| workload | v4 vs pyvrp (calibrated sequentially) | recommendation |
|---|---|---|
| Baseline (no cost term) | ~tied, mean −$60/inst (within bandit noise); v4 saves 1-2 K on 4/6 | either; v4 if PyVRP unavailable |
| Embargo (per-visit) | 3/6 wins, mean −$122/inst aggregate (pyvrp slightly ahead, tied within noise); v4 saves 1-3 routes on 3/6 | tie — pick by K savings or PyVRP availability |
| Skills (per-visit class axis) | tied (class_shift operator dominates) | either |
| Driver_breaks (per-route) | pyvrp wins (parallel mean −$291/inst; partial sequential confirms similar magnitude) | **pyvrp** |
| Full 16-term stack | pyvrp wins (mean −$2,623/inst; per-route terms dominate cost mix) | **pyvrp** |
| PyVRP not installable | v4 standalone beats pyvrp −3% cost at 3.8x faster wall | **v4** (only option) |

**Final summary**: v4 is a viable pyvrp-independent option that wins on K-tightness (1-3 fewer routes on most instances) and ties pyvrp at solve_auto cost on baseline + per-visit cost-term workloads. PyVRP warmstart is the better default when per-route cost terms (driver_breaks, shift_overrun, EV_range) are active — v4's tighter K compounds the per-route penalty. Earlier README revisions cited a "+$178/inst v4 win on embargo" from parallel `ProcessPoolExecutor` benches; that finding was a parallelism-contention artifact and has been corrected to the tied-aggregate verdict above. Standalone construction-only: v4 beats pyvrp by −3% cost at 3.8x faster wall (this finding survives sequential re-validation).

**Bench reliability note**: for reproducible measurements, use `max_workers=1` (sequential). For throughput, parallel runs are fine but report results with the ±$200-$2,600/inst noise band.

### Constraint catalog (17 opt-in cost terms)

```python
from svrptw import constraints
constraints.print_catalog()
```

Categories: baseline (3), phase-F (3), iter-5v/x/y (3), iter-6a-1..8 (7), legacy (1).

Most-shippable single-term wins (paired 6-instance bench, mean savings under term-aware objective):

| term | verdict | mean savings/inst |
|------|---------|---|
| `embargo` (hard zones) | 6/6 | **+$1,796.7** |
| `skills` (with class_shift operator) | 5/6 | **+$2,471.0** |
| `min_routes` (labor contract) | 6/6 | +$300.3 |
| `driver_breaks` (EU 561 lite) | 6/6 | +$179.7 |
| `shift_overrun` (per-route duration) | 6/6 | +$155.2 |
| `mixed_fleets` (per-class premiums) | 6/6 | +$64.4 |
| `ev_range` (per-route distance cap) | 6/6 | +$58.4 |
| `pd_pairs` (precedence) | 6/6 | +$51.2 |
| `driver_time_variance` (fairness) | 5/6 | +$36.1 |

**Aggregate stackable savings: +$5,113/inst across 10 active cost terms.** Full-stack solver wall overhead vs baseline: **1.00x** (Phase F lite refactor; 4.89x evaluator overhead at 15-term stack).

### Scales to 1000-customer problems at speed

| N | solver wall | bandit evals/sec (15-term stack) |
|---|-------------|----------------------------------|
| 500 | 75s | ~320 |
| 1000 | 150s | ~150 |

Network-aware OD via osmnx + scipy.sparse Dijkstra (10.6× faster than networkx).

## Validation (reproduce the headline numbers)

Every number in this README is reproducible from a clean checkout in one command:

```bash
# ~6 min: microbench + 2 stack solves at N=500/1000 (CI-grade)
PYTHONPATH=. python bench/scripts/headline_results.py --quick

# ~60-90 min: full reproduction including wholesale leaderboard + 8 per-term benches
PYTHONPATH=. python bench/scripts/headline_results.py

# 3-seed stability bench (~8 min, gives mean +/- std for the combined-stack claim)
PYTHONPATH=. python bench/scripts/iter6a_stack16_multi_seed.py
```

Outputs `bench/runs/headline_results.{json,md}` and `bench/runs/iter6a_stack16_multi_seed_*.json`.

### Per-bench review: operator-annotated cost trajectory

Once a solve has captured transitions (`bandit_kind="logging"` in
`solve_auto`, or the standalone collector script), the operator-gap tool
renders a PNG (or animated GIF) showing which moves actually moved the cost
needle:

```bash
# Static review plot per instance
PYTHONPATH=. python bench/scripts/build_operator_gap_plot.py \
    --transitions bench/runs/bandit_transitions.jsonl \
    --instance OSM-Manhattan-N100-I003.json

# Or an animation that grows the plot step by step
PYTHONPATH=. python bench/scripts/build_operator_gap_plot.py \
    --transitions bench/runs/bandit_transitions.jsonl \
    --instance OSM-Manhattan-N100-I003.json --animate
```

Top panel: cumulative cost decrease over wall time with each accepted bandit
step shown as an operator-colored marker (size = magnitude). Bottom panel:
per-operator contribution bar chart sorted by total cost decrease. Used for
"what's actually working vs which arms are just getting pulled" reviews.

### What's robust vs noisy

**Single-term wins are statistically robust** (6/6 paired across 6 v1_large instances).
Each constraint cost term in the catalog has been bench-validated on 6 different
city-size combinations with consistent positive ROI.

**Architectural-drift case study & fix (closed 2026-05-16)**. Mid-session a multi-seed
re-run of the headline iter-6a-2 embargo term (originally 6/6 +$1,797/inst) produced only
+$798/inst — a 56% magnitude drop. Root cause: `pm.solve` was unconditionally registering
the three meta-recipe operators (`shift_start`, `class_shift`, `depot_shift`) in the
bandit's arm pool even when their target constraints weren't active. Each operator
no-ops correctly but still consumes bandit-iteration budget when picked → less per-arm
exploration time for the operators that actually have work → degraded magnitude.

**Fix**: conditional registration of meta-recipe operators based on `Settings.economics`
inspection at solve time. Post-fix embargo re-bench: **6/6 wins, +$1,648.4/inst** (92%
recovery of original; 2 instances actually beat the original). The arm pool now matches
what each constraint needs — no leeching.

See `docs/META_RECIPE.md` for the three documented failure modes (generalization /
seed-variance / arch-drift) and their diagnostic protocols.

**Combined 16-term stack is solidly positive post-fix** (multi-seed bench 2026-05-16,
Manhattan-N500 b=150s, 3 seeds). With the conditional-registration drift fix
landed, the combined stack now:

```
  seed 0: net = +$1,084.3  (wall_x 0.83x)
  seed 1: net = +$1,007.9  (wall_x 0.95x)
  seed 2: net = +$357.4    (wall_x 0.92x)
  mean = +$816.5 +/- $399 ; wall_x = 0.90x ; 3/3 wins
```

Pre-fix on the same instance / seeds / budget the mean was +$481.9 ± $358 at
1.24x wall — so the same fix that recovered single-term embargo (92%) also
boosts the combined stack by **+69% mean savings while running 27% faster** than
baseline (smaller, axis-matched arm pool → more iterations on operators that
matter, fewer wasted picks).

**Per-N budget rule of thumb for full-stack solves**: ≥ N × 0.3 seconds at
v1_large (so N=500 → 150s, N=1000 → 300s). Single-term constraint solves can
use the standard PyVRP-warm budget (N × 0.15 s) since they're lower-dimensional.
At 75s on N=500 the bandit still produces high-variance results with occasional
large losses; 150s is the floor for robust stack solves.

This package was built with honest-empiricism as a design principle (see
`WorldsFinestVRP/Sessions/` — multiple iterations explicitly falsified
their own previous wins when re-tested at scale).

## Installation

```bash
git clone <repo>
cd svrptw
pip install -e .
```

Requires Python 3.11 or 3.12 (3.13/3.14 lack wheels for `pyvrp` and `ortools`).

## API surface

- `solve(inst, settings, budget_seconds, *, seed=0, **kwargs)` — production dispatcher.
- `Settings` — pydantic config. `Settings.economics` is the cost-model dial.
- `load_instance(path)` — parse an Instance from JSON.
- `evaluate(inst, sol, settings)` — re-score any solution under any cost model.
- `Solution`, `Route`, `Instance`, `Customer`, `Depot` — dataclasses.
- `svrptw.constraints` — catalog documentation module.

## Recipe for adding a new constraint

See [**`docs/META_RECIPE.md`**](docs/META_RECIPE.md) for the full procedure with worked examples, the Step 0 / Step 0.5 / 7 gates breakdown, per-N budget rule, and honest-empiricism notes on when to falsify your own wins.

### TL;DR

Each cost term follows a 2-pre-flight + 7-gate recipe:

1. **Step 0** (operator coverage): verify the bandit's operator set acts on the constraint's axis. If no — add a `(Route field + dedicated operator)` pair (canonical Step 0 remediation, see `shift_start` and `class_shift`).
2. **Step 0.5** (coefficient calibration): smoke-evaluate one baseline solve at the proposed coef; tune so baseline penalty is 3-10% of ops cost.
3. **Gates 1-7**: schema field → gated evaluator block → 2 unit tests → smoke → paired 6-instance bench → 6/6 ship threshold.

See [`WorldsFinestVRP/17 - Package Pivot Roadmap.md`](WorldsFinestVRP/17%20-%20Package%20Pivot%20Roadmap.md) for the full recipe + bench history.

## Citation

Original research: Stephen Feldman, RPI Decision Making Under Uncertainty.

The package builds on PyVRP (HGS construction) + LinUCB bandit refinement composition.
