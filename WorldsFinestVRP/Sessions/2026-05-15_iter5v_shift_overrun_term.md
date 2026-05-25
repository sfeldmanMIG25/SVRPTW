---
date: 2026-05-15
project: SVRPTW
tags: [iter5v, cost-model-expansion, shift-overrun, structural-advantage, validated]
---

# iter5v — shift-overrun cost term (cost-model expansion, first new dimension)

## TL;DR

Added an opt-in `shift_max_minutes` + `shift_overrun_penalty_per_min` cost
term to `Economics`. Bench on Manhattan-N500 (cap=300 min, $1/min penalty,
2×75 s solves) shows **solve_auto's bandit DID optimize for the new term**:

```
solver        K    ops_cost    overrun_$    cost_w_shift
baseline     17     $765.8      $2039.1      $2804.9
shifted      18     $826.2      $1869.1      $2695.3   <- net $109.6 better
```

The shifted solver added 1 route (K 17 → 18) to spread the load, paying
$60 extra ops cost to reduce overrun by $170. **Net $109.6/inst saving
under the shift-aware objective.** PyVRP at default settings cannot see
this term; solve_auto's LinUCB bandit can.

This is the first concrete validation of the iter-5u strategy:
each new opt-in cost term that solve_auto can optimize and PyVRP cannot
is a new dimension of structural advantage.

## Implementation

### Schema (`svrptw/config/schema.py`)

```python
# iter-5v -- driver-shift / per-route duration overrun. PyVRP can't see this
# (no per-route duration constraint in the cost model); solve_auto's bandit
# can optimise for it. Gated on BOTH being non-zero -> bit-identical default.
shift_max_minutes: float = 0.0
shift_overrun_penalty_per_min: float = 0.0
```

### Evaluator (`svrptw/solvers/common/solution.py`)

Per-route duration is now tracked in the stitching loop (cheap list append,
~50 routes max), then consumed by a gated cost block:

```python
if (e.shift_max_minutes > 0.0
        and e.shift_overrun_penalty_per_min > 0.0
        and route_durations):
    cap_min = float(e.shift_max_minutes)
    coef = float(e.shift_overrun_penalty_per_min)
    for dur in route_durations:
        over = dur - cap_min
        if over > 0:
            cost += over * coef
```

Bit-identical when either coef is 0 (default).

### Tests (`tests/unit/test_cost_terms_smoke.py`)

Two new tests (6 → 8 total):
- `test_shift_overrun_term_is_linear_in_overrun_minutes` — verifies
  doubling the penalty doubles the cost-delta exactly.
- `test_shift_overrun_gating_requires_both_coefs` — verifies setting
  only the cap (no penalty) or only the penalty (no cap) leaves cost
  bit-identical to defaults.

All 8 pass.

## Smoke result (linearity verification on N=100)

```
cost @ shift_max=120 / penalty=$0.5/min:     $4024.20  (delta=+$2342.98)
cost @ shift_max=120 / penalty=$1.0/min:     $6367.18  (delta=+$4685.96)
linearity check |2*delta_half - delta_full| = 0.0000  (OK)
```

Total overrun minutes detected: 4686 across 24 routes. Cap=120 min was
tight enough to force violations on a small instance.

## Bench result (Manhattan-N500, the real test)

Cross-evaluation pattern (each solution scored under BOTH cost models):

```
solver     wall    K   cost_no_shift   cost_w_shift   overrun_$
baseline  75.1s   17           765.8         2804.9      2039.1
shifted   75.1s   18           826.2         2695.3      1869.1
```

Net under the shift-aware objective:
- baseline: $2,804.9
- shifted:  $2,695.3
- **shifted saves $109.6/inst** at the cost of paying +$60.4 in operational
  cost (1 extra route) to reduce shift overrun by $170.0 (-8.3%).

This is exactly the trade-off PyVRP cannot make at default Settings:
PyVRP optimizes pure operational cost, ignoring the per-route duration
constraint. The shifted solve_auto saw the constraint via the gated
penalty and adjusted accordingly.

## Strategic significance

iter-5u argued: "each new opt-in cost term that solve_auto can optimize
and PyVRP cannot is a new dimension of structural advantage that compounds."
iter-5v is the **first concrete validation of that thesis**. The bandit
finds a different solution shape (one more vehicle, redistributed load)
when shift-overrun is in scope.

If this generalises across the other 5 v1_large instances (next iteration),
shift_overrun is shippable as the first production cost-model extension.

## Honest caveats

- Only 1 instance tested (Manhattan-N500). The +$109.6 net saving is
  meaningful but needs the paired 6-instance validation to be a published
  result. iter-5w queued.
- The +8.3% overrun reduction is small in absolute terms. With a tighter
  cap (e.g., 240 min for a strict 4-hour driving block) or higher penalty
  ($2/min), the bandit might find larger gains.
- The bandit's existing 13-arm operator set wasn't designed for shift-aware
  refinement. Adding a "split-long-route" or "swap-from-overrunning-route"
  operator might unlock larger gains; future work.

## Files / artifacts

- `svrptw/config/schema.py` — `Economics.shift_max_minutes`,
  `shift_overrun_penalty_per_min` (both default 0.0)
- `svrptw/solvers/common/solution.py` — `route_durations` tracking +
  gated penalty block
- `tests/unit/test_cost_terms_smoke.py` — 2 new tests (8/8 pass)
- `bench/scripts/iter5v_shift_overrun_smoke.py` — linearity smoke (N=100)
- `bench/scripts/iter5v_shift_overrun_bench.py` — real bench (N=500)
- `bench/runs/iter5v_shift_overrun_bench.{json,log}` — payload + log
- This file
