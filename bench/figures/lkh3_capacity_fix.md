# LKH-3 wrapper capacity fix (SPEC-0-EVAL-01 follow-up)

Date: 2026-05-13

## Symptom

With `RUNS=1` + `TIME_LIMIT=1 s`, LKH-3 CVRPTW was producing
capacity-overloaded routes the evaluator's pre-fix `feasible` flag
silently accepted:

| instance | n_routes | mean util | max util |
|---|---:|---:|---:|
| OSM-Manhattan-N050-I000 (before) | 7 | 1.502 | 1.66 |

## Fix

`svrptw/solvers/classical/lkh3.py` — two small `.par` changes:

```diff
-    "RUNS = 1",
+    "RUNS = 5",
     f"TIME_LIMIT = {max_seconds}",
     "TRACE_LEVEL = 0",
     "MTSP_OBJECTIVE = MINSUM",
+    "MTSP_MIN_SIZE = 1",
     f"VEHICLES = {inst.num_vehicles}",
```

LKH-3 needs multiple restarts to converge to a capacity-feasible
solution when TIME_LIMIT is small (1–5 s). One restart was forcing
it to return whatever heuristic-quality tour it had — frequently
overloaded. The `MTSP_MIN_SIZE=1` is belt-and-braces (prevents the
degenerate 0-customer-route output we hadn't seen but might).

## After fix

| instance | n_routes | mean util | max util | overload |
|---|---:|---:|---:|---:|
| OSM-Manhattan-N050-I000 | 14 | ~0.71 | 1.00 | 0 |
| OSM-Manhattan-N050-I001 | 15 | ~0.71 | 1.00 | 0 |

Cost rises (950.8 vs the pre-fix 774.3) because LKH is now using
more vehicles to actually fit the demand. That's the real cost.

## Where this leaves LKH-3 in the comparison

LKH-3 @ 5 s budget on Manhattan I000: ~951.
portfolio@10 on the same instance: 722.

LKH-3 is now legitimately **30 % more expensive** than portfolio.
The earlier "LKH wins cost" claim was an artifact of the
SPEC-0-EVAL-01 evaluator bug + the wrapper's too-aggressive RUNS=1.

## Files

| Path | Change |
|---|---|
| `svrptw/solvers/classical/lkh3.py` | RUNS=1→5, +MTSP_MIN_SIZE=1 |
