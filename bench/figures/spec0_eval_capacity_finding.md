# SPEC-0-EVAL-01 — capacity-feasibility bug + fix

Date: 2026-05-13

## TL;DR

`evaluate()` was reporting LKH-3's capacity-violating routes as
feasible. Once we add a hard overload penalty at `hard_late_penalty
× units_overflow`, LKH-3 N=50 Manhattan cost jumps **766 → 30 770**.
Portfolio@10 was already the legitimate leader; the earlier "LKH wins
on N=50 cost" claim was an artifact.

## Per-route utilisation (Manhattan N=50 I000, cap=29)

| solver | n_routes | mean util | max util |
|---|---:|---:|---:|
| greedy | 11 | 0.956 | 1.00 |
| portfolio@5 | 11 | 0.956 | 1.00 |
| **lkh3@1** | 7 | **1.502** | **1.66** |

7 routes carrying ~1.5× capacity each. The accountants would notice.

## Smoke after fix (10 Manhattan N=50 instances, 4 solvers)

```
solver          mean legacy   mean new    mean d   mean veh
portfolio@10          674.6      675.1      +0.5       11.0
auction_gart          678.4      678.9      +0.5       11.0
greedy                806.2      807.5      +1.2       11.0
lkh3@1              30766.1    30769.6      +3.5       10.0   <-- correctly infeasible
```

The `+30 000` jump on LKH-3 is the capacity penalty; the matching
`feasible=0` flag now propagates to bench JSONs.

## What this invalidates

Any prior bench artifact that includes LKH-3 in the cost ranking
without filtering on `capacity_overload == 0` is overstated. The
specific files to re-run when convenient:

- `bench/runs/v1_smoke_*.json` containing `lkh3@1` rows
- `bench/figures/v1_vivrp_gemini_smoke.md` — narrative needs an update
  (the cost/logic divergence story changes; see SPEC-0-EVAL-01)
- Any OR-Tools wrapper that didn't pass capacity through (audit
  pending)

## What this means for the logic axis

The earlier story — "Gemini judges LKH-3 better because LKH-3 uses
fewer routes; we should add an underutil penalty so portfolio looks
more like LKH" — was upside down. LKH's "cleaner" map was a route
sheet a dispatcher would refuse to ship because the trucks
physically can't hold what's listed. The logic axis (SPEC-6-LOGIC-01)
still has value, but as a *consistency check on visually-plausible
plans*, not as a target to reshape the cost model toward.

## Next steps

1. Audit `svrptw/solvers/classical/lkh3.py` — is `-C` (capacity
   constraint) set in the LKH parameter file?
2. Add `tests/unit/test_evaluator_capacity.py` — direct regression
   that a hand-built overloaded solution returns `feasible=0` and
   the expected penalty magnitude.
3. Re-run v1 + PyVRP SOTA bench *next session* (jobs died overnight)
   with the corrected evaluator.
4. Update `bench/leaderboard.py` to filter `capacity_overload > 0`
   from the ranking by default.
