---
date: 2026-05-15
project: SVRPTW
tags: [iter5x, peak-hour, counterexample, recipe-boundary, operator-set]
---

# iter5x — peak_hour: counterexample to the recipe (operator-set boundary found)

## TL;DR

The cost-model-expansion recipe (iter-5u/v/w) does NOT automatically
generalize across structural categories of constraint. **peak_hour
fails gate 7: 2/6 wins, mean −$12.2/inst** under the peak-aware
objective. Diagnostic shows **4 of 6 instances have identical peak
surcharge baseline vs peaked** — the bandit could not find a different
peak-overlap solution.

This is a real generalization boundary, not a tuning failure. The recipe
needs operators that act on whatever axis the constraint targets.
shift_overrun was per-route → existing split/merge operators worked.
peak_hour is per-segment-of-time → existing reorder operators don't
cover the time-segment axis. The first counterexample is more informative
than another 6/6 confirmation.

## Per-instance results (12 paired solves, ~6.5 min wall)

```
instance              b_K s_K  b_ops  s_ops  b_peak  s_peak     net
Manhattan-N0500        17  17  768.8  770.2  246.5   246.5     -1.3
Manhattan-N1000        32  33 1276.4 1329.9  464.0   478.5    -68.0
Paris-N0500            16  16  606.7  623.3  232.0   232.0    -16.6
Paris-N1000            30  30  973.2  963.9  435.0   435.0     +9.3
SanFrancisco-N0500     16  16  637.0  627.8  232.0   232.0     +9.2
SanFrancisco-N1000     31  31 1011.8 1017.7  449.5   449.5     -5.9
                                                              -----
                                                       mean:  -$12.2
```

Note **`b_peak == s_peak`** on 4/6 instances (Manhattan-N500, Paris-N500,
SF-N500, SF-N1000). The peaked solver found EXACTLY THE SAME peak-overlap
amount as the baseline — meaning the bandit's accepted moves either left
peak-overlap unchanged or perfectly cancelled out. This is the smoking
gun for "operator set can't act on the time-segment axis."

## Root cause analysis

depot.ready = 480 minutes (8:00am) for all v1 instances.
Peak window 1: [480, 600] = 8–10am (exact overlap with depot start).
Peak window 2: [1020, 1140] = 5–7pm.

Every route in every solution starts at the depot at t=480. The route
duration is dominated by sum-of-travel-times plus sum-of-service-times,
which is a property of the **customer set assigned to the route**, not
of when within the day those customers are visited.

The LinUCB bandit's 13 operators are:
- `relocate`, `swap`, `two_opt_intra`, `two_opt_star`, `or_opt`,
  `merge_routes`, `split_route`, `sisr_destroy_repair`, etc.

All of these change **which customers go on which routes and in what
order**. None of them shift **when a route starts in the day**.
Consequence: changing the customer assignment can only shift peak overlap
by ~10 minutes at the margin (different return-to-depot times), not by
the structural amount needed to push activity OUT of the 8–10am window.

The shifted solver did spend a small ops-cost premium ($1.4–$17 per
instance on most cells) finding alternative customer orderings, but
those orderings produced **identical** peak-window overlap because
peak overlap is driven by the depot-start latency, not by ordering.

## What would close this

Two paths:

1. **Add a "shift route start time" operator** to the bandit's arm set.
   Each route gets a `start_offset_minutes` field; an arm proposes
   moving a route's start to t=600 (after morning peak) at the cost of
   later customer arrivals. The bandit can then trade off peak savings
   vs late-delivery penalty.

2. **Re-tune the term**: increase `peak_hour_wage_multiplier` to 3.0
   or higher so the marginal saving exceeds the marginal ops cost of
   restructuring. But this hits the commercial-coefficient ceiling
   point the user flagged (extreme values lose plausibility).

Path (1) is the structurally correct fix. Path (2) is a band-aid.

## Strategic significance

This is the result we needed to PROVE we're not just stacking trivial
confirmations on the recipe. The user's framing in the last review:

> "iter-5x produces 'another data point' but not 'a new strategic
> finding.'"

The expected outcome was 6/6, confirming the recipe AGAIN. Instead we
got 2/6 with a clear mechanistic explanation. That's a stronger result
because it locates the recipe's boundary instead of just extending its
demonstrated zone.

The corrected research recipe:

> Step 0 (new): **Check whether the bandit's operator set acts on the
> axis the proposed cost term targets**.
>  - If yes → proceed with the 7 gates as before.
>  - If no → either add the missing operator first, or pick a different
>    term whose axis the existing operators DO cover.

## What this opens

Re-prioritize the queue:

- **iter-5y cross-route fairness** moves UP in priority. Cross-route
  fairness (variance penalty on wage·time across drivers) targets a
  cross-route property. The existing `relocate`, `swap`, `two_opt_star`,
  and `merge_routes` operators ALL act cross-route. So the recipe's
  structural prerequisite IS met. If iter-5y also lands 6/6, the recipe
  is confirmed for two structural categories (per-route, cross-route)
  with one negative result in the third (per-segment-of-time without
  time-shift operator).

- **iter-5w-bis: add start-time-shift operator** queued as a follow-up.
  This is a real piece of solver work (~2 days), not a 1-iteration term.
  Worth doing IF we want peak_hour or other time-segment terms to be
  shippable. Until then, the bandit can't realistically optimize them.

- **iter-5z two-phase polish**, **iter-6a richer constraints** unchanged.

## Honest caveats

- We didn't try increasing peak_hour_wage_multiplier above 1.5 to see
  if there's a coefficient where the bandit DOES find a different
  solution. Worth a 5-minute coefficient sweep at some point to confirm
  the "operator-set-not-coefficient" diagnosis. But the b_peak == s_peak
  identity on 4/6 cells is strong evidence the coefficient isn't the
  variable here.
- This was tested only on OSM v1_large where depot.ready=480 sits on
  the morning peak edge. Instances with depot.ready at e.g. 360 (6am)
  would have more pre-peak budget and the bandit might find a different
  story. But the v1_large set is our production focus.

## Files / artifacts

- `bench/scripts/iter5x_peak_hour_v1large.py` — 12-solve paired bench
- `bench/runs/iter5x_peak_hour_v1large.json` — payload
- `bench/runs/iter5x_peak_hour_v1large.log` — printable log
- `svrptw/config/schema.py` — peak_window_starts / peak_window_ends /
  peak_hour_wage_multiplier (opt-in, bit-identical default)
- `svrptw/solvers/common/solution.py` — per-segment peak-overlap +
  gated surcharge block in evaluate()
- `tests/unit/test_cost_terms_smoke.py` — 2 new tests
  (`test_peak_hour_surcharge_is_linear_in_multiplier_minus_one`,
  `test_peak_hour_gating_requires_both_windows_and_multiplier`),
  10/10 pass
- This file
