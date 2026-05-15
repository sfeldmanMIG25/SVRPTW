# SPEC-3-EJECT-01 — Ejection chain operator

```
ID:            SPEC-3-EJECT-01
Title:         Glover-style cyclic ejection chain for VRPTW
Owner role:    OR Engineer
Status:        DRAFT
Inputs:        Instance, Solution, Settings, max_chain_length, max_seconds
Outputs:       svrptw.solvers.common.local_search.ejection_chain
```

## Background

A *k*-ejection chain moves customer c₁ out of its route, inserts a different
customer c₂ in c₁'s old place, ejects c₂'s old occupant, and so on. After
k links the chain closes — either by inserting the last ejected customer
back into the originating route, or into an unused slot. Single-customer
relocate is k=1; 3-opt and Lin-Kernighan are special cases of higher-k
chains.

Why now: many VRPTW solutions are stuck in basins that no single
relocate/swap can escape but a 3- or 4-link chain can. Empirical observation
(Glover 1996, Toth & Vigo 2002 ch.5): ejection chains close 20-40% of the
remaining gap to optimal beyond standard Or-opt local search.

## Behavior

```python
def ejection_chain(inst: Instance, sol: Solution, settings: Settings,
                   max_chain_length: int = 3,
                   max_seconds: float = 5.0) -> Solution:
    """k-cyclic ejection chain.  Try chains of length 2..max_chain_length.
    Accept the first chain whose closed form lowers total operational cost
    and preserves TW feasibility on every involved route.
    """
```

Internally:

1. Pick a candidate source customer c₁ (heuristically: highest
   GART-marginal in its current route — the customer whose removal saves
   the most).
2. Find the best insertion of c₁ into a DIFFERENT route. This creates
   slack in the source route and pressure in the destination.
3. From the destination route's customers, pick c₂ to eject (the one
   with the worst-fit given c₁ now there).
4. Recurse: insert c₂ elsewhere, eject c₃, ... up to depth
   `max_chain_length`.
5. Close the chain: insert the last ejected customer either back into
   the originating route (cyclic close) or accept it as un-routed (only
   if the total reward exceeds `hard_late_penalty`).
6. Accept the chain only if total operational cost strictly drops AND
   every touched route is TW-feasible.

## Acceptance

- On the N=50 smoke set, `ejection_chain(max=3)` applied AFTER our current
  improvement chain finds an additional ≥ 1% cost reduction on at least 50%
  of instances.
- On the N=100 partial bench, `ejection_chain(max=4)` closes ≥ 20% of the
  gap to LKH-3's mean cost (1390 → ≤ 1290).
- A passing chain on a tiny synthetic instance (N=10) is verifiable by hand
  (test in `tests/unit/test_ejection_chain.py`).
- Worst-case time per chain attempt: O(K · max_chain_length · best_insertion_cost).
  With K=20, max=3, best_insertion=O(N) for N=100: ~6000 ops per attempt.
  Budget of 5s allows ~10k attempts.

## Non-goals

- Lin-Kernighan-style edge moves (the classical name for chains on TSP
  edges, not customer relocations). Out of scope; LKH-3 already covers it.
- Population-based ejection (multi-incumbent). Single-incumbent only.

## Dependencies

- SPEC-1-GART-01 (marginal ranking).
- SPEC-2-AUCTION-01 (warm start that the chain improves).
