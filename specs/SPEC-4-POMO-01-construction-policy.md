# SPEC-4-POMO-01 — POMO-lite construction policy

```
ID:            SPEC-4-POMO-01
Title:         Neural construction policy for asymmetric VRPTW
Owner role:    ML Engineer
Status:        DRAFT
Inputs:        Instance batches; GART potential function
Outputs:       svrptw.solvers.learning.pomo
```

## Goal

A one-shot neural construction solver that produces a full multi-route plan
in milliseconds. The target is the speed axis of "beat OR-Tools across all
categories" (ADR-0001 D3): we want a Pareto-dominant point at < 1 second
wall-clock on every instance class.

## Architecture

Transformer encoder over node features, autoregressive decoder. Inputs per
node:

- normalized (x, y) coords
- demand
- TW (ready, due, service)
- shortest depot-distance (in/out)
- in/out asymmetry index

Plus a global vector: N, K, fleet capacity, mean depot distance, asym score,
day length.

Decoder actions: pick next customer, return to depot (start new route), or
terminate. Action mask enforces capacity, TW feasibility, visited-once.

POMO multi-start: K parallel decodings per instance, different seed
customers; the final solution is the lowest-cost over K.

Default sizes: encoder layers 4, dim 128, heads 8. POMO K=50.

## Training

- **Data:** instance generator from SPEC-0-INST-01 (curriculum: train on N=50 first, then N=100, then N=200; N=500 is eval-only).
- **Loss:** REINFORCE with baseline; baseline = mean of the POMO ensemble. Reward = negative operational cost.
- **Reward shaping:** potential-based with `Phi(s) = -GART(unserved nodes)` from SPEC-1-GART-01. Optimal policy is invariant under potential shaping (Ng-Harada-Russell 1999); we verify this on tiny instances.
- **Budget:** Single GPU (RTX 3070 Ti, 8 GB). Training fits in 12-24 hours per N-bin.

## Acceptance

- Inference < 200 ms for N=100 with K=50 starts on GPU.
- Greedy decoding (K=1) produces a feasible solution 100% of the time on `instances/v1`.
- On `instances/v1`, at sub-1-second budget, beats vanilla OR-Tools@1 by ≥ 10% on mean operational cost.

## Non-goals

- Heterogeneous fleet, multi-depot.
- Pretrained foundation-style models (RouteFinder, MVMoE). Single-variant
  policy specific to SVRPTW asymmetric.

## Dependencies

- SPEC-0-INST-01 (training instances).
- SPEC-1-GART-01 (potential function).
