# ADR-0001 — Project direction lock

**Status:** Accepted
**Date:** 2026-05-11
**Principal:** Stephen Feldman

## Context

The original SVRPTW codebase is a stochastic-routing research benchmark with four solvers (Greedy / OR-Tools / DQN / Bayesian Auction). The Spec-Driven Development plan in the principal's brief proposed five phases (Phase 0 hygiene → Phase 5 paper-ready evaluation) on top of the existing instance generator. The principal has now redirected:

1. **Drop the stochastic emphasis.** The publishable lever is no longer "risk-aware routing under log-normal travel delays." It is **one-shot fast solution quality on hard deterministic VRPs**.
2. **Generate truly hard, asymmetric, network-based instances.** Euclidean random points in a 100×100 box are not hard enough to differentiate solvers. Real road networks are.
3. **Target a generational improvement over OR-Tools.** Not "win at one operating point" — **win at sub-1-second budgets, win at unrestricted budgets, and solve cases OR-Tools cannot.** This is the bar the principal set; we will measure honestly against it.
4. **Reuse the frozen GART artifacts** from `D:/VRP-Advanced-Estimator-Integration/` (LightGBM v3/v4 tour-length estimators). No retraining in Phase 1.
5. **Use Concorde (WSL) and LKH-3 as TSP/ATSP oracles** for ground-truth validation of GART and as strong baselines.

## Decisions

### D1. Python 3.11 venv under `D:\SVRPTW\.venv\`
System Python is 3.14 and lacks wheels for `torch`, `ortools`, `numba`. We install Python 3.11.9 and a self-contained venv. The venv path is committed to `.gitignore`; activation is documented in the README. CI uses the same 3.11 image.

### D2. Hard instances generated from OSMnx road networks
The new instance set `instances/v1/` is **deterministic, asymmetric, road-network-based**. Source: OSMnx drive networks for a diverse set of mid-size cities. Each instance: a depot node, N customer nodes, true shortest-path travel times computed on the directed graph (asymmetric), time windows scaled to the network's natural travel-time distribution. The Euclidean generator (`data_generator.py`) is retired but kept for backwards-compat regression.

### D3. Benchmark target: dominate OR-Tools across all axes
- **Speed axis:** one-shot solver < 1 s; we beat OR-Tools' 1 s/10 s/60 s points on cost.
- **Quality axis:** at unrestricted budget our pipeline (one-shot + improvement) matches or beats OR-Tools.
- **Capability axis:** demonstrate cases where OR-Tools' CP-SAT model is brittle (very large N, deep asymmetry, complex objectives) and we still produce feasible high-quality plans.

The Phase 5 leaderboard reports all three axes side-by-side.

### D4. Stochastic component dropped from the critical path
`simulator.py`, `Stochastic_Evaluator.py`, the 30-day Monte Carlo loops — kept as optional regression but **not on the headline benchmark**. Phase 3 of the original plan (Stochastic GART) is deferred indefinitely. Reward shaping in Phase 4 uses the deterministic GART, not the stochastic variant.

### D5. Single GPU only
Training fits an RTX 3070 Ti Laptop (8 GB). No distributed training. POMO/construction models are sized accordingly.

### D6. Gemini API integration is deferred
ViTSP-style visual solution review with Gemini is on the roadmap but blocked on the principal providing an API key. Until then, local-only evaluation.

## Consequences

- The original SDD plan's Phase 3 (Stochastic GART) is removed. Phase numbering shifts: hygiene → GART service → GART-enhanced solvers → POMO+improvement → paper evaluation.
- The `instances/v1/` manifest is *new*, not the existing Euclidean set. Old baselines in `solutions/` are orphaned and excluded from `bench/baselines/v1.json`.
- "Beat OR-Tools across all categories" is a higher bar than the original `≥ 3%` greedy-gain spec. Per-phase gates are tightened in spec amendments.
- "Generation breakthrough" framing — the win condition is **Pareto dominance**, not point wins. Phase 5 must produce evidence of dominance, not cherry-picked operating points.

## Open questions

- Which cities seed the OSMnx instance pool? Decided in `SPEC-0-INST-01`.
- What is the "case OR-Tools cannot do" instance class? Candidates: N ≥ 500, asymmetry-heavy graphs (one-way-dominated downtowns), very tight TWs on long routes. To be specified in `SPEC-0-INST-02`.
- LKH-3 as VRP solver (not just TSP) — eligible as a comparison baseline? Yes; treated alongside OR-Tools in the leaderboard.
