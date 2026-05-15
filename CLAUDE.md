# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

SVRPTW research codebase from Stephen Feldman (RPI, "Decision Making Under Uncertainty"). Benchmarks four solvers for the Stochastic Vehicle Routing Problem with Time Windows against a stochastic simulator (log-normal travel delays, normal service time noise) — Greedy, OR-Tools, DQN, and a Bayesian Auction.

**Active improvement plan.** The repo is being rewritten under a Spec-Driven Development plan (see the principal's brief / `specs/` once created). Direction: drop the stochastic emphasis, generate **hard asymmetric network-based instances**, fold the GART tour-length estimator from `D:/VRP-Advanced-Estimator-Integration/` and `D:/Area-and-Distribution-Free-Estimator-for-TSP/` into the auction/MCTS/POMO solvers, and target one-shot solutions that surpass OR-Tools at sub-second budgets. Concorde (WSL: `/usr/local/bin/concorde`) and LKH-3 (`C:\LKH\LKH-3.exe`) are available as TSP oracles. GART artifacts live as `lgbm_alpha_model_v3.joblib` / `lgbm_alpha_model_v4.joblib` under `D:/VRP-Advanced-Estimator-Integration/estimators/`.

## Repository layout (current, pre-refactor)

Flat `.py` files at the repo root, no package. Everything imports from `config.py` constants and from sibling modules by file name. The Phase 0 refactor target is a `svrptw/` package — until that lands, treat top-level files as the canonical layout.

- `config.py` — global constants (wages $14.50/hr, $0.50/mile, depot 480–960 min, `TRAVEL_TIME_LN_SIGMA=0.6`, `SERVICE_TIME_SIGMA=8.0`). Most numbers in solvers come from here.
- `data_generator.py` — generates the "Extreme" instance set into `instances/data/N{N}_V{V}_I{I}.json` (Euclidean, depot at (50,50), TW 30–90 min, vehicle_factor 0.30, capacity_buffer 1.40).
- `simulator.py` — stochastic transition engine; `StochasticSampler` draws delays.
- `vrp_gym_env.py` — Gymnasium wrapper for the DQN.
- Solvers (each writes JSONs into `solutions/<SolverName>/`):
  - `greedy_evaluator.py` — nearest-neighbor lower bound
  - `deterministic_policy_generator.py` — OR-Tools CP-SAT static plan
  - `Stochastic_Evaluator.py` — replays OR-Tools static plans through `simulator`
  - `rl_trainer.py` — dual-stream DQN (global context + per-node features)
  - `auction_solver.py` — decentralized market mechanism with Numba-accelerated distance matrices and a `DualBayesianEstimator` for risk buffering (this is the headline contribution per the README; the route-recompute-per-bid is the bottleneck GART should eliminate)
  - `mcts_solver.py` / `mcts_hybrid_solver.py` — MCTS variants (currently budget-starved on random rollouts; intended target for GART rollout replacement)
  - `rollout_agent.py`, `cooperative_planner.py`, `adp_solver.py`, `genetic_optimizer.py` — additional experimental policies
- `tuner.py` — Optuna sweep over auction priors and risk buffers
- `metrics_aggregator.py`, `strategy_visualizer.py` — post-hoc analysis (heatmaps, failure overlays)
- `best_solver_params.json` / `best_solver_params_2.json` — Optuna-tuned auction parameters
- `instances/data/*.json` — 100 instances at N ∈ {20, 40, 60, 80, 100}
- `solutions/<Solver>/` — per-solver result dumps + visuals

## Architecture notes that span files

1. **Config is global, not injected.** Every solver does `from config import ...`. Phase 0's `SPEC-0-CFG-01` moves these to a typed pydantic config; until then, do not add new constants to `config.py` — pipe them through solver kwargs.
2. **Distance is Euclidean from `data_generator.euclidean_distance`.** Multiple solvers reimplement it under Numba (`auction_solver.fast_euclidean`). Any move to **asymmetric** distances (the planned hard instances) needs to thread a `dist_matrix` argument everywhere instead of recomputing from coords.
3. **Instance JSON shape.** `{num_customers, num_vehicles, vehicle_capacity, depot: {...}, customers: [...]}` — solvers and the simulator all assume this shape. Loaders live in `deterministic_policy_generator.load_instance`.
4. **Stochastic simulation is 30 days/instance, seeded inside `simulator.py`.** The seeding is not yet hash-of-(instance_id, day); reproducibility regressions usually trace to this.
5. **Cost model.** Operational cost = wage·time + transit·miles + `HARD_LATE_PENALTY=1000` per missed TW. Greedy and OR-Tools optimize Euclidean distance only; auction/DQN optimize cost directly.
6. **The auction's hot path** is route re-insertion cost per bid in `auction_solver.py`. Replacing this with GART marginal estimates is `SPEC-2-AUCTION-01` and is the main publishable lever.

## Common commands

This repo currently has no package, no `pyproject.toml`, and no test runner. Commands are bare scripts run from the repo root.

```powershell
# Generate the instance set
python data_generator.py

# Baselines
python greedy_evaluator.py
python deterministic_policy_generator.py   # writes static OR-Tools plans
python Stochastic_Evaluator.py             # simulates those plans

# DQN
python rl_trainer.py                       # trains, then auto-evaluates

# Auction (the headline solver)
python auction_solver.py
python tuner.py                            # Optuna sweep of auction params

# MCTS variants
python mcts_solver.py
python mcts_hybrid_solver.py
python run_all_mcts.py                     # sweeps both

# Analysis
python metrics_aggregator.py
python strategy_visualizer.py              # writes solutions/<Solver>/visuals/
```

Use Python 3.11 or 3.12 — system default is 3.14 and lacks wheels for `torch`, `ortools`, and `numba`. Verify with `python -c "import torch, ortools, numba"` before running solvers. The DQN benefits from CUDA (RTX 3070 Ti Laptop, 8 GB present).

## External oracles and adjacent repos

- **Concorde** (WSL): `wsl.exe -e bash -c "concorde -x ..."` — exact symmetric TSP. Use for ground-truth tour lengths when validating GART.
- **LKH-3**: `C:\LKH\LKH-3.exe` — high-quality asymmetric TSP/VRP solver. Use as the strong baseline for the new hard asymmetric instances.
- **GART estimators**: `D:/VRP-Advanced-Estimator-Integration/estimators/lgbm_estimator_v4.py` wraps `lgbm_alpha_model_v4.joblib`. The estimator predicts an `alpha` such that `L ≈ alpha · sqrt(n · A)`; features come from `feature_creator_v3.py`. Do **not** retrain — artifacts are frozen.
- **GART training repo** (read-only reference): `D:/Area-and-Distribution-Free-Estimator-for-TSP/` contains the LightGBM v3/v4 training pipeline, SHAP analysis, and TSPLIB benchmarks.

## Working rules specific to this codebase

- **Numba-jitted functions cache to `__pycache__`** — if you change a `@nb.njit` body, also delete the cache or stale signatures will silently win.
- **Do not commit `solutions/` or `instances/data/`** if they're regenerated — the Phase 0 plan freezes a SHA-256 manifest in `instances/v1/manifest.sha256` instead.
- **No silent refactors outside frozen specs** (rule from the SDD plan). If a change isn't covered by a `specs/SPEC-*.md`, surface it and either get a spec amendment or split the PR.
- **Stochastic component is being de-emphasized** per the principal — when in doubt, prefer deterministic-asymmetric formulations over stochastic ones for new work.
- **GPU is available; use it** for any training or batch inference where it helps. CPU-only fallbacks belong in tests, not in solver hot paths.

## Graphify

Knowledge-graph dump lives under `graphify-out/` (created on first `/graphify` run). The Stop hook auto-rebuilds it after each session. If `graphify-out/GRAPH_REPORT.md` exists, prefer it for architecture questions over re-scanning files. Manual rebuild: `python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"`.
