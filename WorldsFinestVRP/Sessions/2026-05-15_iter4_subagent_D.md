---
title: Phase D3+D4+D5 RL roadmap implementation (subagent D)
date: 2026-05-15
tags: [rl, bandit, phase-D, session-log]
---

# Phase D3+D4+D5 — RL roadmap completion

Implemented Phases D3 (offline-training scaffolding), D4 (reward shaping)
and D5 (A/B integration) per `WorldsFinestVRP/11 - RL Roadmap.md`. The
LinUCB bandit's choose/update interface is preserved and dispatched via
a new `bandit_kind` switch in `portfolio.solve` (`linucb` | `logging` |
`mlp`). `solve_auto` exposes `rl_neighborhood: bool` plus `rl_artifact`
which auto-set `bandit_kind="mlp"` and `shape_reward=True`. Reward
shaping is gated on `shape_reward=False` by default so existing solves
are bit-identical to the pre-D4 path (verified: paired runs produce
identical operational_cost at fixed seed). Files created:
`svrptw/solvers/learning/logging_bandit.py` (LoggingBandit wraps
LinUCBBandit and records JSONL transitions, exposed via the new
`pm._LAST_BANDIT` registry slot); `svrptw/solvers/learning/policy_mlp.py`
(MLPBanditPolicy with [128,64] ReLU MLP, 16+3*|ops| input,
Boltzmann-softmax sampling, periodic REINFORCE updates, save/load to
.pt — torch is lazy-imported with a helpful RuntimeError when missing);
`svrptw/solvers/learning/reward_shaping.py` (Phase D4 island/isolated/leg
shaping with default coefs 8/4/0.05); `bench/scripts/collect_bandit_logs.py`,
`bench/scripts/train_policy_offline.py`, `bench/scripts/rl_vs_linucb_ab.py`
(all with `--help` working and a sys.path bootstrap so they run via
`python bench/scripts/...`); `tests/unit/test_rl_smoke.py` (8 passing
tests + 1 skipped torch-missing test). Files edited:
`svrptw/solvers/classical/portfolio.py` (added `bandit_kind`,
`policy_artifact`, `shape_reward`, `shape_coefs` kwargs; dispatch +
shaped-reward composition between `cost_after` and `bandit.update`),
`svrptw/solvers/classical/portfolio_pyvrp_warm.py` (forwarded the four
kwargs in `solve` and added `rl_neighborhood`/`rl_artifact` in
`solve_auto`). All seven exit criteria pass: imports OK,
`pytest tests/unit -q` reports 122 passed / 1 skipped (no regressions),
`test_rl_smoke.py` 8 passed / 1 skipped, both bench-script `--help`
invocations succeed.
