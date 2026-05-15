# SPEC-0-CI-01 — Deterministic CI

```
ID:            SPEC-0-CI-01
Title:         GitHub Actions workflow with regression and smoke gates
Owner role:    Bench Engineer
Status:        FROZEN
Inputs:        .github/workflows/ci.yml
Outputs:       PR-gating CI green/red
```

## Behavior

Every PR runs:

1. **Lint** — `ruff check svrptw/ tests/ specs/`
2. **Typecheck** — `mypy svrptw/`
3. **Unit tests** — `pytest tests/unit/ -x -n auto` (parallel)
4. **Smoke benchmark** — `python -m svrptw.bench run --solvers greedy,ortools --instances v1 --filter N=50 --budget 1 --out /tmp/smoke.json`. Must finish < 5 min.
5. **Smoke regression** — assert per-instance cost is within ±0.5% of `bench/baselines/v1_smoke.json` (a separate, smaller frozen file). Tolerance accounts for floating-point + thread non-determinism.
6. **Spec hygiene** — every `specs/SPEC-*.md` parses; every referenced ID exists.

CI image: `python:3.11-slim` + apt `concorde` (we vendor a Linux Concorde binary in CI; LKH-3 not required in CI).

GPU jobs are gated behind a `gpu` label and only run on self-hosted runners (the workstation).

## Invariants

- A no-op PR is green.
- A deliberate perturbation in any solver hot path turns CI red on step 5.
- Lint/type/test never depend on network access.

## Acceptance

- `.github/workflows/ci.yml` committed.
- A perturbation-test branch demonstrates CI red.
- `bench/baselines/v1_smoke.json` committed (10 instances × 2 solvers × 1 budget).

## Non-goals

- Full v1 benchmark in CI (8 hours — workstation-only).
- Coverage reporting (deferred).
- Auto-formatting commits (use `ruff format` locally, not in CI).

## Dependencies

- SPEC-0-CFG-01, SPEC-0-INST-01, SPEC-0-BENCH-01.
