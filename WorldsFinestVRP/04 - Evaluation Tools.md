# Evaluation infrastructure

## Bench harness

- **`bench/parallel.py`** — `ProcessPoolExecutor` wrapper for parallel solves. Default max_workers=4. Required by `bench/PARALLEL_GUIDANCE.md` for any new bench.
- **`bench/scripts/`** — ~50 scripts, one per experiment. JSON results dumped to `bench/runs/<name>.json`.
- **Status**: working well. Two crash modes have been encountered and fixed:
  1. Print-format strings with `:>+N` on strings → crashed before JSON write. Now: write JSON first, summary print second.
  2. Bench bug truncated stdout in the harness capture; the JSON file is the authoritative result.

## Council shadow benches (SPEC-8-COUNCIL)

- **`svrptw/council/shadow_bench.py`**
  - `run_shadow()` — single-shot: apply operator once after a portfolio@10 warmstart, measure absolute delta + hit rate. Floors: mean_abs_delta ≥ $5, hit_rate ≥ 30%.
  - `composability_run()` — 14th-arm test: portfolio (13 arms) vs portfolio + candidate (14 arms) at same wall budget. Paired seed = same bandit RNG between baseline and augmented runs. Floors: mean_abs_delta ≥ $2, hit_rate ≥ 40%, worst_regress ≤ 1%.
- **Calibration history**:
  - Initial bench used `delta_per_s` (rate metric) — inflated by fast-exit ops. Replaced with mean absolute delta.
  - n_repeats=3 without paired seeding: ~$8 exploration-variance noise floor; all operators clustered at the floor.
  - Paired seeding (`portfolio.solve(seed=)` threading to LinUCBBandit + SISR): variance collapsed; operators now distinguishable.
  - 8-instance subset → 16-instance subset: caught selection bias on LLM-accepted operators ($4 win on 8 became $0.13 on 16).
- **Status**: working well. The composability bench correctly distinguishes "PyVRP-redundant" (-$2.14, 0% hit) from "marginal" ($0.13, 25% hit) from "noise-floor positive" ($4, 38% hit).

## OOD validation protocol

- **Held-out indices**: instance file `I=000` was used in any tuning that involved tuning instances (e.g., cb=8 tuning, the v1 N=200 small-bench). `I=001`-`I=004` are genuinely held out and form the OOD set.
- **Held-out cities**: cb=8 tuning used 4 of 8 cities (Manhattan, Paris, SanFrancisco, Phoenix). The other 4 (Charleston, Austin, Pittsburgh, Cambridge) are extra-held-out for cross-city generalization tests.
- **Combined OOD cumulative result**: 83/88 wins (94%) on truly held-out instances across N=100/200/500.

## LLM operator proposer (svrptw/council/proposer.py)

- Gemini-direct path (genai SDK), single-call per proposal, JSON-schema-constrained output. Free-tier rate limits made OpenRouter committee unusable.
- Prompt = HEADER + 5 seed operators + 6 recent rejection rationales + API CHEAT SHEET + FOOTER ≈ 3200 tokens.
- API CHEAT SHEET was added after n=10 batch yielded 0/10 schema pass (LLM hallucinated Solution constructor signatures); with the cheat sheet n=10 yielded 2/10 schema pass.
- **Yield this session**: 25 total proposals across n=5+10+10 batches → 5 cleared schema gates → 0 cleared paired-seed composability on wider 16-instance bench.
- **Honest assessment**: the proposer is plumbing-correct but produces operators whose value is dominated by selection bias on small bench subsets. Either we need much harder problems where naive operators have room, OR a much stronger LLM, OR the proposer should propose composable-bench *test instances* rather than operators.

## Academic-benchmark loader

- **`svrptw/io/solomon.py::load_solomon`** — parses Solomon `VEHICLE/CUSTOMER` text format. Returns Instance with symmetric Euclidean travel matrices (Solomon convention: time = distance, speed = 1). Verified by hitting C101's published optimum 828.94 within 0.01%.
- **`bench/scripts/fetch_solomon.py`** — downloads the 56 Solomon N=100 from iRB-Lab/py-ga-VRPTW mirror (CVRPLIB's URL is dead as of May 2026).
- **`bench/scripts/fetch_homberger.py`** — downloads 48 of the 120 Gehring-Homberger N=200/400 from ML4VRP/ML4VRP2023 (the mirror has a subset, not full set).
- **Status**: working. Both loaders successfully parse and solve, and our cost decoded back to standard distance units matches published optima.

## Logic-axis / LLM judge (deprecated)

- `svrptw/logic/committee.py` + `logic/ensemble.py` — five-head LogicStudent ensemble distilled from VLM committee preference labels.
- Closed off as a research direction this session: single-judge Gemini labelling has too-coarse score resolution (binary 1.0/0.5) to discriminate among strong solvers.

## Memory / corpus

- **`svrptw/council/memory.py`** — sqlite at `cache/council/corpus.sqlite` recording every proposal (accepted + rejected), with full code, rationale, decision, and shadow result. Currently 27 entries (3 accepted at schema, 24 rejected; 0 accepted at composability under wider bench).

## Multi-objective evaluator

- **`svrptw/solvers/common/solution.py::evaluate`** — operational cost = wage·time + cost·dist + early_wait·penalty + missed·hard_late + overload·hard_late + per_route_fixed_cost·n_routes.
- **`per_route_fixed_cost` (SPEC-7-COST-02)** — default 0.0, gated; enables the multi-objective headline. PyVRP's internal objective doesn't see this field.
