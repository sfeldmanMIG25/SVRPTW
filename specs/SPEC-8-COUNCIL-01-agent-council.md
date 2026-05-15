# SPEC-8-COUNCIL-01 — Three-loop agent council

```
ID:            SPEC-8-COUNCIL-01
Title:         Operator-proposer / component-merger / distillation-watcher
               agent loops sharing a single artifact store + shadow bench
Owner role:    Research Lead
Status:        FROZEN
Depends on:    SPEC-3-PORTFOLIO-01 (LinUCB to absorb new arms),
               SPEC-6-LOGIC-01 (student to evaluate logic-aware proposals),
               SPEC-8-SHADOW-BENCH-01 (sibling spec for shadow arm)
```

## Why

The codebase is at a point where new operators / new wirings can
plausibly outperform the existing 13-arm bandit, but the search
space is too large for a human to enumerate. ReEvo-style reflective
evolution (Liu 2024) demonstrated that an agent council with real
selection pressure + a rejection-corpus memory can propose useful
LNS operators at scale. Our novelty is that the bench gives the
agents a real Pareto signal — cost, wall, and (once SPEC-6-LOGIC-01
ships) dispatcher-acceptance — instead of synthetic eval.

## Three loops

### Loop 1 — Operator proposer

- Agents read `svrptw.solvers.common.local_search` + `local_search_destroy`
  + the operator portfolio in `svrptw.solvers.classical.portfolio`.
- Proposals: Python code conforming to
  `def operator(solution: Solution, context: OperatorContext) -> Solution | None`
  + one-paragraph rationale + one-passing unit test.
- The bench runs accepted proposals as a *shadow arm* of the LinUCB
  bandit (or a sibling bandit dedicated to proposals). Keep iff the
  bandit's posterior mean exceeds a threshold after N pulls.
- Reflection: rejected proposals carry a post-hoc explanation; the
  next generation reads the rejection corpus.

### Loop 2 — Component merger

- Agents propose **config diffs**, not new code. Examples:
  - "SISR uses GART as insertion cost"
  - "Auction's bid valuation uses portfolio's bandit instead of fixed formula"
  - "POMO's reward = logic teacher's score on the resulting render"
- Format: YAML patches against the frozen baseline config in
  `svrptw/config/defaults.yaml`.
- Bench runs each as a new solver variant via the existing harness.

### Loop 3 — Distillation watcher

- Reads accepted variants from Loops 1 + 2 and extracts what each
  one *learned* as a feature or rule.
- Output: a new feature added to `svrptw/logic/features.py`, a rule
  added to greedy initialization, or a constraint added to POMO's
  action mask.
- Goal: after ~100 kept variants, a small distilled student with
  this enriched feature set recovers ≥ 80 % of the gain at ≤ 5 % of
  the inference cost.

## What the council MUST NOT do

1. **No strategic decisions.** Loop-choice, validation cadence, and
   stop conditions are human-set.
2. **No cost-model changes.** Cost-model edits alter the bench's
   selection pressure; need human review.
3. **No logic-axis proposals before the LogicStudent is validated.**
   Right now logic = one Gemini call per eval. Too slow, too biased
   for use as a selection signal. Gates open when the student hits
   the SPEC-6-LOGIC-01 acceptance criteria (≥ 80 % held-out teacher
   agreement, < 50 ms inference, ECE ≤ 0.05).

## What the council MUST have

### Proposal schema (machine-checkable)

```python
# svrptw/council/proposal.py
@dataclass
class OperatorProposal:
    proposal_id: str              # sha1 of (code + rationale)
    rationale: str                # one paragraph
    code: str                     # Python source defining operator()
    unit_test: str                # Python source defining test_* functions
    seed_inspired_by: str | None  # e.g. "TW-anchor relocate"
    generation: int               # 0 for seeded, 1+ for proposed
```

Reject any proposal that:
- Doesn't parse.
- Has the wrong operator signature.
- Has no unit test, or the unit test doesn't pass.

### Shadow bench (SPEC-8-SHADOW-BENCH-01)

- 8 instances at N=50 (1 per city) + 4 at N=100. Run each proposal
  through these as a bandit arm alongside the 13-arm portfolio.
- ~2 minutes total per proposal (heavily parallelisable).
- Accept iff per-proposal bandit posterior mean (cost-delta per
  second) is above the 25th percentile of the existing arm
  population AND not Pareto-dominated on (cost, wall).
- Writes to `bench/runs/council/shadow/<proposal_id>.json` — separate
  namespace from main leaderboard.

### Memory and reflection (`svrptw/council/memory.py`)

- SQLite at `cache/council/corpus.sqlite`.
- Schema:
  ```
  proposals(id PK, generation, rationale, code, unit_test,
            shadow_result_json, decision, reject_reason,
            created_at, accepted_at)
  ```
- Each proposer call retrieves the last 50 rejected proposals'
  rationales + reject reasons as context. Eligibility filter:
  same seed-family or same operator-family.

## Cold-start operator candidates

Six seed proposals to bootstrap Loop 1 (Stephen's list):

1. **GART-guided ruin** — destroy operator. Pick customers to remove
   weighted by GART's marginal-cost-gradient instead of random.
   Should beat SISR-random on tight-TW instances.
2. **Route-pair zip** — order two routes by polar angle from depot,
   interleave by angle, re-evaluate. Spatial-prior move not currently
   present in the operator pool.
3. **TW-anchor relocate** — pick the customer with tightest TW slack
   in any route; try inserting it as the first customer of every
   other route. Mimics manual dispatcher behaviour.
4. **Capacity-rebalance swap** — find two routes with > 90 % vs
   < 50 % util; swap their last-inserted customer pair. Targets
   the under-utilization the logic-axis judge penalises.
5. **Lookahead-2 insertion** — for each candidate insertion, evaluate
   the move's quality after one *more* greedy insertion. Finds moves
   no single-step operator can see.
6. **Convex-hull-respecting 2-opt** — accept only 2-opt moves that
   don't increase the route's convex-hull perimeter. Cheap visual-
   coherence prior the logic judge would likely reward.

These seed Loop 1; the council proposes mutations + new families.

## Concrete acceptance gates

| gate | criterion |
|---|---|
| Loop 1 productive | ≥ 5 proposals accepted into bandit pool per week |
| Shadow → main bench | accepted proposal improves main-bench dom % by ≥ 1.0 pp at one N |
| Distilled student | 100 accepted variants → student ECE ≤ 0.05 on held-out, +5 pp dominance contribution |
| Cost neutrality | council's bench cost ≤ $50/week (OpenRouter free + occasional paid) |

## Files this spec creates

| Path | Role |
|---|---|
| `svrptw/council/__init__.py` | package |
| `svrptw/council/proposal.py` | proposal dataclasses + validators |
| `svrptw/council/shadow_bench.py` | shadow-arm orchestrator |
| `svrptw/council/memory.py` | sqlite corpus |
| `svrptw/council/seeds/` | 6 cold-start operator implementations |
| `bench/scripts/council_loop1.py` | CLI entry point |
| `tests/unit/test_council_*.py` | schema + shadow-bench invariants |
| `SPEC-8-SHADOW-BENCH-01-shadow-arm.md` | sibling spec for shadow infra |

## Order of operations

Per Stephen's first-actions list:

1. Proposal schema + shadow-bench arm (half-day infra). *This spec's primary deliverable.*
2. Wire Loop 1 with the 6 seed operators + a 4-model committee for proposal generation.
3. Run Loop 1 for one week; accumulate rejection corpus.
4. Train LogicStudent on collected labels (gating Loops 2 + 3).
5. Add Loops 2 + 3 once Loop 1 has ≥ 50 accepted-or-rejected operators.
