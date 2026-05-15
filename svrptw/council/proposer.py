"""SPEC-8-COUNCIL-01 Loop 1 — LLM-driven operator proposer.

Reads the existing seed operators + the rejection corpus from the sqlite
memory and asks an LLM to propose a new OperatorProposal (rationale +
code + unit test), aiming to clear the council's composability bench
(SPEC-8-COUNCIL-02).

Uses google-genai SDK directly (same path as the dataset labeller's
Gemini fallback). Free-tier rate limits made the OpenRouter committee
unusable for label batches; the proposer follows the same single-judge
fallback pattern to keep iteration time bounded.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from svrptw.council.memory import _DB_PATH

_LOG = logging.getLogger("svrptw.council.proposer")

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "code": {"type": "string"},
        "unit_test": {"type": "string"},
        "seed_inspired_by": {"type": "string"},
        "generation": {"type": "integer"},
    },
    "required": ["rationale", "code", "unit_test", "seed_inspired_by", "generation"],
}


_PROMPT_HEADER = """\
You are a research-software engineer designing operator code for a
Vehicle Routing Problem with Time Windows (VRPTW) bandit solver.

The solver, `portfolio`, already has 13 native arms (relocate,
swap-star, 2-opt, 2-opt-star, 3-opt-intra, merge_routes, sisr,
ejection_chain, cyclic_3, vehicle_kill, soft_drop, drop_route,
destroy_island, drop_leg). At a 10-second wall-budget on N=50
asymmetric OSM-based instances, the bandit converges close to PyVRP.

Your job: propose ONE new operator that, when added as a 14th arm,
*complements* the existing 13. Operators that duplicate PyVRP's
internal LNS (like 2-opt-star) are rejected — the bench has confirmed
this on classical seeds (cross_route_2opt, gart_guided_ruin).

The acceptance bench (SPEC-8-COUNCIL-02 composability) runs portfolio
at 10s with vs without your operator (3 seeds, 8 instances). It
accepts iff: mean delta ≥ $2 across instances, hit-rate ≥ 40%,
no per-instance regression > 1%, no capacity overload, all feasible.

OPERATOR INTERFACE
    def operator(solution: Solution, context: OperatorContext) -> Solution | None

Where:
    Solution.routes: list[Route]; Route.customers: list[int] (1-indexed customer ids)
    context.instance: Instance with .num_customers, .vehicle_capacity, .depot,
        .customers, .travel_time (NxN), .travel_dist (NxN, asymmetric)
    context.settings: Settings, .economics has wage_per_minute, cost_per_mile, hard_late_penalty
    context.deadline_seconds: float, you MUST stop before exceeding it
    Return: improved Solution (lower operational_cost) or None
    Must NOT introduce capacity overload or infeasibility.

GUARDRAILS (your code is sandbox-validated):
- No banned imports: os, subprocess, socket, threading, ctypes
- Only stdlib + numpy + svrptw modules (svrptw.solvers.common, .config, .io)
- Re-evaluate before returning so .metrics is current

API CHEAT SHEET — use these EXACT signatures (LLMs frequently hallucinate
the wrong ones; the smoke gate has rejected 10/10 proposals for API errors):

```python
# Solution requires ALL of these fields at construction:
from svrptw.solvers.common import Solution, Route, evaluate
new_sol = Solution(
    instance_id=inst.instance_id,     # str
    routes=[Route(customers=[1, 2, 3]), Route(customers=[4, 5])],
    solver=solution.solver,           # str — copy from input
    wall_clock_seconds=solution.wall_clock_seconds,
    budget_seconds=solution.budget_seconds,
    feasible=False,                   # set after evaluate()
)
# evaluate() returns a DICT, not a float. Compare the 'operational_cost' key:
new_sol.metrics = evaluate(inst, new_sol, settings)  # dict
new_sol.feasible = bool(new_sol.metrics["feasible"])
if new_sol.metrics["operational_cost"] < solution.metrics["operational_cost"]:
    return new_sol
return None

# Customer fields: id (int, 1-indexed), demand, ready, due, service.
# Index by id from cust_by_id dict:
from svrptw.solvers.common.local_search import _cust_by_id
cust_by_id = _cust_by_id(inst)        # dict[int, Customer]
demand = cust_by_id[cid].demand        # float
# Routes operate on customer IDs, not Customer objects. NEVER use Customer as a dict key.

# Solution does NOT have .copy(); use list comprehension or import copy.deepcopy.
# Prefer: new_routes = [Route(customers=list(r.customers)) for r in solution.routes]

# Feasibility quick-check on a candidate route's customer list:
from svrptw.solvers.common.local_search import _route_arrival_and_close
ok, _ = _route_arrival_and_close(inst, [cid1, cid2, cid3])   # bool
# Capacity:
load = sum(cust_by_id[c].demand for c in route_customers)
cap = inst.vehicle_capacity
```

WHAT WOULD CLEAR THE BAR — heuristics that are STRUCTURALLY different
from PyVRP's classical moves:
  - Learning-guided ruin selection (ML scores, not pairwise distance)
  - Structural rewrites that span ≥3 routes simultaneously
  - Multi-objective swaps that trade cost for feasibility on tight TW
  - Composite moves: ruin + reinsert with a non-greedy repair
  - Instance-feature-aware moves (e.g., grid-detected routes get axial 2-opt)
"""


_PROMPT_FOOTER = """\
Respond with a SINGLE JSON object matching the OperatorProposal schema.
The "code" field MUST contain a complete Python module exposing
`def operator(solution, context)`. The "unit_test" field MUST contain
a self-contained pytest test that exercises the operator and asserts
its no-regret contract.

DO NOT exceed 80 lines for the operator code. DO NOT include explanatory
prose inside the code (use docstrings only). DO NOT include `import os`
or any banned module. Make the operator deterministic given its inputs.
"""


@dataclass
class ProposerResult:
    proposal: dict
    raw_response: str
    elapsed_s: float
    model: str


def _read_seed(path: Path) -> dict:
    """Extract rationale (top docstring) + code from a seed module."""
    text = path.read_text(encoding="utf-8")
    docstring = ""
    if text.startswith('"""'):
        end = text.find('"""', 3)
        if end > 0:
            docstring = text[3:end].strip()
    return {"name": path.stem, "rationale": docstring, "code": text}


def _read_rejection_corpus(limit: int = 5) -> list[dict]:
    """Pull recent rejection rationales for the LLM to learn from."""
    if not Path(_DB_PATH).exists():
        return []
    conn = sqlite3.connect(_DB_PATH)
    try:
        cur = conn.execute(
            "SELECT seed_inspired_by, reject_reason, shadow_result_json, "
            "rationale FROM proposals WHERE reject_reason IS NOT NULL "
            "OR (shadow_result_json IS NOT NULL AND "
            "    shadow_result_json LIKE '%\"accepted\": false%') "
            "ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
    finally:
        conn.close()
    out = []
    for seed, reject_reason, shadow_json, rationale in rows:
        why = reject_reason or ""
        if not why and shadow_json:
            try:
                why = json.loads(shadow_json).get("reasoning", "")
            except Exception:
                pass
        out.append({"seed": seed or "", "rationale": rationale or "",
                    "why_rejected": why})
    return out


def build_prompt(seed_dir: Path = Path("svrptw/council/seeds"),
                 rejection_limit: int = 5) -> str:
    """Assemble the full proposer prompt: header + seeds + rejection lore + footer."""
    seeds = []
    for sp in sorted(seed_dir.glob("*.py")):
        if sp.name == "__init__.py":
            continue
        seeds.append(_read_seed(sp))
    rejections = _read_rejection_corpus(rejection_limit)

    parts = [_PROMPT_HEADER, "\n## EXISTING SEEDS\n"]
    for s in seeds:
        parts.append(f"\n### {s['name']}\n")
        parts.append(f"Rationale: {s['rationale'][:600]}\n")
        # Truncate code so total prompt fits in context.
        code_clip = s["code"][:1800] + ("\n..." if len(s["code"]) > 1800 else "")
        parts.append(f"```python\n{code_clip}\n```\n")
    if rejections:
        parts.append("\n## REJECTION LORE — what NOT to propose\n")
        for r in rejections:
            parts.append(f"- Seed `{r['seed']}`: rejected because {r['why_rejected'][:200]}\n")
    parts.append("\n" + _PROMPT_FOOTER)
    return "".join(parts)


def propose(
    n: int = 1,
    *,
    model_id: Optional[str] = None,
    timeout_s: float = 60.0,
) -> list[ProposerResult]:
    """Call Gemini-direct n times, parse each response into a proposal dict."""
    import time
    from svrptw.vivrp.assessor import _GeminiBackend  # reuses .env loading
    backend = _GeminiBackend(model_id=model_id)
    prompt = build_prompt()
    out: list[ProposerResult] = []
    for _ in range(n):
        t0 = time.time()
        try:
            # Use the genai client directly with text-only contents.
            cfg = {
                "response_mime_type": "application/json",
                "response_schema": _GeminiBackend._strip_unsupported(PROPOSAL_SCHEMA),
                "temperature": 0.7,  # diversity matters for proposing
            }
            resp = backend._client.models.generate_content(
                model=backend._model_id, contents=[prompt], config=cfg,
            )
            raw = resp.text or ""
        except Exception as e:
            _LOG.info("gemini proposer call failed: %s", e)
            continue
        elapsed = time.time() - t0
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            _LOG.info("proposer response not valid JSON; skipping")
            continue
        out.append(ProposerResult(
            proposal=parsed, raw_response=raw,
            elapsed_s=elapsed, model=backend._model_id,
        ))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="LLM operator proposer (Gemini-direct)")
    p.add_argument("--n", type=int, default=1, help="Proposals to generate")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--model", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = propose(n=args.n, model_id=args.model)
    proposals = [r.proposal for r in results]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(proposals, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({len(proposals)} proposals; "
          f"elapsed: {sum(r.elapsed_s for r in results):.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
