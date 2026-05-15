"""SPEC-8-COUNCIL-01 Loop 1 — operator-proposer orchestrator.

Reads a JSON file containing one or more OperatorProposal entries
(typically produced by an LLM agent), validates each against the
proposal schema (parse → signature → unit-test → smoke), runs the
shadow bench on those that pass, and persists every result to the
sqlite corpus for the next generation of proposers to reflect on.

The LLM agent that *produces* proposals is intentionally out of
scope — this script consumes proposals from any source (human,
council agent, ad-hoc test). Glue an OpenRouter committee call
upstream when you're ready.

Usage:
  python -m bench.scripts.council_loop1 \\
    --proposals data/council/incoming_2026_05_13.json \\
    --shadow

Proposal file format (JSON):
  [
    {
      "rationale": "one paragraph why this should help",
      "code": "def operator(solution, context):\\n    ...",
      "unit_test": "def test_op_smoke():\\n    ...",
      "seed_inspired_by": "TW-anchor relocate",
      "generation": 1
    },
    ...
  ]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from svrptw.council.memory import record, stats
from svrptw.council.proposal import OperatorProposal, validate_proposal
from svrptw.council.shadow_bench import run_shadow


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--proposals", required=True,
                   help="JSON file containing list of OperatorProposal dicts")
    p.add_argument("--shadow", action="store_true",
                   help="Run shadow bench on proposals that pass validation")
    args = p.parse_args()

    raw = json.loads(Path(args.proposals).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raw = [raw]

    accepted = rejected = 0
    for entry in raw:
        prop = OperatorProposal(
            rationale=entry["rationale"],
            code=entry["code"],
            unit_test=entry["unit_test"],
            seed_inspired_by=entry.get("seed_inspired_by"),
            generation=int(entry.get("generation", 0)),
        )
        print(f"\n[council] proposal {prop.proposal_id} (gen {prop.generation})")
        print(f"  inspired-by: {prop.seed_inspired_by or '<none>'}")
        decision = validate_proposal(prop)
        if not decision.can_run_bench:
            print(f"  REJECTED — {decision.reject_reason}")
            record(prop, decision, shadow_result=None)
            rejected += 1
            continue
        print("  schema: parse OK, signature OK, unit-test OK, smoke OK")
        shadow_result = None
        if args.shadow:
            print("  running shadow bench …")
            shadow_result = run_shadow(
                prop.proposal_id, decision.operator_callable,
            )
            print(f"  shadow: posterior_mean={shadow_result['posterior_mean']:.3f}  "
                  f"accepted={shadow_result['accepted']}  "
                  f"({shadow_result['reasoning']})")
        record(prop, decision, shadow_result=shadow_result)
        if shadow_result is None or shadow_result["accepted"]:
            accepted += 1
        else:
            rejected += 1

    print(f"\n[council] this batch: accepted={accepted}, rejected={rejected}")
    print(f"[council] corpus totals: {stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
