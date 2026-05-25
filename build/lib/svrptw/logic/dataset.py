"""SPEC-6-LOGIC-01 preference-pair generator.

Iterates bench solutions, groups by instance_id, emits pairs, queries
the OpenRouter committee (with Gemini direct fallback), saves the
labelled pairs as a parquet/JSON for Bradley-Terry student training.

The output format is the contract that `svrptw.logic.student` consumes:

    {
      "instance_id": str,
      "solution_a_hash": str,
      "solution_b_hash": str,
      "label": float,           # 1.0 if A≻B, 0.0 if B≻A, 0.5 if tie/abstain
      "teacher_score_a": float, # committee consensus
      "teacher_score_b": float,
      "authoritative": bool,    # ensemble σ ≤ threshold
      "method": str,            # "committee", "gemini_solo", or "score_diff"
    }
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.logic.committee import OpenRouterCommittee
from svrptw.logic.teacher import solution_hash
from svrptw.solvers.common import Route, Solution
from svrptw.viz.renderer import render_solution


_LOG = logging.getLogger("svrptw.logic.dataset")


_LABEL_PROMPT = """You are a delivery dispatcher comparing two route plans
for the SAME instance. Score each plan in [0, 1] where:
  1.00 = ship as-is, no concerns
  0.50 = would rework one route
  0.00 = reject; structural problem

Things to weigh:
  - Coverage on tight time windows
  - Visual sanity (no spaghetti routes)
  - Route load balance + plausible breaks
  - First stop near depot, geographic flow

Respond with structured JSON only.
"""


@dataclass
class PreferencePair:
    instance_id: str
    solution_a_hash: str
    solution_b_hash: str
    solver_a: str
    solver_b: str
    label: float           # 1.0 A≻B, 0.0 B≻A, 0.5 tie
    teacher_score_a: Optional[float]
    teacher_score_b: Optional[float]
    authoritative_a: bool
    authoritative_b: bool
    method: str
    rationale: str = ""    # LLM's reasoning — scan to see what it's seeing

    def to_dict(self) -> dict:
        return asdict(self)


def _label_from_scores(
    score_a: Optional[float],
    score_b: Optional[float],
    eps: float = 0.05,
) -> float:
    """Bradley-Terry-style label from raw teacher scores.

    Within `eps` of each other → tie (0.5). Otherwise A wins iff
    score_a > score_b. Returns NaN if either side is None.
    """
    if score_a is None or score_b is None:
        return float("nan")
    if abs(score_a - score_b) < eps:
        return 0.5
    return 1.0 if score_a > score_b else 0.0


def _reconstruct_solution(
    inst, row: dict,
) -> Optional[Solution]:
    """Re-solve the (instance, solver) pair to get routes back.

    The bench JSONs only store metrics — to render and judge solutions
    we need the full Solution. This re-runs the solver deterministically
    (portfolio is bit-stable per our earlier verification).
    """
    solver = row["solver"]
    settings = Settings()
    if solver.startswith("portfolio@"):
        from svrptw.solvers.classical import portfolio as pm
        budget = float(solver.split("@")[1])
        return pm.solve(inst, settings, budget_seconds=budget)
    if solver.startswith("pyvrp@"):
        from svrptw.solvers.classical import pyvrp_solver as pv
        budget = float(solver.split("@")[1])
        return pv.solve(inst, settings, budget_seconds=budget)
    if solver == "auction_gart":
        from svrptw.solvers.classical import auction_gart as ag
        return ag.solve(inst, settings)
    if solver == "greedy":
        from svrptw.solvers.classical import greedy as g
        return g.solve(inst, settings)
    if solver in ("pomo_v1", "pomo_v1_greedy"):
        from svrptw.solvers.learning.pomo.infer import solve as pomo_solve
        return pomo_solve(inst, settings, n_starts=32,
                          greedy_decode=solver.endswith("_greedy"))
    _LOG.warning(f"unknown solver {solver}; skip")
    return None


def _judge_one(
    committee: OpenRouterCommittee, inst, sol: Solution, cache_dir: Path,
) -> tuple[Optional[float], bool]:
    """Render solution + run committee. Returns (score, authoritative)."""
    img_path = cache_dir / "imgs" / f"{inst.instance_id}__{sol.solver}__{solution_hash(sol)}.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    if not img_path.exists():
        # fair_mode=True: route-count-invariant rendering so the VLM
        # can't use "rainbow density" as a proxy for "many routes".
        render_solution(inst, sol, img_path, fair_mode=True)
    label = committee.label(inst, sol, _LABEL_PROMPT, img_path)
    return (label.score, label.authoritative)


_PAIR_PROMPT_HEADER = """You are a delivery dispatcher comparing TWO route
plans for the SAME instance. Both maps below use the SAME color encoding
(viridis: dark blue = empty route, bright yellow = capacity-full route)
and the SAME directional arrows. The exact numeric details for both
plans are provided in the markdown sections below.

For EACH plan, score in [0, 1]:
  1.00 = ship as-is, no concerns
  0.50 = would rework one route
  0.00 = reject; structural problem

Look at BOTH plans side-by-side. Use the IMAGES for spatial reasoning
(crossings, zig-zags, route shape) and the TEXT for exact numerics
(loads, costs, time-window slack). Don't guess numbers from pixels.

Return JSON with score_a, score_b, rationale, confidence.
"""


def _judge_pair_combined(
    committee: OpenRouterCommittee, inst, sol_a: Solution, sol_b: Solution,
    cache_dir: Path, *, rng=None,
) -> tuple[Optional[float], Optional[float], bool, str]:
    """SPEC-6-LOGIC-02 split-channel judging: one combined API call sees
    both rendered plans + both text channels.

    BLINDED: the text channel uses neutral labels "Plan A" / "Plan B"
    instead of solver names. Order is randomized per call (when `rng` is
    passed) so the LLM can't develop an A-vs-B side bias.

    Returns (sa, sb, authoritative, rationale) where sa/sb are the
    scores of the CALLER's sol_a / sol_b — i.e. order is unscrambled
    before returning. The captured rationale lets us scan what the LLM
    says it's seeing.
    """
    from svrptw.logic.solution_text import to_text

    img_a = cache_dir / "imgs" / f"{inst.instance_id}__{sol_a.solver}__{solution_hash(sol_a)}.png"
    img_b = cache_dir / "imgs" / f"{inst.instance_id}__{sol_b.solver}__{solution_hash(sol_b)}.png"
    img_a.parent.mkdir(parents=True, exist_ok=True)
    if not img_a.exists():
        render_solution(inst, sol_a, img_a, fair_mode=True)
    if not img_b.exists():
        render_solution(inst, sol_b, img_b, fair_mode=True)

    # Randomize A/B side to control for ordering bias. If swap=True,
    # caller's sol_a is sent to the LLM as "Plan B" (and vice versa).
    if rng is not None:
        swap = bool(rng.random() < 0.5)
    else:
        swap = False

    if swap:
        first_sol, second_sol = sol_b, sol_a
        first_img, second_img = img_b, img_a
    else:
        first_sol, second_sol = sol_a, sol_b
        first_img, second_img = img_a, img_b

    prompt = (
        _PAIR_PROMPT_HEADER
        + "\n\n" + to_text(inst, first_sol, blind_label="Plan A")
        + "\n\n" + to_text(inst, second_sol, blind_label="Plan B")
    )
    label = committee.label_pair(
        inst, first_sol, second_sol, prompt, first_img, second_img,
    )
    rationale = (label.rationale or "")[:400]
    if swap:
        # Unscramble: what the LLM called score_a is actually our sol_b's score.
        return (label.score_b, label.score_a, label.authoritative, rationale)
    return (label.score_a, label.score_b, label.authoritative, rationale)


def build_pairs(
    bench_path: Path,
    out_path: Path,
    *,
    instances_dir: Path = Path("instances/v1"),
    n_pairs: int = 100,
    seed: int = 0,
    require_authoritative: bool = False,
    cache_dir: Path = Path("bench/runs/logic_dataset_cache"),
    committee: Optional[OpenRouterCommittee] = None,
) -> list[PreferencePair]:
    """Generate up to `n_pairs` labelled preference pairs from a bench JSON.

    Groups bench rows by instance_id, samples within-instance solver
    pairs, judges each side via the committee, derives the Bradley-Terry
    label from the score difference.

    Strategy: skew sampling toward pairs whose cost ranking differs by
    >5 % so the pair is interesting (per SPEC-6-LOGIC-01).
    """
    rng = np.random.default_rng(seed)
    data = json.loads(Path(bench_path).read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data

    by_inst: dict[str, list[dict]] = {}
    for r in rows:
        by_inst.setdefault(r["instance_id"], []).append(r)

    # Candidate pairs: every within-instance solver pair, with a
    # priority score = |cost_a - cost_b| / max(cost_a, cost_b).
    candidates = []
    for iid, rs in by_inst.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                ca, cb = rs[i].get("operational_cost"), rs[j].get("operational_cost")
                if ca is None or cb is None:
                    continue
                pri = abs(ca - cb) / max(ca, cb, 1e-6)
                candidates.append((pri, iid, rs[i], rs[j]))
    # Sort by priority desc, then take up to n_pairs from the front.
    # SPEC-6-LOGIC-01 says "skew toward >5% cost-diff pairs" — this
    # ordering does exactly that without hard-dropping any candidate.
    candidates.sort(reverse=True)
    selected = candidates[:n_pairs]

    if committee is None:
        committee = OpenRouterCommittee()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[PreferencePair] = []
    for k, (_, iid, row_a, row_b) in enumerate(selected, start=1):
        inst_path = instances_dir / f"{iid}.json"
        if not inst_path.exists():
            _LOG.warning(f"instance file not found: {inst_path}; skip")
            continue
        inst = load_instance(str(inst_path))
        t0 = time.perf_counter()
        sol_a = _reconstruct_solution(inst, row_a)
        sol_b = _reconstruct_solution(inst, row_b)
        if sol_a is None or sol_b is None:
            continue
        # SPEC-6-LOGIC-02 split-channel: single combined call sees both
        # solutions' images + both text summaries. Halves API cost +
        # gives VLM side-by-side context. BLINDED + A/B-randomized.
        sa, sb, auth_pair, rationale_pair = _judge_pair_combined(
            committee, inst, sol_a, sol_b, cache_dir, rng=rng,
        )
        auth_a = auth_b = auth_pair
        label = _label_from_scores(sa, sb)
        if require_authoritative and not (auth_a and auth_b):
            continue
        pair = PreferencePair(
            instance_id=iid,
            solution_a_hash=solution_hash(sol_a),
            solution_b_hash=solution_hash(sol_b),
            solver_a=row_a["solver"],
            solver_b=row_b["solver"],
            label=label,
            teacher_score_a=sa,
            teacher_score_b=sb,
            authoritative_a=auth_a,
            authoritative_b=auth_b,
            method="committee_blind",
            rationale=rationale_pair,
        )
        out.append(pair)
        elapsed = time.perf_counter() - t0
        _LOG.info(f"[{k}/{len(selected)}] {iid} {row_a['solver']} vs {row_b['solver']}: "
                  f"sa={sa} sb={sb} label={label} auth={(auth_a, auth_b)} "
                  f"({elapsed:.1f}s)")
        if rationale_pair:
            _LOG.info(f"  rationale: {rationale_pair[:220]}")
        # Flush every 10 pairs in case of mid-run interrupt.
        if k % 10 == 0:
            Path(out_path).write_text(
                json.dumps([p.to_dict() for p in out], indent=2), encoding="utf-8")

    Path(out_path).write_text(
        json.dumps([p.to_dict() for p in out], indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True, help="Path to bench rows JSON")
    p.add_argument("--out", required=True, help="Output preference-pair JSON")
    p.add_argument("--instances", default="instances/v1")
    p.add_argument("--n-pairs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--require-auth", action="store_true",
                   help="Only emit pairs where both sides are authoritative.")
    args = p.parse_args(argv)
    pairs = build_pairs(
        bench_path=Path(args.bench),
        out_path=Path(args.out),
        instances_dir=Path(args.instances),
        n_pairs=args.n_pairs,
        seed=args.seed,
        require_authoritative=args.require_auth,
    )
    print(f"wrote {args.out} ({len(pairs)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
