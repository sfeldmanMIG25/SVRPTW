"""Multi-judge VLM panel batch script.

Renders pairs of solutions in `llm_compare` mode, then dispatches each
pair to the multi-judge panel (OpenRouter Tier-A + Gemini direct +
Anthropic Haiku). Dumps verdicts and consensus to a JSON file.

Two input modes:
  --instances PATH       JSON list of {instance_path, [auto_cost, pyvrp_cost]}.
  --input PATH           Re-judge an existing rebench run (e.g. cb_scaled_rebench.json).

The script always re-solves both arms (auto + pyvrp) at small budgets so
the rendered images correspond to a fresh, deterministic plan. Costs in
the input file are only used for sorting / sanity logging.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.logic.multi_judge import judge_pair
from svrptw.solvers.classical import pyvrp_solver as pv
from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto
from svrptw.viz.renderer import render_llm_compare


_DEFAULT_PROMPT = """\
You are an expert dispatcher comparing two VRP solutions for the SAME
instance. Image 1 is Solution A, Image 2 is Solution B. Each route is
drawn in a distinct tab10 color and labeled with its index (1, 2, ...).
Black square is the depot; small black dots are served customers; gray
"x" markers are unrouted (missed) customers.

Score in [0, 1]:
  1.0  -> Solution A is unambiguously better (cleaner clustering, fewer
        crossings, less obvious detour, fewer unrouted customers)
  0.5  -> Toss-up
  0.0  -> Solution B is unambiguously better

Provide a 2-4 sentence rationale and a confidence in [0, 1].
"""


def _push(stage: str, msg: str = "", **extra: Any) -> None:
    try:
        from webui import client as ui
        ui.push_stage(stage, msg, **extra)
    except Exception:
        pass


def _push_progress(name: str, completed: int, total: int,
                   eta_s: float | None = None) -> None:
    try:
        from webui import client as ui
        ui.push_progress(name, completed, total, eta_seconds=eta_s)
    except Exception:
        pass


def _load_pairs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Returns a list of {instance_path, [auto_cost], [pyvrp_cost]} dicts."""
    out: list[dict[str, Any]] = []
    if args.instances:
        raw = json.loads(Path(args.instances).read_text(encoding="utf-8"))
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            ip = entry.get("instance_path") or entry.get("instance")
            if not ip:
                continue
            out.append({
                "instance_path": str(ip),
                "auto_cost": entry.get("auto_cost"),
                "pyvrp_cost": entry.get("pyvrp_cost"),
            })
    elif args.input:
        raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
        for entry in raw:
            if not isinstance(entry, dict) or entry.get("_failed"):
                continue
            ip = entry.get("instance_path")
            if not ip:
                continue
            out.append({
                "instance_path": str(ip),
                "auto_cost": entry.get("auto_cost"),
                "pyvrp_cost": entry.get("pyvrp_cost"),
            })
    return out


def _render_pair(inst, auto_sol, pyvrp_sol, snap_dir: Path,
                 instance_id: str) -> tuple[Path, Path]:
    snap_dir.mkdir(parents=True, exist_ok=True)
    a_path = snap_dir / f"{instance_id}__A_auto.png"
    b_path = snap_dir / f"{instance_id}__B_pyvrp.png"
    render_llm_compare(inst, auto_sol, a_path,
                       title=f"Solution A: {auto_sol.solver}")
    render_llm_compare(inst, pyvrp_sol, b_path,
                       title=f"Solution B: {pyvrp_sol.solver}")
    return a_path, b_path


def _build_prompt(inst, auto_sol, pyvrp_sol,
                  prior_auto: float | None,
                  prior_pyvrp: float | None) -> str:
    """Default prompt + numeric context (text channel)."""
    parts = [_DEFAULT_PROMPT, "\n## Numeric context (do NOT use to score, ",
             "geometry / clustering only):\n"]
    ctx = {
        "instance_id": inst.instance_id,
        "N": inst.num_customers,
        "vehicle_capacity": float(inst.vehicle_capacity),
        "A_solver": auto_sol.solver,
        "A_routes": int(auto_sol.metrics.get("num_vehicles_used", -1)),
        "A_cost": float(auto_sol.metrics.get("operational_cost", -1)),
        "A_missed": int(auto_sol.metrics.get("missed_deliveries", 0)),
        "B_solver": pyvrp_sol.solver,
        "B_routes": int(pyvrp_sol.metrics.get("num_vehicles_used", -1)),
        "B_cost": float(pyvrp_sol.metrics.get("operational_cost", -1)),
        "B_missed": int(pyvrp_sol.metrics.get("missed_deliveries", 0)),
    }
    if prior_auto is not None:
        ctx["A_prior_run_cost"] = float(prior_auto)
    if prior_pyvrp is not None:
        ctx["B_prior_run_cost"] = float(prior_pyvrp)
    parts.append(json.dumps(ctx, indent=2))
    return "".join(parts)


def _enabled_set(judges_arg: list[str] | None) -> dict[str, bool]:
    if not judges_arg:
        return {"openrouter": True, "gemini": True, "haiku": True}
    s = {j.lower() for j in judges_arg}
    return {
        "openrouter": "openrouter" in s,
        "gemini": "gemini" in s,
        "haiku": "haiku" in s or "anthropic" in s,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--instances", default=None,
                     help="JSON list of {instance_path, [auto_cost, pyvrp_cost]} entries")
    src.add_argument("--input", default=None,
                     help="Re-judge an existing rebench JSON (e.g. bench/runs/cb_scaled_rebench.json)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the number of pairs judged (judges are slow + rate-limited)")
    ap.add_argument("--out", default="bench/runs/judge_solutions_multi.json")
    ap.add_argument("--webui", action="store_true",
                    help="Push live progress + judge panels to webui.event_bus")
    ap.add_argument("--judges", nargs="+", default=None,
                    choices=["openrouter", "gemini", "haiku", "anthropic"],
                    help="Subset of judges to enable (default: all three)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Thread workers per judge call (concurrent judges)")
    ap.add_argument("--budget-auto", type=float, default=10.0,
                    help="Wall-clock budget for the auto solver (seconds)")
    ap.add_argument("--budget-pyvrp", type=float, default=10.0,
                    help="Wall-clock budget for the pyvrp baseline (seconds)")
    ap.add_argument("--snapshots", default="webui/static/snapshots/judge_multi",
                    help="Directory to write rendered PNGs")
    args = ap.parse_args()

    pairs = _load_pairs(args)
    if args.limit:
        pairs = pairs[: int(args.limit)]
    if not pairs:
        print("No pairs to judge. Check --instances / --input arguments.")
        return 1


    enabled = _enabled_set(args.judges)
    print(f"[judge_solutions_multi] {len(pairs)} pair(s); judges enabled = "
          f"{[k for k, v in enabled.items() if v]}")
    _push("judge_multi", f"{len(pairs)} pairs queued",
          n_pairs=len(pairs), judges=[k for k, v in enabled.items() if v])
    _push_progress("judge_solutions_multi", 0, len(pairs))

    snap_dir = Path(args.snapshots)
    settings = Settings()
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for idx, pair in enumerate(pairs):
        ip = pair["instance_path"]
        try:
            inst = load_instance(ip)
        except Exception as e:
            rows.append({"_failed": True, "_error": f"load_instance: {e}",
                         "instance_path": ip})
            print(f"  [{idx + 1}/{len(pairs)}] FAILED to load {ip}: {e}")
            continue
        try:
            auto_sol = solve_auto(inst, settings, budget_seconds=args.budget_auto)
            pyvrp_sol = pv.solve(inst, settings, budget_seconds=args.budget_pyvrp)
        except Exception as e:
            rows.append({"_failed": True, "_error": f"solve: {e}",
                         "instance_path": ip, "instance_id": inst.instance_id})
            print(f"  [{idx + 1}/{len(pairs)}] FAILED to solve {inst.instance_id}: {e}")
            continue

        try:
            a_path, b_path = _render_pair(inst, auto_sol, pyvrp_sol,
                                           snap_dir, inst.instance_id)
        except Exception as e:
            rows.append({"_failed": True, "_error": f"render: {e}",
                         "instance_path": ip, "instance_id": inst.instance_id})
            print(f"  [{idx + 1}/{len(pairs)}] FAILED to render {inst.instance_id}: {e}")
            continue


        prompt = _build_prompt(inst, auto_sol, pyvrp_sol,
                               pair.get("auto_cost"), pair.get("pyvrp_cost"))
        pair_id = f"{inst.instance_id}__auto_vs_pyvrp"

        verdicts, consensus = judge_pair(
            a_path, b_path,
            prompt=prompt,
            enable_openrouter=enabled["openrouter"],
            enable_gemini=enabled["gemini"],
            enable_anthropic_haiku=enabled["haiku"],
            max_workers=args.workers,
            pair_id=pair_id,
            push_to_webui=args.webui,
        )

        rows.append({
            "instance_path": ip,
            "instance_id": inst.instance_id,
            "N": int(inst.num_customers),
            "pair_id": pair_id,
            "image_a": str(a_path),
            "image_b": str(b_path),
            "A_solver": auto_sol.solver,
            "A_cost": float(auto_sol.metrics.get("operational_cost", -1)),
            "B_solver": pyvrp_sol.solver,
            "B_cost": float(pyvrp_sol.metrics.get("operational_cost", -1)),
            "verdicts": [asdict(v) for v in verdicts],
            "consensus": asdict(consensus),
        })

        done = idx + 1
        elapsed = time.perf_counter() - t0
        eta = (elapsed / done) * (len(pairs) - done) if done > 0 else 0.0
        cs = consensus.score if consensus.score is not None else float("nan")
        print(f"  [{done:>3}/{len(pairs)}] {inst.instance_id:<32}  "
              f"n={consensus.n_responded}/{consensus.n_responded + consensus.n_failed}  "
              f"score={cs:.3f}  dissent={consensus.dissent}  "
              f"({elapsed:.0f}s, eta {eta:.0f}s)")
        sys.stdout.flush()
        if args.webui:
            _push_progress("judge_solutions_multi", done, len(pairs), eta_s=eta)


    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {len(rows)} judged pair(s) to {out_path} "
          f"in {time.perf_counter() - t0:.1f}s")
    _push("idle", f"judge_multi finished, {len(rows)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
