"""Overlay ablation study: does showing cost / util / TW change judge scores?

Hypothesis (user, iter 5b): "we want to see visual quality, not anything
else." Cost is an abstraction; if a VLM is shown the cost annotation it
will pick the cheaper one without actually evaluating clusters / route
shape / coverage. Strip the abstraction and see if the same judges
still discriminate the SAME pair.

Protocol:
  1. Render the same (warm, pyvrp) pair under N overlay variants:
        - "minimal":     no title, no cost, no util in legend
        - "utilization": legend shows util%
        - "cost":        legend shows per-route distance
        - "tw":          TW-tightness ring around each customer
        - "full":        everything
  2. Use stable per-route colors (same customer-set -> same color)
     so the visual structure is identical across variants.
  3. Call lightest VLM model K=3 times on each variant.
  4. Compare: does the judge's mean score CHANGE across variants?
     If yes -> the abstraction was driving the judgment, not visual quality.
     If no  -> the judge actually sees structural quality.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# load .env
_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")
sys.path.insert(0, "D:/SVRPTW")


from svrptw.io import load_instance
from svrptw.config import Settings
from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto
from svrptw.solvers.classical import pyvrp_solver as pv
from svrptw.viz.renderer import render_llm_compare
from webui import client as ui
from bench.scripts.judge_consistency import LIGHTEST_FIRST, _PROMPT, _call_one, _push_panel


_OVERLAYS = ["minimal", "utilization", "cost", "tw", "full"]
_OUT_DIR = Path("D:/SVRPTW/webui/static/snapshots/consistency/overlays")
_OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_pair_overlays(inst_path: str, warm_b: float, pyvrp_b: float):
    """Render the (warm, pyvrp) pair under EVERY overlay variant.

    Returns dict {overlay: (img_a, img_b, instance_id)}.
    Stable colors via shared route_color_map across the full sweep.
    """
    inst = load_instance(inst_path)
    iid = inst.instance_id
    print(f"[overlay] solving + rendering {iid}")
    settings = Settings()
    warm = solve_auto(inst, settings, budget_seconds=warm_b)
    pyvrp = pv.solve(inst, settings, budget_seconds=pyvrp_b)
    print(f"[overlay] warm cost={warm.metrics['operational_cost']:.0f} K={int(warm.metrics['num_vehicles_used'])}")
    print(f"[overlay] pyvrp cost={pyvrp.metrics['operational_cost']:.0f} K={int(pyvrp.metrics['num_vehicles_used'])}")
    # Shared color map: route_id (frozenset of customers) -> palette slot.
    color_map: dict = {}
    out: dict[str, tuple[Path, Path, str]] = {}
    for ov in _OVERLAYS:
        a = _OUT_DIR / f"{iid}__{ov}__warm.png"
        b = _OUT_DIR / f"{iid}__{ov}__pyvrp.png"
        # SAME color_map passed BOTH times to keep colors stable across A and B.
        render_llm_compare(inst, warm, a, title="A", overlay_mode=ov,
                            route_color_map=color_map)
        render_llm_compare(inst, pyvrp, b, title="B", overlay_mode=ov,
                            route_color_map=color_map)
        out[ov] = (a, b, iid)
        ui.push_log(f"[overlay] rendered {ov} variant for {iid}")
    return out, warm, pyvrp


# Prompt-tweak: explicit "judge by visual structure" wording. Avoids
# language that primes the VLM to look for cost numbers.
_VISUAL_PROMPT = """Two vehicle-routing solutions for the same customer set are shown
side-by-side. Image A is on the left; Image B is on the right.

Judge by VISUAL STRUCTURE only:
  - Are the routes compact / tightly clustered?
  - Are there fewer crossings between routes?
  - Do routes avoid unnecessary back-and-forth detours?

Score how much you prefer image A on a 0..1 scale where:
  0.0 = strongly prefer B,  0.5 = no preference,  1.0 = strongly prefer A.

Return JSON with these fields exactly:
  {"score": number 0..1, "rationale": "2-4 sentences", "confidence": number 0..1}
You may wrap the JSON in a markdown code block if needed; both will be parsed."""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", default="instances/v1/OSM-Manhattan-N050-I003.json")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--n-models", type=int, default=2,
                   help="how many models from LIGHTEST_FIRST to test (default 2)")
    p.add_argument("--warm-budget", type=float, default=4.0)
    p.add_argument("--pyvrp-budget", type=float, default=10.0)
    p.add_argument("--overlays", nargs="+", default=_OVERLAYS, choices=_OVERLAYS)
    p.add_argument("--out", default="bench/runs/judge_overlay_ablation.json")
    args = p.parse_args()

    ui.push_agent("overlay-ablation", "running",
                   summary=f"{args.instance} x {len(args.overlays)} variants x {args.n_models} models x {args.reps} reps")
    ui.push_stage("overlay-ablation",
                   f"render N={args.instance.rsplit('-N',1)[-1].split('-')[0]} pair under {len(args.overlays)} variants")

    rendered, warm, pyvrp = render_pair_overlays(
        args.instance, args.warm_budget, args.pyvrp_budget)


    models = LIGHTEST_FIRST[:args.n_models]
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, dict[str, float]]] = {}  # model -> overlay -> stats

    # Patch in the visual-only prompt for these calls
    import bench.scripts.judge_consistency as _jc
    _jc._PROMPT = _VISUAL_PROMPT

    total_calls = len(args.overlays) * len(models) * args.reps
    done = 0
    for ov in args.overlays:
        img_a, img_b, iid = rendered[ov]
        img_a_url = f"/snapshots/consistency/overlays/{img_a.name}"
        img_b_url = f"/snapshots/consistency/overlays/{img_b.name}"
        for model in models:
            pair_id = f"{iid}__overlay-{ov}__{model['weight_class']}"
            calls: list[dict[str, Any]] = []
            wc = model["weight_class"]
            ui.push_log(f"[overlay] {pair_id} starting")
            for r in range(args.reps):
                v = _call_one(model, img_a, img_b, run_index=r)
                v["overlay"] = ov
                calls.append(v)
                done += 1
                _push_panel(pair_id, model["id"], calls, img_a_url, img_b_url)
                ui.push_progress("overlay_ablation", done, total_calls)
                if v.get("error"):
                    ui.push_log(f"  [{r+1}/{args.reps}] ERR {v['error'][:80]}")
                else:
                    ui.push_log(f"  [{r+1}/{args.reps}] score={v['score']:.2f}")
                rows.append({**v, "pair_id": pair_id, "instance_id": iid})
            valid = [c["score"] for c in calls if c.get("score") is not None]
            mu = (statistics.mean(valid) if valid else None)
            sd = (statistics.stdev(valid) if len(valid) > 1 else 0.0)
            summary.setdefault(model["id"], {})[ov] = {
                "mean": mu, "stdev": sd, "n": len(valid),
            }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows, "summary": summary},
                                           indent=2))

    print("\n=== overlay ablation summary (mean score per overlay) ===")
    for m, ov_map in summary.items():
        line = f"  {m.split('/')[-1]:35s} | "
        for ov in args.overlays:
            s = ov_map.get(ov, {})
            mu = s.get("mean")
            line += f"{ov}={mu:.2f} " if mu is not None else f"{ov}=ERR "
        print(line)

    ui.push_agent("overlay-ablation", "completed", summary=f"{len(rows)} calls; see {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
