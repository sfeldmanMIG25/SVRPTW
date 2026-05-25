"""LLM-judge consistency study (lightest models first).

Hypothesis: a multi-judge VLM panel is only useful as a selection
signal if individual judges are CONSISTENT on the same input. The
prior session closed the logic-axis explicitly because single-judge
resolution washed out at scale; this study quantifies that per model.

Protocol:
  1. Render N solution pairs (warm vs pyvrp) into webui/static/snapshots/consistency/
  2. For each judge model, lightest first, call it K times on each pair
     at the same prompt + same image bytes.
  3. Measure score variance + rationale agreement per (model, pair).
  4. Push EVERY individual call to the dashboard so the user sees
     - the two images
     - per-call score + rationale snippet
     - running mean/std/range badge

Output: bench/runs/judge_consistency.json with all calls + per-(model,pair) stats.
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

# load .env if present so SVRPTW_OPENROUTER_API_KEY / SVRPTW_GEMINI_API_KEY work
_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")


from svrptw.io import load_instance
from svrptw.config import Settings
from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto
from svrptw.solvers.classical import pyvrp_solver as pv
from svrptw.viz.renderer import render_llm_compare
from webui import client as ui

# Lightest -> heaviest. LOCAL LM Studio first (no rate limits, fastest),
# then Gemini direct (better rate limits than OpenRouter free tier),
# then free-tier OpenRouter (last resort for diversity in committee).
#
# Gemini free-tier rate limits (verified 2026-05-15 via list_models):
#   gemini-2.0-flash-lite: 30 RPM / 1M TPM / 1500 RPD  (smallest, fastest)
#   gemini-2.5-flash-lite: similar
#   gemini-2.0-flash: 15 RPM
#   gemini-2.5-flash: 10 RPM
#   gemini-2.5-pro:   5  RPM
LIGHTEST_FIRST: list[dict[str, Any]] = [
    {"id": "qwen/qwen3-vl-4b", "kind": "lmstudio",
     "tier": "L", "weight_class": "4B-local", "vision": True},
    {"id": "gemini-2.0-flash-lite", "kind": "gemini",
     "tier": "A", "weight_class": "flash-lite-2.0", "vision": True},
    {"id": "gemini-2.5-flash-lite", "kind": "gemini",
     "tier": "A", "weight_class": "flash-lite-2.5", "vision": True},
    {"id": "gemini-2.0-flash", "kind": "gemini",
     "tier": "A", "weight_class": "flash-2.0", "vision": True},
    {"id": "gemini-2.5-flash", "kind": "gemini",
     "tier": "A", "weight_class": "flash-2.5", "vision": True},
    # Gemma open-weight via Google API (separate quota from OpenRouter)
    {"id": "models/gemma-4-26b-a4b-it", "kind": "gemini",
     "tier": "A", "weight_class": "gemma-26b-google", "vision": True},
    {"id": "models/gemma-4-31b-it", "kind": "gemini",
     "tier": "A", "weight_class": "gemma-31b-google", "vision": True},
    {"id": "nvidia/nemotron-nano-12b-v2-vl:free", "kind": "openrouter",
     "tier": "B", "weight_class": "12B-or", "vision": True},
    {"id": "google/gemma-4-26b-a4b-it:free", "kind": "openrouter",
     "tier": "B", "weight_class": "26B-MoE-or", "vision": True},
    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "kind": "openrouter",
     "tier": "B", "weight_class": "30B-MoE-or", "vision": True},
    {"id": "google/gemma-4-31b-it:free", "kind": "openrouter",
     "tier": "B", "weight_class": "31B-or", "vision": True},
]


_PROMPT = """You are evaluating two vehicle-routing solutions for the same set of customers.
Both images use the same color palette and clean basemap so the routes are directly comparable.

Image A is on the left; Image B is on the right (passed to you in that order).
Score how much you prefer image A on a 0..1 scale where:
  0.0 = strongly prefer B,  0.5 = no preference,  1.0 = strongly prefer A.

Judge by routing quality only:
  - shorter total travel distance for the same coverage
  - tighter clusters per route, less back-and-forth
  - fewer crossings between routes

Return JSON exactly matching the schema with score, rationale (2-4 sentences), and confidence."""


_SNAP_DIR = Path(__file__).resolve().parent.parent.parent / "webui" / "static" / "snapshots" / "consistency"
_SNAP_DIR.mkdir(parents=True, exist_ok=True)


def _render_pair(inst_path: str, warm_budget: float = 8.0,
                  pyvrp_budget: float = 30.0,
                  cache: bool = True) -> tuple[Path, Path, str]:
    """Render warm + pyvrp PNGs for one instance. Returns (a, b, instance_id)."""
    inst = load_instance(inst_path)
    iid = inst.instance_id
    a_path = _SNAP_DIR / f"{iid}__warm.png"
    b_path = _SNAP_DIR / f"{iid}__pyvrp.png"

    if cache and a_path.exists() and b_path.exists():
        ui.push_log(f"[render] cached pair for {iid}")
        return a_path, b_path, iid

    ui.push_log(f"[render] solving warm + pyvrp on {iid} ...")
    settings = Settings()
    a_sol = solve_auto(inst, settings, budget_seconds=warm_budget)
    b_sol = pv.solve(inst, settings, budget_seconds=pyvrp_budget)

    render_llm_compare(inst, a_sol, a_path,
                        title=f"A: solve_auto cost={a_sol.metrics['operational_cost']:.0f}")
    render_llm_compare(inst, b_sol, b_path,
                        title=f"B: PyVRP cost={b_sol.metrics['operational_cost']:.0f}")
    ui.push_log(f"[render] {iid}  warm={a_sol.metrics['operational_cost']:.0f}  "
                 f"pyvrp={b_sol.metrics['operational_cost']:.0f}")
    return a_path, b_path, iid


def _call_one(model: dict[str, Any], img_a: Path, img_b: Path,
              run_index: int) -> dict[str, Any]:
    """Call a single judge model on the (img_a, img_b) pair and return verdict."""
    from svrptw.logic.openrouter_client import call_model, JUDGMENT_SCHEMA
    t0 = time.perf_counter()
    try:
        if model["kind"] == "lmstudio":
            from svrptw.logic.lmstudio_client import call_model as _lm_call
            r = _lm_call(model["id"], _PROMPT, image_path=img_a,
                          extra_image_paths=[img_b])
            return {
                "model": model["id"], "kind": "lmstudio",
                "tier": model["tier"], "weight_class": model["weight_class"],
                "run_index": run_index,
                "score": r.score, "rationale": r.rationale,
                "confidence": r.confidence,
                "latency_s": r.latency_s, "error": r.error,
            }
        elif model["kind"] == "openrouter":
            resp = call_model(
                model["id"], _PROMPT, image_path=img_a,
                extra_image_paths=[img_b], schema=JUDGMENT_SCHEMA,
            )
            return {
                "model": model["id"], "kind": "openrouter",
                "tier": model["tier"], "weight_class": model["weight_class"],
                "run_index": run_index,
                "score": float(resp.score),
                "rationale": resp.rationale,
                "confidence": float(resp.confidence),
                "latency_s": resp.latency_s,
                "error": None,
            }
        elif model["kind"] == "gemini":
            # Direct genai call so we can vary the model id per request.
            import google.generativeai as _genai
            from PIL import Image as _PIL
            key = (os.environ.get("SVRPTW_GEMINI_API_KEY")
                   or os.environ.get("GEMINI_API_KEY"))
            if not key:
                raise RuntimeError("SVRPTW_GEMINI_API_KEY not set")
            _genai.configure(api_key=key)
            mdl = _genai.GenerativeModel(model["id"])
            t1 = time.perf_counter()
            resp = mdl.generate_content([
                _PROMPT,
                _PIL.open(img_a),
                _PIL.open(img_b),
            ])
            lat = time.perf_counter() - t1
            text = (resp.text or "").strip()
            from svrptw.logic.lmstudio_client import _parse_score_json
            sc, rat, conf = _parse_score_json(text)
            return {
                "model": model["id"], "kind": "gemini",
                "tier": model["tier"], "weight_class": model["weight_class"],
                "run_index": run_index,
                "score": sc, "rationale": rat, "confidence": conf,
                "latency_s": lat,
                "error": None if sc is not None else f"no score parsed from: {text[:120]}",
            }
        else:
            return {"model": model["id"], "kind": model["kind"],
                    "run_index": run_index, "error": f"unknown kind {model['kind']}",
                    "score": None, "rationale": "", "confidence": None,
                    "tier": model["tier"], "weight_class": model["weight_class"],
                    "latency_s": time.perf_counter() - t0}
    except Exception as e:
        return {"model": model["id"], "kind": model["kind"],
                "run_index": run_index, "error": f"{type(e).__name__}: {e}",
                "score": None, "rationale": "", "confidence": None,
                "tier": model["tier"], "weight_class": model["weight_class"],
                "latency_s": time.perf_counter() - t0}


def _push_panel(pair_id: str, model_id: str, calls: list[dict[str, Any]],
                 img_a_url: str, img_b_url: str) -> None:
    """Push the running list of verdicts for one (pair, model) to dashboard."""
    valid = [c for c in calls if c.get("score") is not None]
    if valid:
        scores = [c["score"] for c in valid]
        consensus = {
            "score": statistics.mean(scores),
            "stdev": (statistics.stdev(scores) if len(scores) > 1 else 0.0),
            "score_range": max(scores) - min(scores),
            "n_responded": len(valid),
            "n_failed": len(calls) - len(valid),
            "image_a_url": img_a_url,
            "image_b_url": img_b_url,
            "model_under_test": model_id,
        }
    else:
        consensus = {
            "score": None, "stdev": None, "score_range": None,
            "n_responded": 0, "n_failed": len(calls),
            "image_a_url": img_a_url, "image_b_url": img_b_url,
            "model_under_test": model_id,
        }
    ui.push_judges(pair_id=pair_id, judges=calls, consensus=consensus)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instances", nargs="+",
                   default=["instances/v1/OSM-Manhattan-N050-I003.json"],
                   help="Instance JSON paths to render & judge. "
                        "Default = N=50 because N=100+ produces too many "
                        "overlapping routes for human OR VLM to read clearly.")
    p.add_argument("--reps", type=int, default=3,
                   help="N calls per (model, pair)")
    p.add_argument("--n-models", type=int, default=2,
                   help="How many models from LIGHTEST_FIRST to test (default 2)")
    p.add_argument("--warm-budget", type=float, default=8.0)
    p.add_argument("--pyvrp-budget", type=float, default=20.0)
    p.add_argument("--out", default="bench/runs/judge_consistency.json")
    args = p.parse_args()

    ui.push_agent("llm-consistency-study", "running",
                   summary=f"{len(args.instances)} pairs x {args.n_models} models x {args.reps} reps", progress=0.0)
    ui.push_stage("judge-consistency",
                   f"{len(args.instances)} pairs x {args.n_models} models x {args.reps} reps")

    # Stage 1: render all pairs
    pairs: list[tuple[Path, Path, str]] = []
    for inst_path in args.instances:
        a, b, iid = _render_pair(inst_path, warm_budget=args.warm_budget,
                                  pyvrp_budget=args.pyvrp_budget)
        pairs.append((a, b, iid))

    models = LIGHTEST_FIRST[:args.n_models]
    total_calls = len(pairs) * len(models) * args.reps
    done = 0
    rows: list[dict[str, Any]] = []


    ui.push_progress("judge_consistency", done, total_calls)

    for (img_a, img_b, iid) in pairs:
        # URLs the dashboard can load (relative to /snapshots/)
        img_a_url = f"/snapshots/consistency/{img_a.name}"
        img_b_url = f"/snapshots/consistency/{img_b.name}"
        for model in models:
            pair_id = f"{iid}__{model['weight_class']}"
            calls: list[dict[str, Any]] = []
            ui.push_log(f"[judge] {pair_id} starting (model={model['id']})")
            for r in range(args.reps):
                v = _call_one(model, img_a, img_b, run_index=r)
                calls.append(v)
                done += 1
                _push_panel(pair_id, model["id"], calls, img_a_url, img_b_url)
                ui.push_progress("judge_consistency", done, total_calls)
                err = v.get("error")
                if err:
                    ui.push_log(f"  [{r+1}/{args.reps}] ERROR {err[:80]}")
                else:
                    ui.push_log(f"  [{r+1}/{args.reps}] score={v['score']:.2f}  rationale={v['rationale'][:60]}...")
                rows.append({**v, "pair_id": pair_id, "instance_id": iid,
                              "image_a": str(img_a), "image_b": str(img_b)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))

    ui.push_agent("llm-consistency-study", "completed",
                   summary=f"{total_calls} calls done across {len(pairs)} pairs x {len(models)} models")
    ui.push_log(f"[judge] DONE -- wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
