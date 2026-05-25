"""Continue the consistency study with models 3-4 + add Paris pair."""
from __future__ import annotations

import os
import sys
import statistics
from pathlib import Path

sys.path.insert(0, "D:/SVRPTW")
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")
_env = Path("D:/SVRPTW/.env")
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from bench.scripts.judge_consistency import (
    LIGHTEST_FIRST, _call_one, _push_panel, _render_pair,
)
from webui import client as ui


def run_pair(img_a: Path, img_b: Path, iid: str, models, reps: int = 3):
    img_a_url = f"/snapshots/consistency/{img_a.name}"
    img_b_url = f"/snapshots/consistency/{img_b.name}"
    for model in models:
        pair_id = f"{iid}__{model['weight_class']}"
        calls = []
        wc = model["weight_class"]
        print(f"-- {model['id']} ({wc}) on {iid} --")
        for r in range(reps):
            v = _call_one(model, img_a, img_b, run_index=r)
            calls.append(v)
            _push_panel(pair_id, model["id"], calls, img_a_url, img_b_url)
            if v.get("error"):
                print(f"  r{r}: ERROR {v['error'][:80]}")
            else:
                rat = (v.get("rationale") or "")[:60]
                print(f"  r{r}: score={v['score']:.2f}  {rat}...")
        valid = [c["score"] for c in calls if c.get("score") is not None]
        if len(valid) > 1:
            mu = statistics.mean(valid); sd = statistics.stdev(valid); rg = max(valid) - min(valid)
            print(f"  mean={mu:.3f}  sigma={sd:.3f}  range={rg:.2f}  -> "
                  f"{'CONSISTENT' if sd < 0.1 else 'NOISY'}")
        elif valid:
            print(f"  single valid = {valid[0]:.3f}")
        else:
            print("  ALL FAILED")
        print()


if __name__ == "__main__":
    ui.push_stage("judge-consistency-run2",
                   "models 3-4 (30B nemotron + 31B gemma) on cached Manhattan; then Paris pair")
    img_a = Path("D:/SVRPTW/webui/static/snapshots/consistency/OSM-Manhattan-N100-I003__warm.png")
    img_b = Path("D:/SVRPTW/webui/static/snapshots/consistency/OSM-Manhattan-N100-I003__pyvrp.png")
    print("=== Manhattan N=100, models 3-4 ===")
    run_pair(img_a, img_b, "OSM-Manhattan-N100-I003", LIGHTEST_FIRST[2:4], reps=3)

    print("=== Render Paris N=200 pair ===")
    pa, pb, piid = _render_pair("instances/v1/OSM-Paris-N200-I003.json",
                                  warm_budget=10.0, pyvrp_budget=20.0)
    # Move to consistency dir if not already there
    print(f"pair: {pa} | {pb}")
    print()
    print("=== Paris N=200, models 1-2 (lightest) ===")
    run_pair(pa, pb, piid, LIGHTEST_FIRST[0:2], reps=3)

    ui.push_log("=== consistency run 2 done ===")
    ui.push_log("see dashboard's VLM judge panel for per-(model,pair) consistency badges")
