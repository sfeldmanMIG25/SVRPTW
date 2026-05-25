"""Run lightest models on the truly-anonymized pair (no title/legend/attribution).

Compares against the prior labeled-minimal results to see whether stripping
the A/B title and the route-count legend changes scores.
"""
from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

# load .env
_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")
sys.path.insert(0, "D:/SVRPTW")

from bench.scripts.judge_consistency import LIGHTEST_FIRST, _call_one, _push_panel
from webui import client as ui


def main() -> int:
    img_a = Path("D:/SVRPTW/webui/static/snapshots/consistency/anon/anon__warm.png")
    img_b = Path("D:/SVRPTW/webui/static/snapshots/consistency/anon/anon__pyvrp.png")
    img_a_url = "/snapshots/consistency/anon/anon__warm.png"
    img_b_url = "/snapshots/consistency/anon/anon__pyvrp.png"

    iid = "Manhattan-N50-anon"
    ui.push_stage("anon-consistency", "lightest 3 models on truly-anonymized pair (no title/legend/attribution)")
    ui.push_agent("anon-consistency", "running",
                   summary="3 models x 3 reps on stripped-identifier images")

    models = LIGHTEST_FIRST[:3]   # 12B, 26B, 30B
    REPS = 3
    summary = []
    for model in models:
        pair_id = f"{iid}__{model['weight_class']}"
        calls = []
        ui.push_log(f"[anon] {pair_id} starting")
        for r in range(REPS):
            v = _call_one(model, img_a, img_b, run_index=r)
            calls.append(v)
            _push_panel(pair_id, model["id"], calls, img_a_url, img_b_url)
            if v.get("error"):
                ui.push_log(f"  r{r}: ERR {v['error'][:80]}")
            else:
                ui.push_log(f"  r{r}: score={v['score']:.2f}  rationale={v.get('rationale','')[:60]}")
        valid = [c["score"] for c in calls if c.get("score") is not None]
        if len(valid) > 1:
            mu = statistics.mean(valid); sd = statistics.stdev(valid)
            summary.append((model["weight_class"], mu, sd, len(valid)))
            print(f"  {model['weight_class']:12s}  mean={mu:.3f}  sd={sd:.3f}  n={len(valid)}")
        elif valid:
            summary.append((model["weight_class"], valid[0], 0.0, 1))
        else:
            summary.append((model["weight_class"], None, None, 0))
            print(f"  {model['weight_class']:12s}  ALL FAILED")

    ui.push_agent("anon-consistency", "completed",
                   summary=f"3 models tested on anon pair; see judge panel")
    print("\n=== ANON RESULTS ===")
    for wc, mu, sd, n in summary:
        if mu is None:
            print(f"  {wc:12s}: all failed")
        else:
            print(f"  {wc:12s}: mean={mu:.3f}  sd={(sd or 0):.3f}  n={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
