"""5-rep consistency on local Qwen3-VL-4B (LM Studio) + sector-colored anon pair."""
from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "D:/SVRPTW")
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")

from bench.scripts.judge_consistency import LIGHTEST_FIRST, _call_one, _push_panel
from webui import client as ui


def main() -> int:
    img_a = Path("D:/SVRPTW/webui/static/snapshots/consistency/anon/anon__warm.png")
    img_b = Path("D:/SVRPTW/webui/static/snapshots/consistency/anon/anon__pyvrp.png")
    img_a_url = "/snapshots/consistency/anon/anon__warm.png"
    img_b_url = "/snapshots/consistency/anon/anon__pyvrp.png"
    qwen = LIGHTEST_FIRST[0]
    print(f"LM Studio model: {qwen['id']}")
    ui.push_agent("lmstudio-anon-consistency", "running",
                   summary=f"{qwen['id']} x 5 reps on anon pair (sector colors, no text)")
    calls = []
    pair_id = "Manhattan-N50-anon__qwen-4b-local"
    REPS = 5
    for r in range(REPS):
        v = _call_one(qwen, img_a, img_b, run_index=r)
        calls.append(v)
        _push_panel(pair_id, qwen["id"], calls, img_a_url, img_b_url)
        if v.get("error"):
            err_short = (v.get("error") or "")[:80]
            print(f"  r{r}: ERR {err_short}")
            ui.push_log(f"  qwen r{r}: ERR {err_short}")
        else:
            print(f"  r{r}: score={v['score']:.3f}  lat={v['latency_s']:.1f}s")
            ui.push_log(f"  qwen r{r}: score={v['score']:.3f} lat={v['latency_s']:.1f}s")
    valid = [c["score"] for c in calls if c.get("score") is not None]
    if len(valid) > 1:
        mu = statistics.mean(valid); sd = statistics.stdev(valid)
        verdict = "CONSISTENT" if sd < 0.10 else "NOISY"
        print(f"\nQwen3-VL-4B: mean={mu:.3f}  sd={sd:.3f}  n={len(valid)}/{REPS}  -> {verdict}")
        ui.push_agent("lmstudio-anon-consistency", "completed",
                       summary=f"qwen3-vl-4b: mean={mu:.3f} sd={sd:.3f} n={len(valid)}/{REPS} {verdict}")
    elif valid:
        print(f"\nQwen3-VL-4B: only 1 valid score={valid[0]:.3f}")
    else:
        print("\nQwen3-VL-4B: ALL FAILED")
        ui.push_agent("lmstudio-anon-consistency", "failed",
                       summary="all 5 calls failed; check LM Studio + model loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
