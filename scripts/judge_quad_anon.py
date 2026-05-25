"""4-judge consistency on the vehicle-id-colored anon pair.

Models: Qwen-4B local + Gemini-2.0-flash-lite + Gemini-2.5-flash-lite + Gemini-2.0-flash.
3 reps each = 12 calls. Streams to dash.
"""
from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "D:/SVRPTW")
_env = Path("D:/SVRPTW/.env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")

from bench.scripts.judge_consistency import LIGHTEST_FIRST, _call_one, _push_panel
from webui import client as ui


def main() -> int:
    img_a = Path("D:/SVRPTW/webui/static/snapshots/consistency/anon/anon__warm.png")
    img_b = Path("D:/SVRPTW/webui/static/snapshots/consistency/anon/anon__pyvrp.png")
    img_a_url = "/snapshots/consistency/anon/anon__warm.png"
    img_b_url = "/snapshots/consistency/anon/anon__pyvrp.png"
    # 4 judges: local Qwen + 3 lightest Gemini
    models = LIGHTEST_FIRST[:4]
    REPS = 3
    print(f"=== quad-judge consistency: {len(models)} models x {REPS} reps ===")
    ui.push_agent("quad-anon-consistency", "running",
                   summary=f"{len(models)} judges (Qwen-local + 3 Gemini-flash variants) x {REPS} reps")

    table: list[tuple[str, list[float], int, int]] = []
    for model in models:
        wc = model["weight_class"]
        pair_id = f"Manhattan-N50-anon__{wc}"
        calls = []
        print(f"\n-- {model['id']} ({model['kind']}) --")
        for r in range(REPS):
            v = _call_one(model, img_a, img_b, run_index=r)
            calls.append(v)
            _push_panel(pair_id, model["id"], calls, img_a_url, img_b_url)
            if v.get("error"):
                err_short = (v.get("error") or "")[:80]
                print(f"  r{r}: ERR {err_short}")
                ui.push_log(f"  {wc} r{r}: ERR {err_short}")
            else:
                print(f"  r{r}: score={v['score']:.3f}  lat={v['latency_s']:.1f}s")
                ui.push_log(f"  {wc} r{r}: score={v['score']:.3f}  lat={v['latency_s']:.1f}s")
        valid = [c["score"] for c in calls if c.get("score") is not None]
        table.append((wc, valid, len(valid), REPS))

    print("\n=== ROLLUP ===")
    print(f"  {'model':22s} | {'mean':>6s} {'sd':>6s} {'n':>4s} | verdict")
    for wc, valid, n_ok, n_tot in table:
        if len(valid) > 1:
            mu = statistics.mean(valid); sd = statistics.stdev(valid)
            verdict = "CONSISTENT" if sd < 0.10 else "NOISY"
            print(f"  {wc:22s} | {mu:>6.3f} {sd:>6.3f} {n_ok:>2d}/{n_tot} | {verdict}")
        elif valid:
            print(f"  {wc:22s} | {valid[0]:>6.3f} {'-':>6s} {n_ok:>2d}/{n_tot} | single")
        else:
            print(f"  {wc:22s} | {'-':>6s} {'-':>6s} {n_ok:>2d}/{n_tot} | ALL FAILED")

    ui.push_agent("quad-anon-consistency", "completed",
                   summary=f"see judge panel; {len(table)} models tested")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
