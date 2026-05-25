"""Single-image absolute-quality consistency on Qwen3-VL-4B (LM Studio).

Hypothesis test: does Qwen-4B produce a stable absolute quality score on
ONE image (not pair comparison)? If yes, the visual judge can score
solutions standalone, not just relatively. This unlocks scoring during
solver runs without requiring a baseline.
"""
from __future__ import annotations
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "D:/SVRPTW")
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")

from svrptw.logic.lmstudio_client import call_model
from webui import client as ui


_PROMPT = """You are evaluating ONE vehicle-routing solution. The image shows
routes (each color = one vehicle) overlaid on a real city map.

Score the absolute visual quality on a 0..1 scale where:
  0.0 = very poor (many crossings, spaghetti, unbalanced)
  0.5 = mediocre (typical baseline solver output)
  1.0 = excellent (compact non-overlapping clusters, balanced loads,
                    routes follow streets sensibly)

Judge by visual structure only:
  - Compact clusters per color (vehicle), few crossings between colors
  - Routes don't double-back; visit sequences look efficient
  - Spatial coverage looks sensible

Return JSON exactly: {"score": number 0..1, "rationale": "2 sentences"}"""


def main() -> int:
    img = Path("D:/SVRPTW/webui/static/snapshots/consistency/anon/anon__warm.png")
    if not img.exists():
        print(f"FATAL: image missing: {img}")
        return 1
    REPS = 5
    scores: list[float] = []
    print(f"=== single-image consistency on Qwen3-VL-4B (5 reps) ===")
    ui.push_agent("qwen-single-image-consistency", "running",
                   summary=f"5 reps absolute-quality scoring on warm.png")
    pair_id = "Manhattan-N50-anon__qwen-4b-single-image"
    calls = []
    for r in range(REPS):
        v = call_model("qwen/qwen3-vl-4b", _PROMPT, image_path=img)
        if v.score is None:
            print(f"  r{r}: ERR {v.error or 'no score'}")
            calls.append({"model": "qwen/qwen3-vl-4b", "kind": "lmstudio",
                          "tier": "L", "weight_class": "4B-single",
                          "run_index": r, "score": None,
                          "rationale": v.rationale, "confidence": v.confidence,
                          "latency_s": v.latency_s, "error": v.error})
        else:
            print(f"  r{r}: score={v.score:.3f}  lat={v.latency_s:.1f}s")
            scores.append(v.score)
            calls.append({"model": "qwen/qwen3-vl-4b", "kind": "lmstudio",
                          "tier": "L", "weight_class": "4B-single",
                          "run_index": r, "score": v.score,
                          "rationale": v.rationale, "confidence": v.confidence,
                          "latency_s": v.latency_s, "error": None})
    if len(scores) > 1:
        mu = statistics.mean(scores); sd = statistics.stdev(scores)
        verdict = "CONSISTENT" if sd < 0.10 else "NOISY"
        print(f"\nQwen single-image: mean={mu:.3f}  sd={sd:.3f}  n={len(scores)}/{REPS}  -> {verdict}")
        ui.push_agent("qwen-single-image-consistency", "completed",
                       summary=f"single-image: mean={mu:.3f} sd={sd:.3f} n={len(scores)}/{REPS} {verdict}")
        # also push to judges panel for visibility in the dashboard
        from bench.scripts.judge_consistency import _push_panel
        _push_panel(pair_id, "qwen/qwen3-vl-4b (single-image)", calls,
                     "/snapshots/consistency/anon/anon__warm.png", "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
