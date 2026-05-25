"""Re-render anon pair (sector colors) + probe LM Studio for vision models."""
from __future__ import annotations
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, "D:/SVRPTW")
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")

from svrptw.io import load_instance
from svrptw.config import Settings
from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto
from svrptw.solvers.classical import pyvrp_solver as pv
from svrptw.viz.renderer import render_llm_compare


def main() -> int:
    inst = load_instance("instances/v1/OSM-Manhattan-N050-I003.json")
    warm = solve_auto(inst, Settings(), budget_seconds=4)
    pyvrp = pv.solve(inst, Settings(), budget_seconds=10)
    out_dir = Path("D:/SVRPTW/webui/static/snapshots/consistency/anon")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap: dict = {}
    render_llm_compare(inst, warm, out_dir / "anon__warm.png",
                        overlay_mode="minimal", route_color_map=cmap)
    render_llm_compare(inst, pyvrp, out_dir / "anon__pyvrp.png",
                        overlay_mode="minimal", route_color_map=cmap)
    print("rendered anon pair with sector-based colors; cmap size:", len(cmap))

    print()
    print("LM Studio probe @ http://127.0.0.1:1234/v1/models ...")
    try:
        with urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=3) as r:
            data = json.load(r)
            models = data.get("data", [])
            print(f"  found {len(models)} models loaded:")
            for m in models:
                mid = m.get("id", "?")
                print(f"  - {mid}")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        print(f"  not reachable: {e}")
    except Exception as e:
        print(f"  unexpected: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
