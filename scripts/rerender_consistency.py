"""Re-render the consistency pair with the new contextily-based renderer."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "D:/SVRPTW")
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")

from svrptw.io import load_instance
from svrptw.config import Settings
from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto
from svrptw.solvers.classical import pyvrp_solver as pv
from svrptw.viz.renderer import render_llm_compare


def main(inst_path: str = "instances/v1/OSM-Manhattan-N100-I003.json",
          warm_b: float = 8.0, pyvrp_b: float = 15.0) -> int:
    inst = load_instance(inst_path)
    cost_key = "operational_cost"
    routes_key = "num_vehicles_used"
    print(f"rendering pair for {inst.instance_id}")
    warm = solve_auto(inst, Settings(), budget_seconds=warm_b)
    pyvrp = pv.solve(inst, Settings(), budget_seconds=pyvrp_b)
    wc = float(warm.metrics[cost_key]); wk = int(warm.metrics[routes_key])
    pc = float(pyvrp.metrics[cost_key]); pk = int(pyvrp.metrics[routes_key])
    print(f"  warm  cost={wc:.1f}, K={wk}")
    print(f"  pyvrp cost={pc:.1f}, K={pk}")

    out_dir = Path("D:/SVRPTW/webui/static/snapshots/consistency")
    out_dir.mkdir(parents=True, exist_ok=True)
    a = render_llm_compare(inst, warm, out_dir / f"{inst.instance_id}__warm.png",
                            title=f"A: solve_auto  cost={wc:.0f}  K={wk}")
    b = render_llm_compare(inst, pyvrp, out_dir / f"{inst.instance_id}__pyvrp.png",
                            title=f"B: PyVRP  cost={pc:.0f}  K={pk}")
    print(f"wrote {a.name} ({a.stat().st_size} bytes)")
    print(f"wrote {b.name} ({b.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    import sys as _sys
    _p = _sys.argv[1] if len(_sys.argv) > 1 else "instances/v1/OSM-Manhattan-N100-I003.json"
    _wb = float(_sys.argv[2]) if len(_sys.argv) > 2 else 8.0
    _pb = float(_sys.argv[3]) if len(_sys.argv) > 3 else 15.0
    raise SystemExit(main(_p, _wb, _pb))
