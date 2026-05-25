"""Side-by-side comparison of every available solver on the same
instances: timing, peak memory, and quality measured by a shared
evaluator (openvrp's cost model + the 8-metric quality catalog).

Solvers wired in:
  - openvrp_native     — openvrp.solve_od construction='fast' (default, PyVRP-free)
  - openvrp_pyvrp      — openvrp.solve_od construction='pyvrp' (routes through svrptw bandit)
  - pyvrp_standalone   — direct PyVRP HGS construction (no bandit)
  - ortools            — Google OR-Tools VRP solver
  - svrptw_greedy      — nearest-neighbor baseline

Output: stdout table + JSON file + HTML report.

Run:
  PYTHONPATH=D:/SVRPTW python examples_openvrp/bench_all_solvers.py
  PYTHONPATH=D:/SVRPTW python examples_openvrp/bench_all_solvers.py --n 100 250 500 --budget 60 --seeds 1
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil

from openvrp import (
    Depot,
    SolveOptions,
    Stop,
    TimeWindow,
    VehicleClass,
    solve_od,
)


# ============================================================
# Instance generation (deterministic; matches bench_n1000.py)
# ============================================================


def build_instance(n: int, seed: int) -> dict[str, Any]:
    """Build a deterministic OD-mode instance bundle reusable by every
    solver wrapper. Returns a dict carrying T, D, depots, stops, fleet,
    plus svrptw-Instance fields (n, capacity, depot_ready/due, customers).
    """
    rng = np.random.default_rng(seed)
    coords = [(0.0, 0.0)]
    for _ in range(n):
        coords.append((float(rng.uniform(-50, 50)), float(rng.uniform(-50, 50))))
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    base = np.hypot(dx, dy)
    asym = 1.0 + rng.random((n + 1, n + 1)) * 0.3
    T = (base * asym) * 60.0   # seconds
    np.fill_diagonal(T, 0)
    D = T.copy()               # meters proxy
    demand_arr = rng.integers(1, 6, size=n).astype(np.int64)
    depots = [Depot(id="D0", node_index=0)]
    stops = [
        Stop(id=f"S{i}", node_index=i, demand={"weight": float(demand_arr[i - 1])},
             service_seconds=300.0,
             time_windows=[TimeWindow(earliest=0.0, latest=8 * 3600)])
        for i in range(1, n + 1)
    ]
    fleet = [VehicleClass(id="van", count=None, capacity={"weight": 100.0},
                          home_depot_id="D0",
                          cost_per_second=0.005, cost_per_meter=0.001,
                          fixed_cost=20.0)]
    return {"T": T, "D": D, "n": n, "demand": demand_arr,
            "depots": depots, "stops": stops, "fleet": fleet, "seed": seed}


def build_svrptw_instance(b: dict[str, Any]) -> Any:
    """Adapter: openvrp instance bundle → svrptw.io.Instance.

    svrptw uses MINUTES for time and MILES for distance internally.
    """
    from svrptw.io.instance import Customer, Depot as SVDepot, Instance
    T_min = b["T"] / 60.0
    D_mi = b["D"] / 1609.344
    customers = [
        Customer(id=i + 1, node_id=i + 1, x=0.0, y=0.0,
                 demand=int(b["demand"][i]),
                 ready=0, due=480, service=5)
        for i in range(b["n"])
    ]
    depot = SVDepot(node_id=0, x=0.0, y=0.0, ready=0, due=24 * 60)
    return Instance(
        instance_id=f"bench-N{b['n']}-s{b['seed']}",
        city="bench",
        num_customers=b["n"], num_vehicles=max(1, b["n"]),
        vehicle_capacity=100, depot=depot, customers=customers,
        travel_time=T_min, travel_dist=D_mi,
        asymmetry_score=0.0, seed=b["seed"],
    )


# ============================================================
# Shared cost evaluator (apples-to-apples)
# ============================================================


def shared_evaluate(routes: list[list[int]], b: dict[str, Any]) -> dict[str, float]:
    """Compute openvrp-style operational cost on a list of route stop-id
    sequences (each is a list of 1..n customer matrix-indices, no depot).

    Uses the SAME cost model the openvrp adapter applies, so every
    solver's output is comparable.
    """
    T = b["T"]; D = b["D"]
    demand = b["demand"]
    cap = 100.0
    # SVRPTW economics (per the canonical Economics defaults)
    wage_per_min = 14.5 / 60.0           # $0.241/min
    cost_per_mile = 0.5
    hard_late_penalty = 1000.0
    fixed_cost_per_route = 20.0

    served: set[int] = set()
    total_time_min = 0.0       # minutes of travel
    total_dist_mi = 0.0        # miles
    total_wait_min = 0.0
    total_late_min = 0.0
    total_overload = 0.0
    n_routes = 0

    for seq in routes:
        if not seq:
            continue
        n_routes += 1
        load = float(sum(demand[c - 1] for c in seq))
        if load > cap:
            total_overload += load - cap
        clock_sec = 0.0
        prev = 0
        for c in seq:
            tt_sec = float(T[prev, c])
            td_mi = float(D[prev, c]) / 1609.344
            total_time_min += tt_sec / 60.0
            total_dist_mi += td_mi
            clock_sec += tt_sec
            # Customer TW (0..8h)
            if clock_sec < 0:
                wait = -clock_sec; total_wait_min += wait / 60.0
                clock_sec = 0
            if clock_sec > 8 * 3600:
                total_late_min += (clock_sec - 8 * 3600) / 60.0
            else:
                clock_sec += 300   # 5-min service
                served.add(c)
            prev = c
        # Return-to-depot
        ret_sec = float(T[prev, 0])
        total_time_min += ret_sec / 60.0
        total_dist_mi += float(D[prev, 0]) / 1609.344

    missed = b["n"] - len(served)
    cost = (wage_per_min * total_time_min
            + cost_per_mile * total_dist_mi
            + wage_per_min * total_wait_min
            + hard_late_penalty * missed
            + wage_per_min * total_late_min
            + hard_late_penalty * total_overload
            + fixed_cost_per_route * n_routes)
    # Quality metric: count inter-route segment crossings using xy coords
    # (cheap O(K^2 * L^2); fine at N<=500).
    crossings = _route_crossings(routes, b)
    util_cv = _load_cv(routes, b)
    return {
        "objective": round(cost, 2),
        "n_routes": float(n_routes),
        "served": float(len(served)),
        "missed": float(missed),
        "late_min": round(total_late_min, 1),
        "overload": round(total_overload, 1),
        "route_crossings": float(crossings),
        "load_balance_cv": round(util_cv, 3),
        "feasible": float(missed == 0 and total_late_min == 0 and total_overload == 0),
    }


def _ccw(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_cross(a, b, c, d) -> bool:
    d1 = _ccw(c, d, a); d2 = _ccw(c, d, b)
    d3 = _ccw(a, b, c); d4 = _ccw(a, b, d)
    return ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4))


def _route_crossings(routes: list[list[int]], b: dict[str, Any]) -> int:
    """Inter-route segment crossings — visual-chaos proxy."""
    rng = np.random.default_rng(b["seed"])
    coords = [(0.0, 0.0)]
    for _ in range(b["n"]):
        coords.append((float(rng.uniform(-50, 50)), float(rng.uniform(-50, 50))))
    total = 0
    seg_per_route = []
    for seq in routes:
        if not seq:
            continue
        path = [coords[0]] + [coords[c] for c in seq] + [coords[0]]
        seg_per_route.append([(path[i], path[i + 1]) for i in range(len(path) - 1)])
    for i in range(len(seg_per_route)):
        for j in range(i + 1, len(seg_per_route)):
            for s1 in seg_per_route[i]:
                for s2 in seg_per_route[j]:
                    if _seg_cross(s1[0], s1[1], s2[0], s2[1]):
                        total += 1
    return total


def _load_cv(routes: list[list[int]], b: dict[str, Any]) -> float:
    loads = [sum(b["demand"][c - 1] for c in seq) for seq in routes if seq]
    if len(loads) < 2:
        return 0.0
    mu = sum(loads) / len(loads)
    if mu == 0:
        return 0.0
    var = sum((x - mu) ** 2 for x in loads) / len(loads)
    return (var ** 0.5) / mu


# ============================================================
# Per-solver adapters — each returns list[list[int]] of customer ids
# ============================================================


def _extract_openvrp_routes(sol) -> list[list[int]]:
    if sol.status == "infeasible" and not sol.routes:
        # Surface the failure reason so we can debug rather than silently
        # reporting an empty Solution as a "result."
        raise RuntimeError("openvrp returned infeasible: "
                           + "; ".join(sol.diagnostics.feasibility_blockers[:2]))
    return [[int(v.stop_id.lstrip("S")) for v in r.visits if v.kind != "depot"]
            for r in sol.routes]


def run_openvrp_native(b: dict, budget: float) -> list[list[int]]:
    sol = solve_od(b["T"], b["stops"], b["depots"], b["fleet"],
                   options=SolveOptions(budget_seconds=budget, seed=b["seed"],
                                        construction="fast"),
                   distance_matrix=b["D"])
    return _extract_openvrp_routes(sol)


def run_openvrp_pyvrp(b: dict, budget: float) -> list[list[int]]:
    sol = solve_od(b["T"], b["stops"], b["depots"], b["fleet"],
                   options=SolveOptions(budget_seconds=budget, seed=b["seed"],
                                        construction="pyvrp"),
                   distance_matrix=b["D"])
    return _extract_openvrp_routes(sol)


def run_pyvrp_standalone(b: dict, budget: float) -> list[list[int]]:
    from svrptw.config import Settings
    from svrptw.solvers.classical import pyvrp_solver as pv
    inst = build_svrptw_instance(b)
    sv_sol = pv.solve(inst, Settings(), budget_seconds=budget)
    return [list(r.customers) for r in sv_sol.routes if r.customers]


def run_ortools(b: dict, budget: float) -> list[list[int]]:
    from svrptw.config import Settings
    from svrptw.solvers.classical import ortools_solver as ot
    inst = build_svrptw_instance(b)
    sv_sol = ot.solve(inst, Settings(), budget_seconds=budget)
    return [list(r.customers) for r in sv_sol.routes if r.customers]


def run_svrptw_greedy(b: dict, budget: float) -> list[list[int]]:
    from svrptw.config import Settings
    from svrptw.solvers.classical import greedy as gr
    inst = build_svrptw_instance(b)
    sv_sol = gr.solve(inst, Settings())
    return [list(r.customers) for r in sv_sol.routes if r.customers]


SOLVERS: dict[str, Callable[[dict, float], list[list[int]]]] = {
    "openvrp_native":    run_openvrp_native,
    "openvrp_pyvrp":     run_openvrp_pyvrp,
    "pyvrp_standalone":  run_pyvrp_standalone,
    "ortools":           run_ortools,
    "svrptw_greedy":     run_svrptw_greedy,
}


# ============================================================
# Measurement: wall + peak RSS (background sampler)
# ============================================================


@dataclass
class Measurement:
    solver: str
    n: int
    seed: int
    budget_s: float
    wall_s: float
    peak_rss_mb: float
    delta_rss_mb: float
    ok: bool
    error: str = ""
    quality: dict[str, float] = field(default_factory=dict)


class _RSSSampler(threading.Thread):
    """Background thread that samples RSS every 100 ms; reports peak.

    Note: don't name the stop-event ``_stop`` — that shadows
    ``threading.Thread._stop`` which the runtime uses internally.
    """
    daemon = True

    def __init__(self, proc: psutil.Process):
        super().__init__()
        self.proc = proc
        self.peak = proc.memory_info().rss
        self._sample_stop = threading.Event()

    def run(self) -> None:
        while not self._sample_stop.is_set():
            try:
                rss = self.proc.memory_info().rss
                if rss > self.peak:
                    self.peak = rss
            except Exception:
                pass
            self._sample_stop.wait(0.1)

    def stop_sampling(self) -> int:
        self._sample_stop.set()
        self.join(timeout=1.0)
        return self.peak


def measure(solver_name: str, fn: Callable, b: dict, budget: float) -> Measurement:
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    sampler = _RSSSampler(proc)
    sampler.start()
    t0 = time.perf_counter()
    routes: list[list[int]] = []
    err = ""
    ok = True
    try:
        routes = fn(b, budget)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        ok = False
    wall = time.perf_counter() - t0
    peak = sampler.stop_sampling()
    quality = shared_evaluate(routes, b) if ok else {}
    return Measurement(
        solver=solver_name, n=b["n"], seed=b["seed"], budget_s=budget,
        wall_s=round(wall, 3),
        peak_rss_mb=round(peak / 1024 / 1024, 1),
        delta_rss_mb=round((peak - rss_before) / 1024 / 1024, 1),
        ok=ok, error=err, quality=quality,
    )


# ============================================================
# Bench runner
# ============================================================


def run_sweep(n_values: list[int], budget: float, seeds: int,
              solvers_to_run: list[str]) -> list[Measurement]:
    results: list[Measurement] = []
    for n in n_values:
        for seed in range(seeds):
            b = build_instance(n, seed)
            for s in solvers_to_run:
                fn = SOLVERS[s]
                m = measure(s, fn, b, budget)
                results.append(m)
                _print_row(m)
    return results


def _print_row(m: Measurement) -> None:
    q = m.quality
    if m.ok:
        print(f"  {m.solver:18s}  N={m.n:>4}  s{m.seed}  "
              f"wall={m.wall_s:6.2f}s  rss={m.peak_rss_mb:7.1f}MB  "
              f"K={int(q.get('n_routes', 0)):>3}  "
              f"obj=${q.get('objective', 0):>9.2f}  "
              f"feas={'Y' if q.get('feasible') else 'n'}  "
              f"miss={int(q.get('missed', 0))}  "
              f"xings={int(q.get('route_crossings', 0))}")
    else:
        print(f"  {m.solver:18s}  N={m.n:>4}  s{m.seed}  "
              f"wall={m.wall_s:6.2f}s  rss={m.peak_rss_mb:7.1f}MB  "
              f"FAIL: {m.error[:80]}")


# ============================================================
# Output: JSON + HTML
# ============================================================


def write_json(results: list[Measurement], path: Path) -> None:
    path.write_text(json.dumps(
        {"results": [asdict(m) for m in results],
         "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        indent=2,
    ), encoding="utf-8")


def write_html(results: list[Measurement], path: Path) -> None:
    by_solver: dict[str, list[Measurement]] = {}
    for m in results:
        by_solver.setdefault(m.solver, []).append(m)
    n_values = sorted({m.n for m in results})

    def mean(xs: list[float]) -> float:
        return sum(xs) / max(1, len(xs))

    rows_html: list[str] = []
    for solver in sorted(by_solver.keys()):
        for n in n_values:
            ms = [m for m in by_solver[solver] if m.n == n]
            if not ms:
                continue
            ok = all(m.ok for m in ms)
            walls = [m.wall_s for m in ms]
            rsses = [m.peak_rss_mb for m in ms]
            objs = [m.quality.get("objective", float("nan")) for m in ms if m.ok]
            ks = [m.quality.get("n_routes", float("nan")) for m in ms if m.ok]
            feas = [m.quality.get("feasible", 0) for m in ms if m.ok]
            xings = [m.quality.get("route_crossings", 0) for m in ms if m.ok]
            miss = [m.quality.get("missed", 0) for m in ms if m.ok]
            wall_disp = f"{mean(walls):.2f}s"
            rss_disp = f"{mean(rsses):.0f}MB"
            obj_disp = (f"${mean(objs):.0f}" if objs and not any(o != o for o in objs)
                        else "—")
            k_disp = f"{mean(ks):.0f}" if ks else "—"
            feas_disp = ("Y" if all(f == 1 for f in feas) else
                         ("n" if feas else "—"))
            xing_disp = f"{mean(xings):.0f}" if xings else "—"
            miss_disp = f"{int(mean(miss))}" if miss else "—"
            status_cls = "ok" if ok else "bad"
            status_pill = ("ok" if ok and feas_disp == "Y" else
                           ("warn" if ok else "bad"))
            rows_html.append(
                f"<tr>"
                f"<td>{solver}</td><td class='num'>{n}</td>"
                f"<td class='num mono'>{wall_disp}</td>"
                f"<td class='num mono'>{rss_disp}</td>"
                f"<td class='num'>{k_disp}</td>"
                f"<td class='num mono'>{obj_disp}</td>"
                f"<td class='num'>{xing_disp}</td>"
                f"<td class='num'>{miss_disp}</td>"
                f"<td><span class='pill {status_pill}'>{feas_disp}</span></td>"
                f"</tr>"
            )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>OpenVRP — all-solvers comparison</title>
<style>
:root {{ --bg: #0f1115; --fg: #e6edf3; --dim: #8b949e; --accent: #7ee787;
  --warn: #f0883e; --bad: #f85149; --card: #161b22; --border: #30363d; --muted: #21262d; }}
* {{ box-sizing: border-box; }}
html, body {{ background: var(--bg); color: var(--fg); margin: 0;
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding: 28px 22px 60px; }}
h1 {{ font-size: 22px; margin: 0 0 6px; }}
p.lede {{ color: var(--dim); margin: 0 0 18px; font-size: 13px; }}
h2 {{ font-size: 14px; margin: 22px 0 8px; color: var(--dim);
  text-transform: uppercase; letter-spacing: .06em; }}
table {{ width: 100%; border-collapse: collapse;
  background: var(--card); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }}
th, td {{ padding: 7px 11px; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ background: var(--muted); color: var(--dim); font-weight: 500;
  text-transform: uppercase; font-size: 11px; letter-spacing: .06em; }}
tr:last-child td {{ border-bottom: none; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.mono, .mono {{ font-family: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
  font-size: 12.5px; }}
.pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px;
  font-weight: 500; letter-spacing: .02em; }}
.pill.ok   {{ background: rgba(126,231,135,.12); color: var(--accent); }}
.pill.warn {{ background: rgba(240,136,62,.12); color: var(--warn); }}
.pill.bad  {{ background: rgba(248,81,73,.12); color: var(--bad); }}
.notes {{ color: var(--dim); font-size: 12px; margin-top: 8px; }}
footer {{ margin-top: 40px; color: var(--dim); font-size: 11px; text-align: center; }}
</style></head><body><div class="wrap">

<h1>OpenVRP — all-solvers comparison</h1>
<p class="lede">Same deterministic asymmetric Euclidean instances; same
budget; same shared cost evaluator. Means across {len({m.seed for m in results})}
seed(s). Generated {time.strftime("%Y-%m-%d %H:%M")}.</p>

<h2>Results (mean per solver × N)</h2>
<table>
<tr>
  <th>Solver</th><th class="num">N</th>
  <th class="num">Wall</th><th class="num">Peak RSS</th>
  <th class="num">K</th><th class="num">Objective</th>
  <th class="num">Crossings</th><th class="num">Missed</th><th>Feas</th>
</tr>
{''.join(rows_html)}
</table>
<p class="notes">All solvers measured with the same shared evaluator
(openvrp cost model + the geometric quality metrics) so the objective
column is apples-to-apples. <code>K</code> is route count;
<code>Crossings</code> is inter-route segment intersections (visual-chaos
proxy); <code>Missed</code> is unserved customers. <code>Feas Y</code>
means all customers served on time within capacity.</p>

<footer>Generated by examples_openvrp/bench_all_solvers.py · raw data in
docs/openvrp/bench_all_solvers.json</footer>
</div></body></html>
"""
    path.write_text(html, encoding="utf-8")


# ============================================================
# Main
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[50, 100, 250])
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--solvers", nargs="+", default=list(SOLVERS.keys()))
    ap.add_argument("--out-json",
                    default="docs/openvrp/bench_all_solvers.json")
    ap.add_argument("--out-html",
                    default="WorldsFinestVRP/solver_comparison.html")
    args = ap.parse_args()

    unknown = [s for s in args.solvers if s not in SOLVERS]
    if unknown:
        print(f"WARNING: unknown solver(s) {unknown}; available: "
              f"{sorted(SOLVERS.keys())}")
        args.solvers = [s for s in args.solvers if s in SOLVERS]

    print(f"=== bench_all_solvers: N={args.n}  budget={args.budget}s  "
          f"seeds={args.seeds}  solvers={args.solvers} ===")
    results = run_sweep(args.n, args.budget, args.seeds, args.solvers)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(results, out_json)
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    write_html(results, out_html)
    print(f"\n[wrote] {out_json}")
    print(f"[wrote] {out_html}")


if __name__ == "__main__":
    main()
