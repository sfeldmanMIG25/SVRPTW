"""Bench orchestrator with explicit per-task timings.

Coordinates the full multi-solver × multi-N shootout, capturing
explicit timing for each (solver, N) cell. Designed to be called by
sub-agents that each handle one N slice in parallel — but also runs
serially as a single process if invoked directly.

Output: ``harness/reports/orchestrator_run.json`` + per-N JSONs
consumable by the dashboard renderer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from examples_openvrp.bench_all_solvers import (   # noqa: E402
    SOLVERS,
    build_instance,
    measure,
)


@dataclass
class TaskTiming:
    """Explicit per-task wall + breakdown."""
    task_id: str
    n: int
    seed: int
    solver: str
    wall_s: float
    peak_rss_mb: float
    ok: bool
    quality: dict = field(default_factory=dict)
    error: str = ""


def run_slice(n: int, seed: int, budget: float,
              solvers: list[str]) -> list[TaskTiming]:
    """Run every solver on one (n, seed) instance; return explicit timings."""
    out: list[TaskTiming] = []
    b = build_instance(n, seed)
    for s in solvers:
        if s not in SOLVERS:
            continue
        m = measure(s, SOLVERS[s], b, budget)
        out.append(TaskTiming(
            task_id=f"N{n}-s{seed}-{s}",
            n=n, seed=seed, solver=s,
            wall_s=m.wall_s, peak_rss_mb=m.peak_rss_mb,
            ok=m.ok, quality=m.quality, error=m.error,
        ))
    return out


def write_slice_json(n: int, results: list[TaskTiming], dest_dir: Path) -> Path:
    """Persist one N slice's results to JSON for later aggregation."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"slice_n{n}.json"
    path.write_text(json.dumps(
        {"n": n, "results": [asdict(r) for r in results],
         "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        indent=2,
    ), encoding="utf-8")
    return path


def aggregate(dest_dir: Path, n_values: list[int]) -> Path:
    """Roll up every slice JSON into one orchestrator-level JSON."""
    all_results = []
    slice_timings = {}
    for n in n_values:
        p = dest_dir / f"slice_n{n}.json"
        if not p.exists():
            continue
        blob = json.loads(p.read_text(encoding="utf-8"))
        for r in blob["results"]:
            all_results.append(r)
        slice_timings[n] = blob.get("generated_at")
    agg_path = dest_dir / "orchestrator_run.json"
    agg_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_values": n_values,
        "slices": slice_timings,
        "results": all_results,
    }, indent=2), encoding="utf-8")
    return agg_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[50, 100, 250])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--solvers", nargs="+", default=list(SOLVERS.keys()))
    ap.add_argument("--mode", choices=["serial", "slice"], default="serial",
                    help="serial=run all in process; slice=run ONE N value "
                         "(used by sub-agents).")
    ap.add_argument("--out-dir", default="harness/reports")
    args = ap.parse_args()

    dest = Path(args.out_dir)
    print(f"=== orchestrator: mode={args.mode} n={args.n} seed={args.seed} "
          f"budget={args.budget}s solvers={args.solvers} ===")

    if args.mode == "serial":
        for n in args.n:
            t0 = time.perf_counter()
            print(f"\n--- slice N={n} ---")
            results = run_slice(n, args.seed, args.budget, args.solvers)
            for r in results:
                q = r.quality
                tag = "OK " if r.ok else "FAIL"
                print(f"  [{tag}] {r.task_id:34s} wall={r.wall_s:6.2f}s  "
                      f"rss={r.peak_rss_mb:6.1f}MB  "
                      f"K={int(q.get('n_routes', 0)):>3}  "
                      f"obj=${q.get('objective', 0):>9.2f}  "
                      f"xings={int(q.get('route_crossings', 0)):>3}  "
                      f"feas={'Y' if q.get('feasible') else 'n'}")
            write_slice_json(n, results, dest)
            print(f"  slice wall: {time.perf_counter() - t0:.2f}s")
        agg = aggregate(dest, args.n)
        print(f"\n[wrote] {agg}")
    elif args.mode == "slice":
        # Sub-agent mode — exactly one N value
        if len(args.n) != 1:
            print("--mode=slice requires exactly one --n value", file=sys.stderr)
            sys.exit(2)
        n = args.n[0]
        t0 = time.perf_counter()
        results = run_slice(n, args.seed, args.budget, args.solvers)
        path = write_slice_json(n, results, dest)
        wall = time.perf_counter() - t0
        ok_count = sum(1 for r in results if r.ok)
        print(f"slice N={n}: {ok_count}/{len(results)} ok  wall={wall:.2f}s")
        print(f"[wrote] {path}")


if __name__ == "__main__":
    main()
