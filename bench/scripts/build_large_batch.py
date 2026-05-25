"""Bulk-generate large realistic VRPTW instances.

For each (city, N, rep) combination, calls the build_large_instance
generator unless the target file already exists. Skip-on-exist makes
this safe to re-run after a partial failure.

Live progress is pushed to the webui dashboard via webui.client.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "D:/SVRPTW")
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")

from webui import client as ui  # noqa: E402


def _instance_path(out_dir: Path, city: str, n: int, rep: str) -> Path:
    iid = f"OSM-{city}-N{n:04d}-{rep}"
    return out_dir / f"{iid}.json"


def _run_one(city: str, n: int, rep: str, out_dir: Path,
             vehicle_capacity: int, day_start: int, day_end: int,
             seed: int) -> tuple[bool, float, str]:
    """Generate a single instance via build_large_instance.main().

    Returns (ok, elapsed_s, msg). Skips (returns ok=True quickly) if
    the JSON file already exists on disk.
    """
    target = _instance_path(out_dir, city, n, rep)
    if target.exists():
        return True, 0.0, f"skip (exists): {target.name}"
    # Inject argv for the inner script's argparse. Import via importlib
    # since bench/scripts is not a proper package (no __init__.py).
    import importlib.util
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "build_large_instance", here / "build_large_instance.py"
    )
    bli = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(bli)
    saved_argv = sys.argv[:]
    sys.argv = [
        "build_large_instance.py",
        "--city", city,
        "--N", str(n),
        "--rep", rep,
        "--out-dir", str(out_dir),
        "--vehicle-capacity", str(vehicle_capacity),
        "--day-start", str(day_start),
        "--day-end", str(day_end),
        "--seed", str(seed),
    ]
    t0 = time.perf_counter()
    try:
        rc = bli.main()
        ok = rc == 0
    except SystemExit as e:
        ok = (e.code in (0, None))
    except Exception as ex:
        sys.argv = saved_argv
        return False, time.perf_counter() - t0, f"FAIL {type(ex).__name__}: {ex}"
    finally:
        sys.argv = saved_argv
    elapsed = time.perf_counter() - t0
    return ok, elapsed, f"wrote {target.name} in {elapsed:.0f}s" if ok else "FAIL"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cities", nargs="+", required=True,
                   help="city names known to build_large_instance._CITY_BBOX")
    p.add_argument("--N", type=int, nargs="+", required=True,
                   help="customer counts to generate per city")
    p.add_argument("--reps", nargs="+", default=["I000"],
                   help="rep ids (e.g. I000 I001)")
    p.add_argument("--out-dir", default="instances/v1_large")
    p.add_argument("--vehicle-capacity", type=int, default=200)
    p.add_argument("--day-start", type=int, default=480)
    p.add_argument("--day-end", type=int, default=960)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-cap-min", type=float, default=None,
                   help="stop early if total wall exceeds this many minutes")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combos = [(c, n, r) for c in args.cities for n in args.N for r in args.reps]
    total = len(combos)
    print(f"[batch] {total} instance combos: cities={args.cities} N={args.N} reps={args.reps}")
    ui.push_agent("build-batch", "running",
                  summary=f"queued {total} instances", progress=0.0)
    t_start = time.perf_counter()
    done = 0; ok_n = 0; skip_n = 0; fail_n = 0
    results: list[tuple[str, bool, float, str]] = []
    for i, (city, n, rep) in enumerate(combos, start=1):
        iid = f"OSM-{city}-N{n:04d}-{rep}"
        elapsed_total = time.perf_counter() - t_start
        if args.time_cap_min is not None and elapsed_total / 60.0 >= args.time_cap_min:
            print(f"[batch] time cap reached ({elapsed_total/60:.1f} min); stopping early")
            ui.push_log(f"[batch] time cap hit; stopping with {done}/{total} done")
            break
        ui.push_agent("build-batch", "running",
                      summary=f"[{i}/{total}] {iid}",
                      progress=(i - 1) / total)
        ui.push_log(f"[batch] start {iid}")
        ok, elapsed, msg = _run_one(
            city=city, n=n, rep=rep, out_dir=out_dir,
            vehicle_capacity=args.vehicle_capacity,
            day_start=args.day_start, day_end=args.day_end,
            seed=args.seed,
        )
        results.append((iid, ok, elapsed, msg))
        done += 1
        if msg.startswith("skip"):
            skip_n += 1
        elif ok:
            ok_n += 1
        else:
            fail_n += 1
        print(f"[batch] [{i}/{total}] {msg}")
        ui.push_log(f"[batch] done {iid}: {msg}")
        ui.push_progress("build-batch", done, total)

    elapsed_total = time.perf_counter() - t_start
    summary = f"{ok_n} new, {skip_n} skipped, {fail_n} failed in {elapsed_total/60:.1f} min"
    print(f"[batch] DONE: {summary}")
    for iid, ok, elapsed, msg in results:
        print(f"  {iid}: {msg}")
    ui.push_agent("build-batch",
                  "completed" if fail_n == 0 else "failed",
                  summary=summary, progress=1.0)
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
