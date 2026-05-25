"""Phase D5 — paired-seed A/B between LinUCB and the trained MLP policy.

For each instance in the configured set, runs `solve_auto` twice:
  - LinUCB (the current default)
  - MLP policy with rl_neighborhood=True + reward shaping on
Same seed each pair; reports the per-instance delta and aggregate stats.

Usage:
    python bench/scripts/rl_vs_linucb_ab.py
    python bench/scripts/rl_vs_linucb_ab.py --rl-artifact bench/runs/policy_v1.pt
    python bench/scripts/rl_vs_linucb_ab.py --webui
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from glob import glob
from pathlib import Path

# Bootstrap import path for `python bench/scripts/rl_vs_linucb_ab.py`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio_pyvrp_warm as pwm


def _resolve_paths(repo_root: Path, pattern: str | None,
                   instances: list[str] | None) -> list[Path]:
    if instances:
        return [Path(p) if Path(p).is_absolute() else repo_root / p
                for p in instances]
    if pattern is None:
        # Defaults: held-out v1 N=100 + N=200 I=3, 4 instances.
        out: list[Path] = []
        for N in (100, 200):
            for I in (3, 4):
                pat = str(repo_root /
                          f"instances/v1/OSM-*-N{N}-I00{I}.json")
                out.extend(Path(p) for p in sorted(glob(pat)))
        return out
    pat = pattern if Path(pattern).is_absolute() else str(repo_root / pattern)
    return [Path(p) for p in sorted(glob(pat))]


def _maybe_webui_callback(enabled: bool, label: str):
    if not enabled:
        return None
    try:
        from svrptw.webui.client import push_snapshot  # type: ignore
    except Exception:
        # Webui module is optional; if missing, callback is a no-op.
        return None

    def _cb(op_name, improvement, ops_applied, sol):
        try:
            push_snapshot(label, op_name, float(improvement),
                          int(ops_applied), sol)
        except Exception:
            pass

    return _cb


def _run_one(inst_path: Path, settings: Settings, *, budget: float,
             seed: int, rl_artifact: str | None,
             webui_cb_a, webui_cb_b) -> dict:
    inst = load_instance(inst_path)
    sol_a = pwm.solve_auto(
        inst, settings, budget_seconds=budget,
        seed=seed,
        rl_neighborhood=False,
        on_accept=webui_cb_a,
    )
    sol_b = pwm.solve_auto(
        inst, settings, budget_seconds=budget,
        seed=seed,
        rl_neighborhood=True, rl_artifact=rl_artifact,
        on_accept=webui_cb_b,
    )
    ca = sol_a.metrics["operational_cost"]
    cb = sol_b.metrics["operational_cost"]
    return {
        "instance": inst_path.name,
        "linucb_cost": ca,
        "rl_cost": cb,
        "delta": cb - ca,                 # negative = RL won
        "rl_better": bool(cb < ca - 1e-6),
        "linucb_routes": int(sol_a.metrics["num_vehicles_used"]),
        "rl_routes": int(sol_b.metrics["num_vehicles_used"]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pattern", type=str, default=None)
    p.add_argument("--instances", nargs="*", default=None)
    p.add_argument("--budget", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rl-artifact", type=str,
                   default="bench/runs/policy_v1.pt",
                   help="MLP policy checkpoint produced by "
                   "train_policy_offline.py.")
    p.add_argument("--output", type=str,
                   default="bench/runs/rl_vs_linucb_ab.json")
    p.add_argument("--webui", action="store_true",
                   help="Push live snapshots to the web UI if available.")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    paths = _resolve_paths(repo_root, args.pattern, args.instances)
    if not paths:
        print("No instances matched.", file=sys.stderr)
        return 2
    rl_artifact = args.rl_artifact
    if not Path(rl_artifact).is_absolute():
        rl_artifact = str(repo_root / rl_artifact)
    if not Path(rl_artifact).exists():
        print(f"RL artifact not found: {rl_artifact}", file=sys.stderr)
        print("  Run bench/scripts/train_policy_offline.py first.",
              file=sys.stderr)
        return 2

    settings = Settings()
    cb_a = _maybe_webui_callback(args.webui, "linucb")
    cb_b = _maybe_webui_callback(args.webui, "rl")
    rows: list[dict] = []
    for pth in paths:
        print(f"[ab] {pth.name} ...", file=sys.stderr, flush=True)
        try:
            row = _run_one(pth, settings, budget=args.budget, seed=args.seed,
                           rl_artifact=rl_artifact,
                           webui_cb_a=cb_a, webui_cb_b=cb_b)
        except Exception as e:
            print(f"  ! {pth.name} failed: {e}", file=sys.stderr)
            continue
        rows.append(row)
        print(f"  linucb={row['linucb_cost']:.2f} rl={row['rl_cost']:.2f} "
              f"delta={row['delta']:+.2f}", file=sys.stderr)


    if not rows:
        print("No successful pairs.", file=sys.stderr)
        return 2
    deltas = [r["delta"] for r in rows]
    summary = {
        "n_pairs": len(rows),
        "rl_wins": sum(1 for r in rows if r["rl_better"]),
        "linucb_wins": sum(1 for r in rows
                           if r["delta"] > 1e-6),
        "ties": sum(1 for r in rows if abs(r["delta"]) <= 1e-6),
        "mean_delta": float(statistics.fmean(deltas)),
        "median_delta": float(statistics.median(deltas)),
        "min_delta": float(min(deltas)),
        "max_delta": float(max(deltas)),
    }
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
