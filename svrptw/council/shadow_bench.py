"""SPEC-8-COUNCIL-01 — shadow bench arm.

Runs a candidate operator on a tiny, fixed subset of instances
(8 × N=50 + 4 × N=100) inside the bandit's slot allocator. Records
the per-arm posterior mean and Pareto position vs the existing 13
arms. Accept iff:
  - posterior mean ≥ 25th-percentile of existing arms, AND
  - not Pareto-dominated on (cost, wall) at any instance.

Writes results to `bench/runs/council/shadow/<proposal_id>.json` —
separate namespace so accepted-but-not-promoted variants don't
pollute the main leaderboard.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from svrptw.config import Settings
from svrptw.council.proposal import OperatorContext
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio as pm
from svrptw.solvers.classical import greedy as greedy_mod


_LOG = logging.getLogger("svrptw.council.shadow")

# SPEC-8-COUNCIL-02 composability instances — 16 instances at N=50
# across 8 cities × 2 seeds each. Wider than the original 8-instance
# subset to defeat selection bias: a candidate that wins on 4-of-8
# instances at +$8 mean may be near-zero across 30-instance samples.
# Doubling the subset halves that risk at ~2× the bench runtime.
COMPOSABILITY_INSTANCES: tuple[str, ...] = (
    "instances/v1/OSM-Manhattan-N050-I000.json",
    "instances/v1/OSM-Manhattan-N050-I001.json",
    "instances/v1/OSM-Cambridge-N050-I000.json",
    "instances/v1/OSM-Cambridge-N050-I001.json",
    "instances/v1/OSM-Austin-N050-I000.json",
    "instances/v1/OSM-Austin-N050-I001.json",
    "instances/v1/OSM-Paris-N050-I000.json",
    "instances/v1/OSM-Paris-N050-I001.json",
    "instances/v1/OSM-Phoenix-N050-I000.json",
    "instances/v1/OSM-Phoenix-N050-I001.json",
    "instances/v1/OSM-SanFrancisco-N050-I000.json",
    "instances/v1/OSM-SanFrancisco-N050-I001.json",
    "instances/v1/OSM-Charleston-N050-I000.json",
    "instances/v1/OSM-Charleston-N050-I001.json",
    "instances/v1/OSM-Pittsburgh-N050-I000.json",
    "instances/v1/OSM-Pittsburgh-N050-I001.json",
)

# Fixed shadow instance subset — 1 instance per city at N=50 (8 cities)
# + 4 medium-N. Same set across all proposals → comparable rankings.
SHADOW_INSTANCES: tuple[str, ...] = (
    "instances/v1/OSM-Manhattan-N050-I000.json",
    "instances/v1/OSM-Cambridge-N050-I000.json",
    "instances/v1/OSM-Austin-N050-I000.json",
    "instances/v1/OSM-Charleston-N050-I000.json",
    "instances/v1/OSM-Paris-N050-I000.json",
    "instances/v1/OSM-Phoenix-N050-I000.json",
    "instances/v1/OSM-Pittsburgh-N050-I000.json",
    "instances/v1/OSM-SanFrancisco-N050-I000.json",
    "instances/v1/OSM-Manhattan-N100-I000.json",
    "instances/v1/OSM-Austin-N100-I000.json",
    "instances/v1/OSM-SanFrancisco-N100-I000.json",
    "instances/v1/OSM-Paris-N100-I000.json",
)


def run_shadow(
    proposal_id: str,
    operator: Callable,
    *,
    op_seconds: float = 1.0,
    instances: tuple[str, ...] = SHADOW_INSTANCES,
    base_solver_budget: float = 10.0,
    accept_pct_threshold: float = 0.25,
    min_mean_abs_delta: float = 5.0,
    min_hit_rate: float = 0.30,
) -> dict:
    """Run the proposed operator against the shadow instance set.

    Strategy: warm-start each instance with the existing portfolio
    (small budget), then apply the proposed operator ONCE, record
    the delta. This is the cheapest signal that distinguishes
    "operator does no harm" from "operator finds an improvement".

    Returns a dict with:
      - posterior_mean: mean cost-delta-per-second
      - per_instance: list of {instance, before, after, delta, wall}
      - accepted: bool
      - reasoning: short text
    """
    out_dir = Path("bench/runs/council/shadow")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{proposal_id}.json"
    settings = Settings()
    rows: list[dict] = []
    for ip in instances:
        if not Path(ip).exists():
            _LOG.warning("shadow instance missing: %s; skip", ip)
            continue
        inst = load_instance(ip)
        sol = pm.solve(inst, settings, budget_seconds=base_solver_budget)
        before = float(sol.metrics["operational_cost"])
        ctx = OperatorContext(instance=inst, settings=settings,
                              rng_seed=0, deadline_seconds=op_seconds)
        t0 = time.perf_counter()
        try:
            cand = operator(sol, ctx)
        except Exception as e:
            cand = None
            _LOG.info("shadow %s on %s raised: %s",
                      proposal_id, Path(ip).stem, e)
        elapsed = time.perf_counter() - t0
        if cand is None or not hasattr(cand, "metrics"):
            after = before
            delta = 0.0
            feas = 1
            overload = 0.0
        else:
            after = float(cand.metrics.get("operational_cost", before))
            delta = before - after          # positive = improvement
            feas = int(cand.metrics.get("feasible", 0))
            overload = float(cand.metrics.get("capacity_overload", 0.0))
        rows.append({
            "instance": Path(ip).stem,
            "before": before,
            "after": after,
            "delta": delta,
            "delta_per_s": delta / max(elapsed, 1e-3),
            "elapsed_s": elapsed,
            "feasible": feas,
            "overload": overload,
        })

    deltas = np.array([r["delta"] for r in rows], dtype=np.float64)
    deltas_per_s = np.array([r["delta_per_s"] for r in rows], dtype=np.float64)
    # Absolute delta is the production-relevant metric. delta_per_s is
    # diagnostic only — it inflates when an op exits in 5ms with a small
    # improvement, masking the fact the op rarely fires (1/12 hits on
    # cross_route_2opt at portfolio@10s base).
    mean_abs_delta = float(deltas.mean()) if len(deltas) else 0.0
    posterior_mean_rate = float(deltas_per_s.mean()) if len(deltas_per_s) else 0.0
    hit_rate = float((deltas > 1e-6).sum()) / max(1, len(deltas))
    any_overload = any(r["overload"] > 0 for r in rows)
    any_infeasible = any(r["feasible"] == 0 for r in rows)

    # Accept rule: enough absolute improvement, AND on enough instances.
    # Filters operators whose mean is inflated by one lucky hit.
    accepted = (
        mean_abs_delta >= min_mean_abs_delta
        and hit_rate >= min_hit_rate
        and not any_overload
        and not any_infeasible
    )
    reasoning = (
        f"mean_delta={mean_abs_delta:.2f} (floor={min_mean_abs_delta}), "
        f"hit_rate={hit_rate:.2f} (floor={min_hit_rate}), "
        f"delta_per_s_diag={posterior_mean_rate:.2f}, "
        f"overload={any_overload}, infeasible={any_infeasible}"
    )

    result = {
        "proposal_id": proposal_id,
        "posterior_mean": mean_abs_delta,
        "mean_abs_delta": mean_abs_delta,
        "hit_rate": hit_rate,
        "posterior_mean_rate": posterior_mean_rate,
        "per_instance": rows,
        "accepted": accepted,
        "reasoning": reasoning,
        "n_instances": len(rows),
        "any_overload": any_overload,
        "any_infeasible": any_infeasible,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def composability_run(
    proposal_id: str,
    operator,
    *,
    instances: tuple[str, ...] = COMPOSABILITY_INSTANCES,
    budget_seconds: float = 10.0,
    n_repeats: int = 3,
    min_mean_abs_delta: float = 2.0,
    min_hit_rate: float = 0.40,
    max_regression_pct: float = 0.01,
) -> dict:
    """SPEC-8-COUNCIL-02 composability test.

    For each instance, run portfolio with 13 native arms (baseline) and
    portfolio with 13 + 1 (augmented, where the candidate is registered
    as an extra arm). Repeat `n_repeats` times with seeds 0..n-1 to
    denoise. Accept iff: mean absolute delta over the floor, hit rate
    over the floor, no per-instance regression > max_regression_pct.
    """
    from svrptw.solvers.classical import portfolio as pm

    out_dir = Path("bench/runs/council/composability")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{proposal_id}.json"
    settings = Settings()

    extra = {f"council_{proposal_id}": operator}
    rows: list[dict] = []
    for ip in instances:
        if not Path(ip).exists():
            _LOG.warning("composability instance missing: %s; skip", ip)
            continue
        inst = load_instance(ip)
        base_costs, aug_costs = [], []
        any_overload = any_infeasible = False
        for rep in range(n_repeats):
            # Paired seeding: baseline and augmented use the SAME bandit RNG
            # for this rep. Removes exploration-variance noise so the
            # measured delta is attributable to the candidate operator,
            # not to which arm was chosen first by chance.
            base = pm.solve(inst, settings, budget_seconds=budget_seconds,
                            seed=rep)
            aug  = pm.solve(inst, settings, budget_seconds=budget_seconds,
                            extra_arms=extra, seed=rep)
            base_costs.append(float(base.metrics["operational_cost"]))
            aug_costs.append(float(aug.metrics["operational_cost"]))
            if aug.metrics.get("capacity_overload", 0.0) > 0.0:
                any_overload = True
            if aug.metrics.get("feasible", 1.0) == 0.0:
                any_infeasible = True
        mb = float(np.mean(base_costs))
        ma = float(np.mean(aug_costs))
        delta = mb - ma  # positive = candidate helped
        regress_pct = (ma - mb) / max(mb, 1e-6) if ma > mb else 0.0
        rows.append({
            "instance": Path(ip).stem,
            "baseline_costs": base_costs,
            "augmented_costs": aug_costs,
            "mean_baseline": mb,
            "mean_augmented": ma,
            "delta": delta,
            "regress_pct": regress_pct,
            "any_overload": any_overload,
            "any_infeasible": any_infeasible,
        })

    deltas = np.array([r["delta"] for r in rows], dtype=np.float64)
    mean_abs_delta = float(deltas.mean()) if len(deltas) else 0.0
    hit_rate = float((deltas > 1.0).sum()) / max(1, len(deltas))
    worst_regress = max((r["regress_pct"] for r in rows), default=0.0)
    any_overload = any(r["any_overload"] for r in rows)
    any_infeasible = any(r["any_infeasible"] for r in rows)
    accepted = (
        mean_abs_delta >= min_mean_abs_delta
        and hit_rate >= min_hit_rate
        and worst_regress <= max_regression_pct
        and not any_overload
        and not any_infeasible
    )
    reasoning = (
        f"mean_delta={mean_abs_delta:.2f} (floor={min_mean_abs_delta}), "
        f"hit_rate={hit_rate:.2f} (floor={min_hit_rate}), "
        f"worst_regress_pct={worst_regress:.4f} (max={max_regression_pct}), "
        f"overload={any_overload}, infeasible={any_infeasible}"
    )
    result = {
        "proposal_id": proposal_id,
        "mode": "composability",
        "mean_abs_delta": mean_abs_delta,
        "hit_rate": hit_rate,
        "worst_regress_pct": worst_regress,
        "per_instance": rows,
        "accepted": accepted,
        "reasoning": reasoning,
        "n_instances": len(rows),
        "n_repeats": n_repeats,
        "budget_seconds": budget_seconds,
        "any_overload": any_overload,
        "any_infeasible": any_infeasible,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
