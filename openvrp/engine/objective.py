"""Caller-configurable objective (SPEC-OPENVRP-04 §3, SPEC-OPENVRP-00 D8).

```
objective =  operational_cost_weight * operational_cost
           + vehicle_count_weight    * vehicles_used
           + Σ_k  quality_terms[k]   * normalized_penalty_k
           + drop_penalty + soft_tw_penalty + embargo_penalty + ...
```

Active fleet minimization (D7) applies even when ``vehicle_count_weight=0``:
the solver always prefers the smaller fleet among solutions of equal
weighted objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from openvrp.schema.input import ObjectiveConfig


@dataclass
class ObjectiveBreakdown:
    operational_cost: float = 0.0
    vehicle_count_cost: float = 0.0
    quality_penalties: dict[str, float] = field(default_factory=dict)
    soft_tw_penalty: float = 0.0
    drop_penalty: float = 0.0
    embargo_penalty: float = 0.0
    per_route_fixed_cost: float = 0.0
    fleet_fixed_cost: float = 0.0
    other_terms: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (self.operational_cost
                + self.vehicle_count_cost
                + sum(self.quality_penalties.values())
                + self.soft_tw_penalty
                + self.drop_penalty
                + self.embargo_penalty
                + self.per_route_fixed_cost
                + self.fleet_fixed_cost
                + sum(self.other_terms.values()))


def compose_objective(operational_cost: float,
                      vehicles_used: int,
                      quality_metrics: dict[str, float],
                      penalties: dict[str, float],
                      objective: ObjectiveConfig) -> ObjectiveBreakdown:
    """Compose the weighted objective. ``quality_metrics`` must already
    be in penalty form (lower=better, non-negative; "higher better"
    metrics like ``time_window_slack`` are negated by the metrics module).

    ``penalties`` carries the enumerated cost terms:
    ``per_route_fixed_cost``, ``fleet_fixed_cost``, ``soft_tw_penalty``,
    ``drop_penalty``, ``embargo_penalty``.
    """
    br = ObjectiveBreakdown()
    br.operational_cost = objective.operational_cost_weight * operational_cost
    br.vehicle_count_cost = objective.vehicle_count_weight * float(vehicles_used)
    for k, w in objective.quality_terms.items():
        if w == 0.0:
            continue
        br.quality_penalties[k] = w * float(quality_metrics.get(k, 0.0))
    br.soft_tw_penalty = float(penalties.get("soft_tw_penalty", 0.0))
    br.drop_penalty = float(penalties.get("drop_penalty", 0.0))
    br.embargo_penalty = float(penalties.get("embargo_penalty", 0.0))
    br.per_route_fixed_cost = float(penalties.get("per_route_fixed_cost", 0.0))
    br.fleet_fixed_cost = float(penalties.get("fleet_fixed_cost", 0.0))
    return br


__all__ = ["ObjectiveBreakdown", "compose_objective"]
