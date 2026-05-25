# Objective control

The caller drives optimization from the top. Default = pure operational
cost + active fleet minimization (D7). Each quality metric can be
folded in via a non-zero weight.

```python
from openvrp import Constraints, ObjectiveConfig, QUALITY_CATALOG

print("Catalog metrics:", QUALITY_CATALOG)
# ('route_crossings', 'mean_detour_ratio', 'load_balance_cv',
#  'load_balance_gini', 'time_window_slack', 'intra_route_compactness',
#  'cross_route_overlap', 'quality_per_route')

constraints = Constraints(
    drop_penalty=1000.0, allow_drops=False,
    per_route_fixed_cost=50.0,
    objective=ObjectiveConfig(
        operational_cost_weight=1.0,
        vehicle_count_weight=0.0,            # always-active even at 0
        quality_terms={
            "route_crossings": 0.5,          # visual chaos
            "load_balance_cv": 1.0,          # spread work evenly
            "time_window_slack": 0.1,        # reward staying ahead of TWs
        },
        soft_tw_penalty_per_sec=0.05,
    ),
)
```

## The always-on quality report

Even if `quality_terms` is empty, **every catalog metric is computed
and reported** in `sol.quality_report.solution_level`:

```python
print(sol.quality_report.solution_level)
# {'route_crossings': 3.0, 'mean_detour_ratio': 0.04,
#  'load_balance_cv': 0.18, ..., 'quality_per_route': 0.27}
```

This is the difference between optimizing for a metric and merely
inspecting it. You can ship a pure-cost optimizer and still surface the
operational-quality table for downstream review.

## No VLM scoring anywhere

There is no visual / vision-language-model metric in the catalog and
**none can be added** (SPEC-OPENVRP-00 D9). Quality here means
*measurable operational geometry, balance, and slack* — not what a
model "thinks" the routes look like. This is a deliberate scope choice.

## Sign convention (penalty form)

Each metric is normalized into a non-negative penalty form (lower =
better), so the weight always has the same sign meaning:

| key | meaning |
|---|---|
| `route_crossings` | inter-route segment intersections |
| `mean_detour_ratio` | actual leg / straight OD - 1.0 |
| `load_balance_cv` | CV of route loads |
| `load_balance_gini` | Gini of route loads |
| `time_window_slack` | -mean normalized slack (negated so higher slack is lower penalty) |
| `intra_route_compactness` | mean intra-route spread |
| `cross_route_overlap` | bbox overlap across routes / total bbox area |
| `quality_per_route` | K-fair composite ÷ route count |

## Fleet minimization (D7)

Even with `vehicle_count_weight=0`, the solver always prefers the
smaller fleet among solutions of equal weighted objective. The
`vehicle_count_weight` only *intensifies* that pressure.

`sol.vehicles_minimum_found` reports the smallest fleet a greedy
post-hoc merge proves sufficient. If it's smaller than
`sol.vehicles_used`, the diagnostics record the cost delta — the
caller can decide whether to re-solve with `vehicle_count_weight`
turned up.
