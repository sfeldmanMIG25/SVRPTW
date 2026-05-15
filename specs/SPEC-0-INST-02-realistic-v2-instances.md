# SPEC-0-INST-02 — Realistic v2 instance set

```
ID:            SPEC-0-INST-02
Title:         Operationally-realistic asymmetric VRPTW instance set
Owner role:    OR Engineer
Status:        FROZEN
Inputs:        Frozen cities.yaml, master seed, feasibility probe budget
Outputs:       instances/v2/ + instances/v2/manifest.sha256 + instances/v2/REPORT.md
```

## Why this exists

v1 (SPEC-0-INST-01) uses OSM road networks but is feature-poor: single
hard TW per stop, homogeneous fleet, no breaks, no zones. Our research
(see Track-1 brief below) confirms what real dispatchers care about and
what makes academic instances trivially solvable.

**Hard rule:** every feature must serve at least one of two goals —
**(a) realism** (a real dispatcher would recognise it) or **(b) discriminating power**
(it produces a measurable spread between SOTA solvers).

## Features (10, mostly additive to v1)

1. **Mixed hard + soft time windows.**
   - Each customer carries `tw_kind ∈ {hard, soft}` (60% hard, 40% soft).
   - Soft TW lateness is charged at `soft_late_penalty_per_minute = wage * 2`
     up to a hard cutoff at `due + 60 min`; past that, drop.
   - SVRPBench (2025) ablation: TWs alone inflate cost 5-6×; mixed hard/soft
     produces a larger spread between solvers.

2. **Driver-zone compatibility.**
   - Each instance partitions customers into 3-8 named zones (k-means on
     lat/lon + Voronoi cleanup).
   - Each vehicle has an allow-list of zones; a vehicle may serve a
     non-listed zone at a penalty (`out_of_zone_penalty_per_stop = wage * 30 min`).
   - 70% of customers are in the "easy" zones for ≥ 2 vehicles; remainder
     are zone-locked (hard constraint, no out-of-zone).

3. **Statutory breaks.**
   - **EU 561/2006 mode:** 45 min break after 4.5 h driving (continuous);
     11 h daily rest is implicit (depot-out + depot-in within 14 h day).
   - **FMCSA mode:** 30 min break after 8 h on-duty.
   - Break is scheduled either at a depot/rest-area node (set of 5-10
     candidate nodes per city) or in-route at a customer with a 5-minute
     wait window where the driver is reading not unloading.
   - Mode chosen per instance via the `breaks_regime` field (US vs EU).

4. **Heterogeneous fleet with vehicle-arc compatibility.**
   - 3 vehicle classes: van (capacity 50, OK everywhere), box truck (100,
     no narrow residential streets), tractor (200, only on
     arterial/highway tagged edges).
   - Forbidden-arc list per class — derived from OSM `highway=` tags.
   - Per-class travel-time matrix; the existing single-matrix loader is
     extended to `(class, N+1, N+1)`.

5. **Time-dependent travel times.**
   - AM peak 7-9, PM peak 16-18: travel times × `1.0 + 0.6 * Gaussian
     centred on peak`.
   - Off-peak: log-normal noise σ = 0.15.
   - Travel-time matrix is now a **function** T(i, j, depart_time) — the
     loader returns a `TravelTimeFn` callable, not a flat array.
   - **Stochastic component is OFF on the headline benchmark** per
     ADR-0001 D4; this enters as an optional `--stochastic` flag for
     robustness experiments only.

6. **Service-time variability by stop archetype.**
   - Each customer gets `archetype ∈ {curbside, apartment, commercial}`
     (40 / 35 / 25 split).
   - Service mean × multiplier (curbside 1.0, apartment 2.0, commercial 3.0).
   - Within archetype: log-normal noise σ = 0.20.
   - Driver multiplier ∈ [0.85, 1.25] applied per vehicle.

7. **Multi-trip / depot reload.**
   - Optional `allow_reload` flag per instance.
   - If set, vehicle may return to depot mid-shift to pick up more capacity
     (zero-cost reload, but counts as a stop visit).

8. **Pattern-break stops.**
   - Lunch window: each driver must take a 30-minute non-productive
     window somewhere in 11:30-13:30.
   - Fueling: if route exceeds 150 mi, must visit a fueling node.

9. **Pickup-and-delivery (PDP) precedence.**
   - 20% of instances have 10-30% of customers as PDP pairs (same vehicle,
     pickup before delivery).
   - Schema: `customer.pd_pair_id` links pairs.

10. **Multi-day / periodic visits.**
    - 10% of instances have customers with a `day_of_week` allow-list (1-5).
    - The horizon stays a single day, but the customer is only orderable
      on the assigned days; multi-day planning is a separate Phase.

## Instance JSON schema additions

```jsonc
{
  // ... v1 fields preserved ...
  "schema_version": "2.0",
  "zones": [{"id": "Z1", "centroid": [...], "members": [3, 7, 12, ...]}, ...],
  "vehicles": [
    {"class": "van", "capacity": 50, "allow_zones": ["Z1", "Z2"],
     "driver_service_multiplier": 1.05, "max_shift_minutes": 600},
    ...
  ],
  "breaks_regime": "EU561",
  "time_dependent_travel": true,
  "rest_areas": [{"node_id": ..., "x": ..., "y": ...}, ...],
  "customers": [
    {  // ... v1 fields ...
       "tw_kind": "soft", "soft_late_cap_minutes": 60,
       "archetype": "apartment", "service_noise_sigma": 0.20,
       "pd_pair_id": null,
       "day_of_week_mask": null
    },
    ...
  ]
}
```

## Validation protocol (hard requirement)

Per the Track-1 brief, every v2 instance passes:

1. **Feasibility probe** — OR-Tools CP-SAT @ 60 s; must produce a
   feasible-or-low-drop solution (≤ 5% missed).
2. **Discriminating power** — run **LKH-3 @ 60 s** and **HGS-VRPTW @ 60 s**
   (the Kool/Vidal 2022 reference impl, or `pyvrp` once we add it).
   Keep only instances where `|cost_LKH3 − cost_HGS| / min > 2%`.
3. **Lower-bound gap** — solve LP relaxation of the set-partitioning
   formulation; reject if the gap to LKH-3 is `< 1%` (= trivially solvable).
4. **Deduplication** — fingerprint = (n, K, asymmetry, TW tightness, zone
   count, breaks regime, vehicle class distribution). Hash; reject collisions.

## Acceptance

- `python -m svrptw.instances_gen build --version v2 --config specs/data/cities.yaml`
  produces 200 instances (8 cities × 5 sizes × 5 each) in < 8 hours.
- Each instance JSON parses against the v2 schema.
- `instances/v2/REPORT.md` records per-instance discriminating-power
  margin, OR-Tools probe time, LP gap.
- Loader (`svrptw.io.load_instance`) auto-detects schema_version and
  returns the appropriate object (back-compat: v1 still loads).

## Comparison benchmark sets we also support (as separate adapters)

- **Solomon** (`svrptw.io.solomon`) — 100-customer canonical baseline.
- **Gehring-Homberger** (`svrptw.io.gh`) — 200-1000.
- **Sartori-Buriol PDPTW 2020** (`svrptw.io.sb_pdptw`) — closest to our OSM instances.
- **Amazon Last Mile 2021/2022** (`svrptw.io.amazon`) — gold standard for
  zone realism. Public on AWS Open Data.
- **SVRPBench 2025** (`svrptw.io.svrpbench`) — newest realistic+hard reference.

## Non-goals

- Full periodic VRP (multi-day) — single-day horizon stays.
- Real-time dynamic / re-routing — static planning only.
- Real-traffic API integration — synthetic peak-hour multipliers only.

## Dependencies

- SPEC-0-INST-01 (the v1 generator we extend).
- HGS-VRPTW reference (Kool/Vidal 2022) for discriminating-power check.
- `pyvrp` package for HGS comparison.
- SCIP or OR-Tools LP for lower-bound check.

## References

- Vidal et al., HGS-VRPTW (Kool/Vidal 2022)
- Sartori & Buriol, OVIG generator (https://github.com/cssartori/ovig)
- Merchan et al., Amazon Last Mile Routing Challenge dataset
- SVRPBench 2025, arXiv:2505.21887
- EU 561/2006, FMCSA HOS regulations
