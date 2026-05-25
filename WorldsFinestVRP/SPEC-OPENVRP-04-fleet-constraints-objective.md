# SPEC-OPENVRP-04 — Fleet, Constraints & Objective Control

**Status:** FROZEN
**Owner role:** Spec Author + OR Engineer
**Depends on:** SPEC-OPENVRP-00, 01
**Modules:** `openvrp/engine/{constraints,fleet,objective}.py`,
`openvrp/engine/_adapter.py`

This spec encodes the reason OpenVRP exists: rich heterogeneous-fleet constraints,
caller-controlled objective, active fleet minimization, and operational-quality
metrics the caller can fold into optimization — with no visual/VLM scoring anywhere.

---

## 1. Heterogeneous fleet semantics

`VehicleClass` is a template; physical vehicles are instantiated from it. Unbounded
`count` is permitted; the solver **drives the number of vehicles used down to the
minimum** consistent with the active objective (SPEC-OPENVRP-00 D7) and reports both
the used and the minimal-sufficient count.

- **Capacity**: multi-dimensional; route is feasible iff every prefix, every
  dimension, cumulative load ≤ `class.capacity[dim]`. PD load is +at pickup,
  −at delivery; running load governs the check.
- **Speed**: leg time = base_OD_time / `speed_factor`; distance is class-invariant
  (per-class distance differences are modeled via zones, documented limitation).
- **Cost**: `fixed_cost·[used]` + `cost_per_second·duration` +
  `cost_per_meter·distance` + peak surcharge + penalties; `per_route_fixed_cost`
  adds globally on top.
- **Skills**: stop serviceable by class C iff `required_skills ⊆ C.provides_skills`.
- **Class-zone access**: kind=access ⇒ only whitelisting classes may enter;
  kind=embargo ⇒ none may enter during `active_window` unless `embargo_soft`.

---

## 2. Driver-hour rulesets (full — SPEC-OPENVRP-00 D5)

`ShiftRule.ruleset ∈ {none, eu561, us_hos, custom}`. `eu561` and `us_hos` implement
the **complete** regulatory structure, not a single drive-cap:

- **eu561**: 4.5 h continuous driving → ≥45 min break (or 15+30 split); 9 h daily
  driving, extendable to 10 h ≤2×/week; 11 h daily rest, reducible to 9 h ≤3× between
  weekly rests; 56 h weekly / 90 h fortnightly driving; 45 h weekly rest, reducible to
  24 h with compensation. Multi-day routes accrue and reset these correctly.
- **us_hos** (property-carrying): 11 h driving within a 14 h on-duty window; 30 min
  break after 8 h driving; 10 h off-duty reset; 60/70 h in 7/8 days; 34 h restart.
- Explicit `ShiftRule` fields override individual parameters without collapsing the
  rest of the ruleset.
- The solver inserts breaks and rests as timeline events at the latest feasible
  position; it never relocates a stop to dodge a rule. Multi-day feasibility (daily
  vs weekly rest) is tracked across the route's accrued driving/on-duty.

---

## 3. Composite operational-quality metric catalog (SPEC-OPENVRP-00 D8, D9)

Every metric below is **always computed and reported** in
`Solution.quality_report`; each becomes an **optimized objective term** when the
caller sets a weight in `ObjectiveConfig.quality_terms`. All are
operational/geometric. There is **no visual or VLM metric** and none may be added.

| key | meaning | sign (penalty form) |
|---|---|---|
| `route_crossings` | count of inter-route segment intersections | lower better |
| `mean_detour_ratio` | mean (leg path length / straight OD) | lower better |
| `load_balance_cv` | coeff. of variation of vehicle loads | lower better |
| `load_balance_gini` | Gini of vehicle loads | lower better |
| `time_window_slack` | mean unused slack before TW close | higher better → penalty = −slack |
| `intra_route_compactness` | mean intra-route spatial spread | lower better |
| `cross_route_overlap` | convex-hull overlap area across routes | lower better |
| `quality_per_route` | the K-fair composite (overall quality ÷ routes) | higher better → penalty = −value |

Normalization: each metric maps to a non-negative penalty via a documented,
instance-scale-invariant transform so weights are comparable across instance sizes.
The objective the search optimizes is:

```
objective =  operational_cost_weight · operational_cost
           + vehicle_count_weight   · vehicles_used
           + Σ_k  quality_terms[k]  · normalized_penalty_k
           + hard-constraint feasibility (enforced, not weighted)
           + drop / soft-TW / embargo / overrun penalties (priced per Constraints)
```
Active fleet minimization (D7) applies even when `vehicle_count_weight=0`: the solver
always prefers the smaller fleet among solutions of equal weighted objective; the
weight only *intensifies* that pressure (lets the caller trade cost for fewer
vehicles deliberately).

---

## 4. Feasibility precedence (documented, deterministic)

Highest first: (1) each stop on exactly one route or dropped (if `allow_drops`);
(2) skills / class-zone access; (3) capacity (every dim, every prefix); (4) hard
time windows + depot windows; (5) driver-hour rests/breaks (inserted to satisfy,
never dropped); (6) hard shift window + `max_route_seconds`/`meters`; (7) soft
penalties (soft TW, soft embargo, overrun) — minimized, not enforced. Violating 1–6
⇒ `infeasible`; satisfying 1–6 with nonzero 7 ⇒ `feasible`, priced into objective.

---

## 5. Conditional operator registration (NON-OPTIONAL)

The search's operator pool is filtered at solve time to operators whose target axis
is active in the problem. Excluded when their axis is inactive:
`shift_start` (no peak multiplier+windows and no time-windowed embargo),
`class_shift` (no skill/cost differentiation across classes),
`depot_shift` (<2 depots), recharge insertion (no `max_route_meters`),
PD-precedence moves (no PD pairs), and any quality-targeting operator whose metric
has zero weight in `ObjectiveConfig`.

Rationale: registering no-op operators consumes search budget and measurably degrades
result magnitude (established in the research record; the fix recovered the loss and
reduced wall below baseline). This is correctness. A guard test suite asserts the
predicates and that the active pool is exposed in diagnostics.

---

## 6. Research-core adapter

`openvrp/engine/_adapter.py` is the only bridge from the public schema to the
vendored research search core (bandit, evaluator, constructions). Public names never
leak internal names. Golden snapshot tests pin the mapping; drift fails CI. The
vendored core is used unmodified except where the performance mandate
(SPEC-OPENVRP-03 §4) requires improvement — such changes are recorded as ADRs and
must not regress the conformance suite or the mandate.

---

## 7. Conformance suite (SPEC-OPENVRP-00 gate #5, must pass fully)

- E1 mixed capacity across classes respected at every prefix.
- E2 skills: skill-gated stops served only by capable classes (or dropped w/ reason).
- E3 class-zone access honored (no forbidden `ZoneEnterEvent`).
- E4 embargo window honored (or penalized iff `embargo_soft`).
- E5 **full EU 561**: a multi-day route exhibits continuous-drive break, a 15+30
  split break, a daily rest, a reduced daily rest within its allowance, and a weekly
  rest — each as a correctly-typed event with regulation-correct duration/placement.
- E6 **full US HOS**: 11 h drive / 14 h window / 30 min after 8 h / 10 h reset /
  60-70 h / 34 h restart all exercised.
- E7 shift-start window honored; applied offset reflected in `ShiftStartEvent`.
- E8 shift overrun (soft) priced, solution still feasible.
- E9 multi-depot home-binding: routes start/end at the class's home depot.
- E10 PD precedence + same-route + carried-load correctness.
- E11 EV range never exceeded; long demand forces `RechargeEvent`.
- E12 labor floor `min_routes` respected.
- E13 **fleet minimization**: with unbounded fleet, `vehicles_used ==
  vehicles_minimum_found` unless a weighted term strictly justifies more (diagnostics
  quantify the trade).
- E14 conditional registration: feature-free problem registers none of the
  conditional operators; all-feature problem registers all; verified via diagnostics.
- E15 **objective control**: weighting a `quality_terms` key changes the chosen
  solution and lowers that metric in `quality_report` versus the pure-cost run; all
  catalog metrics are reported even at zero weight.
- E16 precedence: a constructed skill/proximity conflict resolves per §4, not by
  nearest-vehicle.

## 8. Non-goals
- No new constraint *types* beyond this catalog; new ones arrive as additive schema
  fields following the research meta-recipe and the conditional-registration rule.
- No per-class distance/detour modeling beyond zones (documented limitation).
- No visual/VLM metric, ever (SPEC-OPENVRP-00 D9).
