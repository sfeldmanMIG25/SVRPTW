# FAQ / migration

**Q. I already have an OD matrix. What's the shortest path to a solve?**
A. `from openvrp import solve_od; sol = solve_od(T, stops, depots, fleet, options=SolveOptions(budget_seconds=10.0))`.

**Q. I use OR-Tools today. What maps where?**
A. OR-Tools `RoutingIndexManager` ⇄ openvrp `ODMatrix.index_of`. OR-Tools dimensions ⇄ openvrp `VehicleClass.capacity` (multi-dim). OR-Tools time-window callbacks ⇄ openvrp `Stop.time_windows`. The pattern is much shorter — see the quickstart.

**Q. Determinism — what's the contract?**
A. Identical `Problem` + `seed` ⇒ identical stop sequence in every route across runs, across `threads=1` vs `N`, and across repeated `fast` constructions. Timings are a pure function of sequence + input. See `tests_openvrp/unit/test_determinism.py`.

**Q. Is it really free?**
A. MIT. All core and `[network]` dependencies are permissive (numpy/scipy BSD, pydantic/networkx/osmnx MIT/BSD, shapely/pyproj BSD). PyVRP (MIT) is fine as the optional `[pyvrp]` accelerator.

**Q. Can I change what "best" means?**
A. Yes — `Constraints.objective.quality_terms` folds any of the 8 catalog metrics into the optimization with a caller-set weight. Default = pure operational cost + active fleet minimization. There is no visual/VLM scoring and none can be added (D9).

**Q. Does it minimize trucks?**
A. Yes, always (D7). Set `Constraints.objective.vehicle_count_weight > 0` to *intensify* that pressure. `sol.vehicles_minimum_found` is the audit — if it differs from `sol.vehicles_used`, the diagnostics quantify the trade.

**Q. What about EU 561 / US HOS — is it the real ruleset or a single drive-cap simplification?**
A. The full regulation. `ShiftRule(ruleset='eu561')` instantiates 4.5h continuous-drive break (or 15+30 split), 9h daily drive, 11h daily rest (reducible to 9h ≤3×/wk), 45h weekly rest. `ruleset='us_hos'` instantiates 11h drive in 14h on-duty, 30-min break after 8h, 10h reset, 60/70h in 7/8d, 34h restart. Multi-day routes accrue and reset correctly.

**Q. How big can N get?**
A. The PyVRP-free native solver runs **N=1000 in single-digit seconds** on the bench. Larger N is bounded by your time budget; the algorithm is deadline-cooperative, so a tight budget returns a valid (likely suboptimal) solution within ~budget + finalize.

**Q. Why isn't ArcGIS in the perf benchmark?**
A. Licensed; can't be redistributed in a benchmark harness. The competitive claim is vs PyVRP, OR-Tools, and LKH-3.
