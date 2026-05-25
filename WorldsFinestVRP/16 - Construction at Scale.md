---
title: Construction at scale (N≥1000) — drop PyVRP dependency
project: SVRPTW
tags: [construction, scale, dual-graph, phase-G]
updated: 2026-05-15
---

# Phase G — Construction at scale, dual-graph clustering, drop PyVRP

## Strategic shift (user direction, iter-5m)

1. **DROP the LLM proposer.** 25 proposals, 0 production wins. Operator
   library is saturated. The plumbing stays as historical data; the
   active code path is removed from `solve_auto`.
2. **Bugs**: `policy_mlp.load()` ops-order is fixed (set comparison).
3. **Build our own construction** that matches PyVRP on quality at
   matched wall — explicitly to drop PyVRP as a hard dep at scale.
4. **Use the dual graph (osmnx) for clustering metrics**, NOT the
   Euclidean coords. The graph is already loaded for road-routing —
   reuse it for clustering signals at O(|edges|) instead of O(N²).
5. **Target N≥1000.** Current N=50–500 is the comfort zone; 1000+ is
   where speed and scaling matter for production.
6. **Training overfit/generalization guards** for the MLP policy.

## Why dual-graph clustering wins on speed

Current metrics suite (`svrptw/metrics/quality.py`):
- `intra_dist_mean` — O(K · n_route²) pairwise distances per route
- `inter_route_crossings` — O(K² · |segments|²) cross-pair segment intersections
- `convex_hull_overlap_count` — O(K² · vertices²) SAT
- `_segments_cross` — pure geometric, can't reuse network info

At N=200 the metrics call already adds ~5-20ms per `evaluate()`.
At N=1000 those n² terms blow up: ~25-100ms per evaluate, and the
bandit calls evaluate() ~20-100 times per second. Unworkable.

**The osmnx graph already has structure** that maps onto VRP clustering:
- **Node degree**: customers on dense intersections vs cul-de-sacs
- **Betweenness centrality** (precomputable once per instance):
  customers on through-routes get high BC; cul-de-sacs get low.
  Routes that mix high-BC + low-BC stops are spaghetti; clean routes
  cluster within a BC band.
- **Community detection** (Louvain on the customer-induced subgraph):
  routes that span communities = cross-community bleeding = bad.
- **Edge-weight-based shortest path overlap**: two routes overlap if
  their shortest-path polylines share many edges. O(|polyline|) not O(N²).

These are O(|graph_edges|) or O(|polyline|·K) — independent of N²
once the graph is loaded.

## What ships in Phase G

### G1. Drop LLM proposer from active path (DONE this iter)
- Spec amendment in `Sessions/2026-05-15_phase_a_closed.md` (next iter):
  "council/ frozen as historical; no `extra_arms` from council in default solve_auto."
- Existing `extra_arms=` kwarg stays but no caller passes proposer arms.
- `bench/scripts/council_*` scripts marked stale.

### G2. Graph-aware metrics module: `svrptw/metrics/graph_quality.py` (sub-agent)
Operates on (Instance, Solution, optional precomputed osmnx graph).
- `route_bc_mean(route, G_subgraph)` — mean betweenness centrality of
  route's customer nodes; tight cluster has low variance.
- `route_community_purity(route, communities)` — fraction of route's
  customers in the dominant community. 1.0 = perfect, 0.5 = bleeding.
- `cross_route_edge_overlap(route_a, route_b, G)` — # of road-segments
  used by BOTH routes (the network-aware "crossing" replacement).
- `quality_index_graph(...)` — synthesized 0..1, drop-in for the
  Euclidean `quality_index` but O(|graph_edges|) instead of O(N²).

### G3. Scale-aware construction: `svrptw/solvers/classical/fast_construct_v2.py` (sub-agent)
Targets N≥1000 at sub-second wall:
1. Precompute osmnx graph (cached). One-time per city.
2. Louvain community-detect on the customer-induced subgraph.
3. Per community, sweep / regret-2 within the community.
4. Greedy stitching of cross-community customers (the long tail).
5. ONE pass of `two_opt_intra` per route (no `swap_star` — too slow).
6. Return.

Target wall: 0.5s at N=1000, 1.5s at N=2000.

### G4. New instance set at N=1000+ (sub-agent stretch goal)
Extend `svrptw/instances_gen/osmnx_generator.py` to produce N=1000 and
N=2000 OSM instances for benching. If too big a lift this iter,
defer — focus on G2/G3 first.

### G5. RL training guards (future iter)
- K-fold cross-validation: hold out 20% of bandit transitions, early
  stop when validation loss stops improving (not train loss).
- Per-op pickup distribution sanity: if a single op gets >50% picks
  the policy is collapsing — restart with higher entropy bonus.
- Document in `svrptw/solvers/learning/policy_mlp.py`.

## Acceptance criteria

- **G1**: solve_auto with default settings doesn't import council; bit-identical legacy behavior.
- **G2**: graph-aware metrics give a quality_index within 0.05 of the Euclidean one on Manhattan N=200, AND compute in <2ms per call (vs ~10-20ms for Euclidean).
- **G3**: fast_construct_v2 runs in <1s at N=1000 AND produces quality_index_graph >= 0.5.
- **Strict-dominance bench** (Phase G end): fast_construct_v2 + bandit_30s_qw=1.0 on N=1000 vs PyVRP@30s. Metric: unified score (cost + graph_quality_index). Win = mean unified ≥ PyVRP at matched wall.

## Files to add (planned)
- `svrptw/metrics/graph_quality.py` — G2
- `svrptw/solvers/classical/fast_construct_v2.py` — G3
- `bench/scripts/scale_n1000_construction.py` — G4
- `tests/unit/test_graph_quality_smoke.py` — G2 smoke
- `tests/unit/test_fast_construct_v2_smoke.py` — G3 smoke

## Cross-references
- [[01 - Progress Report]]
- [[10  - Human Reflection]] — user's vision
- [[11 - RL Roadmap]] — training guards section
- [[13 - Construction Bypass]] — Phase E predecessor (proved fast_construct didn't dominate)
- [[15 - Cost-Model Exploration]] — Phase F (paused if G yields larger wins)
- [[Sessions/2026-05-15_phase_a_closed]] — Phase A done; this is the unblock
