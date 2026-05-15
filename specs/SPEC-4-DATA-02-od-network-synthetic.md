# SPEC-4-DATA-02 — Network-OD synthetic instance generator

```
ID:            SPEC-4-DATA-02
Title:         Replace Euclidean-noise synthetic with a true
               OD-network generator: every distance/time is a path
               through a generated road-like graph, not a planar metric
Owner role:    Data Engineer
Status:        FROZEN
Supersedes:    svrptw/instances_gen/synthetic.py's Euclidean+noise path
               for *new* training runs; old path retained as `--euclidean`
               for back-compat with cached datasets
Depends on:    SPEC-0-INST-02 (rich-field schema), SPEC-4-DATA-01 (scale to N=10k)
```

## Why

POMO v4 (6L, 150 epochs, synthetic) tested at **881.1** on Manhattan
N=50 vs POMO v3 (4L, 80 epochs, v1 OSM) at **822.1**. v4 lost 7 %
despite more capacity and 2× the epochs. The post-mortem
(`bench/figures/pomo_v3_vs_v4.md`) traced it to training-distribution
mismatch: synthetic is Euclidean coordinates with multiplicative
edge noise, OSM is asymmetric road-network with hub-and-spoke +
one-way + traffic-light geometry.

A model trained on the Euclidean approximation **does not learn the
inductive biases that win on real OSM data**, because Euclidean
asymmetry is i.i.d. noise on every edge; OSM asymmetry is
*structured* (driven by one-ways, freeway on-ramps, congestion
profiles). The remedy is to generate synthetic that reproduces the
structure, not just the marginal asymmetry statistics.

The user's quote: "I would strongly prefer that all data be on
network models (modeling a genuine OD) not euclidean space, if that
means improving our synthetic generation so be it."

## Behaviour

```python
from svrptw.instances_gen.network_synthetic import generate

inst = generate(
    N=200,
    seed=0,
    graph_topology="grid-with-arterials",   # see options below
    asymmetry_source="oneways",             # natural source, not noise
    congestion_profile="peak-rush",
)
```

### Topology generators

Each produces a directed graph over which paths give T (time) and
D (distance) — *not* coordinates.

1. **`grid-with-arterials`** — a small-world variant of a Manhattan-style
   grid. Most blocks are 2-way; a configurable fraction of
   one-direction streets ("one-ways") give natural asymmetry.
   Cross-cutting arterials add hub structure. *This is the default.*
2. **`fractal-radial`** — radial spokes from a downtown with
   inter-radial cross-streets. Approximates European old-city
   plans. One-way ratio increases toward the centre.
3. **`mesh-with-bridges`** — two grids separated by a "river," joined
   by a handful of bridges. Strong asymmetry on bridge approaches.
4. **`osm-like-sampled`** — sample a real OSM graph at training
   time via networkx-osmnx; treat the sample as one instance. Most
   faithful, slowest.

For each topology the generator computes:

- **Shortest-path time matrix `T`** (Dijkstra with `weight=travel_time`).
- **Shortest-path distance matrix `D`** (Dijkstra with `weight=length`).
- Both are asymmetric because the underlying graph is directed.
- Edge weights derive from `congestion_profile`: free-flow + a
  time-varying multiplier; instances generated for a fixed hour-of-
  day get a deterministic congestion snapshot.

### Customer placement

Customers are sampled as *nodes in the graph*, not (x, y) pairs.
The `customer` record keeps a `node_id` field; legacy coordinate
consumers receive a lat/lon round-trip via the graph's embedded
coordinates. This preserves rendering for ViVRP without re-anchoring
the cost model on coordinates.

### Heterogeneous fleet, breaks, chargers (v3 schema)

All pass through unchanged. Vehicles continue to traverse the
shortest-path matrices; charger locations are graph nodes, not
coordinates.

### Distance / time matrix size

Dense: O(N²) entries. For N=10 k the matrix is 800 MB — too large
to materialise. The generator emits a **sparse k-NN matrix** in
that regime (the existing `_build_sparse_neighbors` path), with
`k=64` covering ~ 99 % of moves the solver actually takes (we verify
this number on the v1 set).

## Acceptance gates

1. **Network-grounded asymmetry.** On a 100-instance N=200 sample,
   `(T - T.T).abs().mean() / T.mean() ≥ 0.05` (matches v1 OSM
   distribution; pure noise generators are flatter).
2. **POMO transfer.** POMO v5 trained for 80 epochs on N=50
   `grid-with-arterials` matches or exceeds v3 (822) on Manhattan
   N=50 held-out. (If it doesn't, the generator is still wrong.)
3. **Scale.** Generating one N=10 k instance fits in ≤ 1.5 GB peak
   RSS and ≤ 60 s wall-clock (sparse matrix path).
4. **Back-compat.** The `--euclidean` flag reproduces the old
   generator bit-exact. v3 instances re-load with no schema change.

## Non-goals

- Not replacing v1 OSM as the headline benchmark. v1 stays as the
  ground-truth "real-world" set; this is the *training* generator.
- Not modelling traffic dynamically. One generated instance =
  one hour-of-day snapshot. Time-dependent travel times stay in
  SPEC-0-INST-02 if/when we get there.

## Files this spec creates

| Path | Role |
|---|---|
| `svrptw/instances_gen/network_synthetic.py` | new generator |
| `svrptw/instances_gen/topologies/grid_arterials.py` | topology builder |
| `svrptw/instances_gen/topologies/fractal_radial.py` | topology builder |
| `svrptw/instances_gen/topologies/mesh_bridges.py` | topology builder |
| `svrptw/instances_gen/topologies/osm_sampled.py` | networkx-osmnx wrapper |
| `tests/unit/test_network_synthetic_asym.py` | asymmetry distribution match |

## Migration

- `instances_gen/synthetic.py` keeps working; new `--euclidean` flag
  is documented as legacy.
- POMO v5 retraining (SPEC-4-POMO-01) uses the new generator by
  default. v4 checkpoint is archived; v5 supersedes.
