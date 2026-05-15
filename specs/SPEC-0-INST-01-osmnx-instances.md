# SPEC-0-INST-01 — Hard asymmetric road-network instance set v1

```
ID:            SPEC-0-INST-01
Title:         OSMnx-derived asymmetric VRP instance set v1
Owner role:    OR Engineer
Status:        FROZEN
Inputs:        City list (specs/data/cities.yaml), master seed
Outputs:       instances/v1/*.json + instances/v1/manifest.sha256
```

## Behavior

Replace the Euclidean random-points generator (`data_generator.py`) with a deterministic, asymmetric, network-based generator. Each instance is a JSON file with the schema below; the file set is frozen by a SHA-256 manifest committed to the repo.

### Generation procedure

1. **City pool.** Start with 8 mid-size cities chosen to span topology classes:
   - Manhattan-style grid: `Manhattan, NY` (one-way dominated)
   - European radial: `Paris, France`
   - Hub-and-spoke: `Boston, MA`
   - Sprawl: `Phoenix, AZ`
   - Coastal-constrained: `San Francisco, CA`
   - Dense colonial: `Boston, MA` (already listed — replace with `Charleston, SC`)
   - Mid-size US grid: `Austin, TX`
   - Mid-size US radial: `Pittsburgh, PA`

   Locked list and OSM relation IDs in `specs/data/cities.yaml`. OSMnx pulls are cached under `instances/v1/osm_cache/` with SHA-256 verification.

2. **Per city, per size N ∈ {50, 100, 200, 500}, build 5 instances** → 8 × 4 × 5 = **160 instances**.

3. **Customer sampling.** Within each city's network:
   - Pick depot as a node within 5% of the city centroid.
   - Sample N customer nodes via spatially-stratified Poisson-disk over the network's node set (avoids clumping).
   - Reject any node within 50 m of the depot.

4. **Travel-time matrix.** Compute directed shortest paths from every customer (and depot) to every other, using OSMnx `length / maxspeed` per-edge. Matrix is asymmetric. Stored in compressed `.npz` alongside the JSON.

5. **Time windows.** Compute the network's natural travel-time scale `T_med = median(matrix)`. For each customer:
   - Service time: uniform in [5, 15] minutes (deterministic; seeded).
   - TW width: uniform in [1.5 · T_med, 3.0 · T_med] (tight by construction).
   - TW center: feasibility-aware random — sampled such that ≥ 1 vehicle schedule exists. The generator runs a quick OR-Tools feasibility probe; instances that fail are resampled (max 5 attempts) and discarded if still infeasible. Feasibility-rate is logged.

6. **Fleet.** Vehicle count = `ceil(N · 0.3)`, capacity = `ceil(1.4 · total_demand / num_vehicles)`. Demand: uniform integer in [1, 10].

### Instance JSON schema

```jsonc
{
  "instance_id": "OSM-Austin-N100-I003",
  "city": "Austin, TX",
  "osm_relation_id": 113314,
  "graph_sha256": "...",
  "num_customers": 100,
  "num_vehicles": 30,
  "vehicle_capacity": 47,
  "depot": {"node_id": 53231233, "x": -97.7437, "y": 30.2672, "ready": 480, "due": 960},
  "customers": [
    {"id": 1, "node_id": 53231256, "x": ..., "y": ..., "demand": 4,
     "ready": 510, "due": 605, "service": 8}
  ],
  "travel_time_matrix_path": "matrices/OSM-Austin-N100-I003.npz",   // (N+1, N+1) directed minutes
  "travel_distance_matrix_path": "matrices/OSM-Austin-N100-I003_dist.npz",
  "asymmetry_score": 0.41,              // mean |t_ij - t_ji| / max(t_ij, t_ji)
  "generator_seed": 12345,
  "schema_version": "1.0"
}
```

## Invariants

- Manifest SHA-256 checked at every benchmark run; mismatch is a hard error.
- Asymmetry score per instance ≥ 0.05 (we have measurable directional structure; pure-Euclidean instances would score ~0).
- Every instance passes feasibility probe (at least one OR-Tools-found solution exists at 60s budget).
- Travel-time matrix is non-negative, zero on the diagonal, satisfies triangle inequality up to 1e-6 tolerance (it should — it's shortest paths on a directed graph).

## Acceptance

- `python -m svrptw.instances_gen build --config specs/data/cities.yaml --out instances/v1` produces 160 instances in < 4 hours on a single workstation.
- `python -m svrptw.bench check_instances` recomputes hashes and exits 0.
- `instances/v1/manifest.sha256` committed.
- `instances/v1/REPORT.md` records: per-city node counts, mean asymmetry score, mean TW tightness, feasibility-probe success rate.
- Loading any instance via `svrptw.io.load_instance(path)` returns a populated `Instance` dataclass with the asymmetric matrix already attached.

## Non-goals

- Real-time traffic. Travel times are static `length / maxspeed` (free-flow).
- Pickup-and-delivery, multi-depot, EV charging. Single depot, drop-off only.
- Synthetic / non-real cities. We commit to OSM data.

## Dependencies

- SPEC-0-CFG-01 (Settings.seed used as master seed).
- OSMnx ≥ 1.9, networkx ≥ 3.2 in the `.venv`.

## Open questions

- City list: 8 cities is a starting point; principal may want fewer/more. Recorded under "City pool" — easy to amend.
- Should we include one "OR-Tools-hostile" extreme? E.g., one city with extreme one-way density (e.g., parts of Cambridge, MA) at N=500. Proposed as `SPEC-0-INST-02` once `INST-01` is built and we can measure where OR-Tools breaks.
