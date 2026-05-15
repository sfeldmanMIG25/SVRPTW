# SPEC-5-VIVRP-02 — Compound solution assessment

```
ID:            SPEC-5-VIVRP-02
Title:         Compound quality signal blending objective, VLM scores, and structural proxies
Owner role:    ML Engineer + Bench Engineer
Status:        DRAFT
Inputs:        Instance, Solution, Settings, VLM backend
Outputs:       svrptw.vivrp.compound.assess_compound -> CompoundAssessment
```

## Why

Operational cost alone (wage·time + transit·miles + missed·penalty) misses:

- **Cluster coherence** — a dispatcher's ability to assign neighborhoods to
  trucks (captured by `vivrp_clustering` today).
- **Customer-facing reliability** — variance in arrival times, willingness
  to absorb a single delay without cascading misses.
- **Driver-day shape** — front-loaded routes vs late-loaded ones differ
  in operational risk even at identical cost.
- **Visual interpretability** — would a human accept this plan? (captured
  by `vivrp_interpretability`.)

Phase 5 turn revealed a real tradeoff: `auction_gart` had the **lowest
operational cost (717)** AND the **lowest ViVRP score (2.75)** on the
smoke set. A single number can't represent that. The paper needs a
compound metric.

## Behavior

```python
@dataclass
class CompoundAssessment:
    # Hard numeric metrics from evaluate()
    operational_cost: float
    missed_deliveries: float
    feasible: bool
    # Visual-interpretability scores (1-10) from ViVRP
    vivrp_overall: int
    vivrp_clustering: int
    vivrp_geometry: int
    vivrp_interpretability: int
    # Computed structural proxies (no VLM, deterministic)
    cluster_coherence: float    # silhouette-like score on routes
    route_balance: float        # 1 - std(route_lengths) / mean(route_lengths)
    slack_resilience: float     # mean TW slack per customer (min over route)
    cohesion: float             # mean intra-route distance / mean inter-route distance
    # Compound score
    compound: float             # weighted blend; see Weights
    breakdown: dict[str, float] # per-component contribution
```

## Weights

Default (tunable per use case):

- 0.50 × normalized(−operational_cost)
- 0.20 × normalized(vivrp_overall)
- 0.10 × normalized(cluster_coherence)
- 0.10 × normalized(route_balance)
- 0.10 × normalized(slack_resilience)

Normalization: min-max over the leaderboard's row set so all components
are in [0, 1] before weighting. Weights are stored in
`svrptw/vivrp/compound_weights.yaml`.

## Training the VLM (front-end model quality training)

Phase 5 also opens the option to **fine-tune** the VLM on calibrated
preference pairs:

1. Generate ~500 (instance, solver_A_solution, solver_B_solution) triples
   from the bench.
2. Score each pair with the current Gemini Flash backend → preference label.
3. Use the labels as a teacher signal to fine-tune a smaller local VLM
   (Qwen2-VL-7B-Instruct if 8 GB allows; Qwen2-VL-2B otherwise) via LoRA.
4. Validate that the distilled model agrees with Gemini on a held-out
   100-pair set with Cohen's κ ≥ 0.6.

This is gated behind the v1 bench and runs offline. Gemini Flash is the
teacher; the local VLM is the deploy-time student so the bench harness
stays offline-capable.

## Acceptance

- `python -m svrptw.vivrp.compound --instance ... --solution ...` prints a
  populated `CompoundAssessment` JSON.
- Compound score is monotone in operational cost when all other components
  tie (sanity check).
- The compound leaderboard (`bench/leaderboard.py --compound`) shows a
  different Pareto frontier than cost-only, demonstrating the metric is
  non-redundant.

## Non-goals

- A single "true" quality metric. The compound is a configurable opinion.
- Real-time customer feedback simulation. Synthetic proxies only.

## Dependencies

- SPEC-5-VIVRP-01 (the existing VLM backend pipeline).
- SPEC-3-PORTFOLIO-01 (so we have a portfolio of solvers to compare).
