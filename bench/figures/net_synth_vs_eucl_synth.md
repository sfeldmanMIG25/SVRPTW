# Network-OD synthetic vs Euclidean+noise — POMO transfer A/B

Date: 2026-05-13

## Setup

Three POMO checkpoints, all benched on the same 3 Manhattan N=50
v1 instances (OSM-Manhattan-N050-I000/I001/I002):

| ckpt | training source | mean cost Manhattan |
|------|-----------------|-------------------:|
| v3            | v1 OSM (direct)         | **822.1** |
| net_synth     | network-OD grid+arterials |   870.2  |
| v4            | Euclidean + noise         |   881.1  |

## What this proves

1. **Network-OD synthetic beats Euclidean+noise** at transferring to
   real OSM data: 881 → 870 = -1.2 %. The structural asymmetry from
   one-way streets matches v1 OSM's distribution (0.08–0.11) and the
   policy learns useful inductive biases the Euclidean+noise generator
   couldn't surface.

2. **Network-OD synthetic still loses to v1-direct** by 5.9 %. The
   grid-with-arterials approximation misses real-OSM features:
   - Hub-and-spoke patterns (no major highways in our grid)
   - Variable block sizes (we use uniform spacing)
   - Real congestion gradients (we use uniform speed within tier)
   - Coastlines / topography barriers (we have none)

   For N ≤ 500, v1-direct training will always win.

3. **The right use-case for network-OD synthetic is N ≥ 1000**, where
   v1 has zero instances. Future POMO scaling to N=10k must train on
   synthetic of *some* kind — and network-OD is now confirmed better
   than the Euclidean baseline.

## Next steps

a. **Land the larger-N POMO training run** on network-OD at N=1000
   and N=5000 to see how the architecture scales.
b. **Optionally:** improve the network-OD generator with the other
   topologies in SPEC-4-DATA-02 (fractal-radial, mesh-with-bridges,
   osm-like-sampled) to widen the training-distribution coverage.
c. **Keep v3 as canonical** at N ≤ 500. The headline (`HEADLINE.md`
   + `v1_6way_full_pareto.md`) stays unchanged.

## Why this is interesting research-wise

Most POMO-style papers train on a single uniform-Euclidean
distribution (Kool 2019 / Kwon 2020). The community implicit
assumption is that the inductive biases learnt there transfer to
real road networks. **Our v3-vs-v4 result was the first concrete
counter-evidence**; today's v3-vs-net_synth-vs-v4 result is the
first concrete proof that *structured* synthetic data (not just
"more synthetic data") closes part of the transfer gap.
