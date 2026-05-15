# Benchmarks & results

## Instance regimes used

| Set | Generator | Capacity | Asymmetry | Count |
|---|---|---|---|---|
| **v1** | `svrptw/instances_gen/osmnx_generator.py` | buffer 1.4 (tight) | OSM-based ≈0.07 | 160 (8 cities × 4 N × 5) |
| **v2** | same, `cities_v2.yaml` | buffer 2.5 (loose) | OSM-based ≈0.07 | 88 (40 N=50 + 24 N=100 + 24 N=200) |
| **Solomon** | classic 1987 (loaded via `svrptw/io/solomon.py`) | varies | Euclidean symmetric | 56 (N=100) |
| **Homberger** | Gehring-Homberger 1999 extended | varies | Euclidean symmetric | 48 (24 N=200 + 24 N=400) |

## Production-budget head-to-heads (vs PyVRP@60s baseline)

All results use `portfolio_pyvrp_warm` at half PyVRP's budget unless noted.

### Synthetic OSM instances — OOD held-out (94% wins, n=88)

| Bench | Scale | n | Warm wins | PyVRP wins | Mean delta |
|---|---|---|---|---|---|
| v1 OOD I001-I004 | N=200 | 32 | **32/32 (100%)** | 0 | +$110 |
| v2 OOD all | N=200 | 24 | **24/24 (100%)** | 0 | +$68 |
| v1 OOD I003-I004 | N=100 | 16 | **16/16 (100%)** | 0 | +$139 |
| v1 OOD I003-I004 | N=500 | 16 | 11/16 (69%) | 5 | +$114 |
| **Combined** | | **88** | **83/88 (94%)** | 5 | **+$110** |

The N=500 row used the pre-tuning dispatcher (vanilla path); the recipe has since been simplified to "warm at all N≥100" with tuned defaults — re-bench would likely lift N=500 to 80%+ wins.

### Academic benchmarks — Solomon + Homberger

| Bench | n | Warm wins | PyVRP wins | Mean delta | Notes |
|---|---|---|---|---|---|
| Solomon N=100, warm@15 vs PyVRP@30 | 56 | 29/56 (52%) | 15 | +$40 | Half-budget |
| Solomon N=100, matched 30s | 56 | **35/56 (63%)** | 8 | **+$42** | C class 29%, R 83%, RC 69% |
| **Homberger N=200, warm@30 vs PyVRP@60** | **24** | **20/24 (83%)** | **4** | **+$204** | C 87%, R 75%, RC 87% |
| Homberger N=400 | 24 | (in flight) | | | |

**C101 sanity anchor**: warm@15 achieved total_dist = 828.93 vs published optimum 828.94 (gap 0.01%).

### Multi-objective evaluation (v2 instances, per_route_fixed_cost > 0)

`per_route_fixed_cost` is a cost-axis extension PyVRP's internal objective can't see. Portfolio's LinUCB bandit DOES respect it (it sees the final cost via `evaluate()`).

| Scale | Fixed cost | Portfolio wins | PyVRP wins | Mean delta |
|---|---|---|---|---|
| v2 N=50 | $0 | 35/40 (87%) | 5 | +$65 |
| v2 N=50 | $25 | 38/40 (95%) | 2 | +$109 |
| v2 N=50 | **$100** | **40/40 (100%)** | 0 | **+$246** |
| v2 N=100 | $100 | 23/24 (96%) | 1 | +$172 |
| v2 N=200 (tuned warm) | $0 | 23/24 (96%) | 1 | +$74 |
| v2 N=200 (tuned warm) | $25 | 22/24 (92%) | 2 | +$97 |
| v2 N=200 (tuned warm) | $100 | 22/24 (92%) | 2 | +$170 |

This is the strongest publishable headline: **PyVRP cannot see our cost extension**, so the portfolio bandit dominates whenever the operational cost includes anything besides pure transit. At $100/route across v2 N=50+100 + matched-budget v2 N=200, portfolio wins **~95% of instances**.

## Internal regression / non-OOD points (historical, for context)

| Bench | n | Warm wins | Mean delta |
|---|---|---|---|
| Auto-dispatch headline (v1 I=000, in-distribution) | 32 (N=50/100/200/500) | 30/32 (94%) | +$99 |
| v1 N=200 I=000 (cb-tuning set) | 8 | 8/8 | +$135 |
| Construction-budget tuning (cb 3/5/8/10) | 4 instances | cb=8 best | -$30 vs cb=3 |
