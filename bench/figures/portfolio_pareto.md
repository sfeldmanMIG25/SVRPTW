# Pareto report

_Generated from 180 rows across 3 N-bins._


## N = 50

| solver | n_inst | operational_cost | wall_clock_seconds | Pareto |
|---|---|---|---|---|
| `greedy` | 10 | 800.63 | 0.00 | **Y** |
| `regret_k@10` | 10 | 747.92 | 1.93 | **Y** |
| `auction_gart` | 10 | 708.74 | 3.93 | **Y** |
| `portfolio@10` | 10 | 707.69 | 4.09 | **Y** |
| `portfolio@30` | 10 | 707.69 | 4.09 |  |
| `lkh3@1` | 10 | 829.51 | 1.24 |  |

**Hypervolume:** 782.7   (lower-cost & higher-quality is better)


## N = 100

| solver | n_inst | operational_cost | wall_clock_seconds | Pareto |
|---|---|---|---|---|
| `lkh3@1` | 10 | 807.87 | 1.62 | **Y** |
| `greedy` | 10 | 1494.69 | 0.00 | **Y** |
| `regret_k@10` | 10 | 1451.09 | 4.01 |  |
| `portfolio@30` | 10 | 1366.09 | 5.29 |  |
| `portfolio@10` | 10 | 1365.97 | 5.29 |  |
| `auction_gart` | 10 | 1364.86 | 8.90 |  |

**Hypervolume:** 7170   (lower-cost & higher-quality is better)


## N = 200

| solver | n_inst | operational_cost | wall_clock_seconds | missed_deliveries | Pareto |
|---|---|---|---|---|---|
| `regret_k@10` | 10 | 2633.74 | 9.75 | 0.00 |  |
| `portfolio@30` | 10 | 2570.90 | 8.68 | 0.00 | **Y** |
| `portfolio@10` | 10 | 2574.42 | 8.13 | 0.00 | **Y** |
| `auction_gart` | 10 | 2562.61 | 22.01 | 0.00 | **Y** |
| `greedy` | 10 | 2633.74 | 0.02 | 0.00 | **Y** |
| `lkh3@1` | 10 | 16722.36 | 1.67 | 16.00 |  |

**Hypervolume:** 6.728e+06   (lower-cost & higher-quality is better)

