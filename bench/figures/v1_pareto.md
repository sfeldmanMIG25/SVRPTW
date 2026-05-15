# Pareto report

_Generated from 960 rows across 3 N-bins._


## N = 50

| solver | n_inst | operational_cost | wall_clock_seconds | Pareto |
|---|---|---|---|---|
| `greedy` | 40 | 883.06 | 0.00 | **Y** |
| `portfolio@30` | 40 | 748.92 | 3.01 | **Y** |
| `portfolio@10` | 40 | 749.34 | 3.04 |  |
| `auction_gart` | 40 | 752.66 | 4.08 |  |
| `ortools_gart@10` | 40 | 821.03 | 7.31 |  |
| `lkh3@1` | 40 | 889.93 | 1.29 |  |
| `ortools@10` | 40 | 952.41 | 10.00 |  |
| `ortools@1` | 40 | 947.78 | 1.00 |  |

**Hypervolume:** 2883   (lower-cost & higher-quality is better)


## N = 100

| solver | n_inst | operational_cost | wall_clock_seconds | missed_deliveries | Pareto |
|---|---|---|---|---|---|
| `portfolio@30` | 40 | 1481.21 | 5.23 | 0.00 | **Y** |
| `portfolio@10` | 40 | 1482.71 | 5.08 | 0.00 | **Y** |
| `auction_gart` | 40 | 1455.76 | 11.13 | 0.00 | **Y** |
| `ortools_gart@10` | 40 | 1544.44 | 8.55 | 0.00 |  |
| `ortools@10` | 40 | 1806.76 | 10.00 | 0.00 |  |
| `ortools@1` | 40 | 1796.17 | 1.00 | 0.00 |  |
| `greedy` | 40 | 1634.50 | 0.01 | 0.00 | **Y** |
| `lkh3@1` | 40 | 915.80 | 1.52 | 0.07 | **Y** |

**Hypervolume:** 1.243e+04   (lower-cost & higher-quality is better)


## N = 200

| solver | n_inst | operational_cost | wall_clock_seconds | missed_deliveries | Pareto |
|---|---|---|---|---|---|
| `portfolio@30` | 40 | 2833.12 | 12.79 | 0.00 | **Y** |
| `portfolio@10` | 40 | 2840.48 | 9.29 | 0.00 | **Y** |
| `auction_gart` | 40 | 2803.87 | 26.70 | 0.00 | **Y** |
| `ortools_gart@10` | 40 | 3169.59 | 9.62 | 0.00 |  |
| `ortools@10` | 40 | 3431.65 | 10.01 | 0.00 |  |
| `ortools@1` | 40 | 3423.82 | 1.01 | 0.00 |  |
| `greedy` | 40 | 2940.84 | 0.02 | 0.00 | **Y** |
| `lkh3@1` | 40 | 36159.56 | 1.63 | 35.15 |  |

**Hypervolume:** 4.19e+07   (lower-cost & higher-quality is better)

