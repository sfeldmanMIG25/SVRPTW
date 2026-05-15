# Pareto report

_Generated from 720 rows across 3 N-bins._


## N = 50

| solver | n_inst | operational_cost | wall_clock_seconds | Pareto |
|---|---|---|---|---|
| `auction_gart` | 40 | 794.85 | 3.71 | **Y** |
| `greedy` | 40 | 883.06 | 0.00 | **Y** |
| `ortools_gart@10` | 40 | 818.92 | 7.27 |  |
| `lkh3@1` | 40 | 902.63 | 1.24 |  |
| `ortools@10` | 40 | 953.61 | 10.00 |  |
| `ortools@1` | 40 | 949.66 | 1.00 |  |

**Hypervolume:** 2469   (lower-cost & higher-quality is better)


## N = 100

| solver | n_inst | operational_cost | wall_clock_seconds | missed_deliveries | Pareto |
|---|---|---|---|---|---|
| `auction_gart` | 40 | 1543.29 | 8.71 | 0.00 | **Y** |
| `ortools_gart@10` | 40 | 1548.92 | 8.42 | 0.00 | **Y** |
| `ortools@10` | 40 | 1810.30 | 10.00 | 0.00 |  |
| `ortools@1` | 40 | 1798.88 | 1.00 | 0.00 |  |
| `greedy` | 40 | 1634.50 | 0.01 | 0.00 | **Y** |
| `lkh3@1` | 40 | 902.06 | 1.53 | 0.03 | **Y** |

**Hypervolume:** 1.097e+04   (lower-cost & higher-quality is better)


## N = 200

| solver | n_inst | operational_cost | wall_clock_seconds | missed_deliveries | Pareto |
|---|---|---|---|---|---|
| `auction_gart` | 40 | 2886.99 | 20.55 | 0.00 | **Y** |
| `ortools_gart@10` | 40 | 3150.19 | 9.61 | 0.00 |  |
| `ortools@10` | 40 | 3429.76 | 10.01 | 0.00 |  |
| `ortools@1` | 40 | 3422.05 | 1.00 | 0.00 |  |
| `greedy` | 40 | 2940.84 | 0.02 | 0.00 | **Y** |
| `lkh3@1` | 40 | 29879.86 | 1.63 | 28.93 |  |

**Hypervolume:** 2.151e+07   (lower-cost & higher-quality is better)

