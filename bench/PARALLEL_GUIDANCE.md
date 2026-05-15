# Bench parallelization policy

**Mandate (user directive, 2026-05-14)**: bench scripts MUST use parallel
processes to maximize CPU utilization. Sequential for-loops over instances
waste >70% of available compute on this machine (laptop with 8-16 cores,
RTX 3070 Ti).

## Use the helper

```python
from bench.parallel import map_instances

# Each task is (instance_path, params) -> dict result.
def solve_one(args):
    ip, fixed_cost = args
    from svrptw.config import Settings, Economics
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as ppw
    inst = load_instance(str(ip))
    s = Settings(economics=Economics(per_route_fixed_cost=fixed_cost))
    sol = ppw.solve(inst, s, budget_seconds=30.0)
    return {"instance_id": inst.instance_id, "fixed_cost": fixed_cost,
            "cost": sol.metrics["operational_cost"]}

tasks = [(ip, fc) for ip in paths for fc in (0.0, 25.0, 100.0)]
rows = map_instances(solve_one, tasks, max_workers=4)
```

## Rules

1. **Default to 4 workers.** PyVRP uses C++ threading internally so 4
   processes saturates ~12-16 logical cores. Going higher invites
   cache thrashing.
2. **Each worker reloads heavyweight modules** (numba, torch, GART).
   First call per worker pays ~5-15s warm-up. For benches <30 solves,
   this overhead can dominate — set `max_workers=2` for small benches.
3. **No shared state across workers.** Pass instance paths, not loaded
   Instance objects (pickling cost). Each worker loads its own.
4. **Write JSON FIRST in main.** After futures resolve and before any
   summary print — format-string crashes have killed bench data twice
   this session.
5. **Capture stdout per task.** Workers can't print interleavedly;
   collect partial logs in the returned dict if you want progress.

## Backward compatibility

Existing sequential bench scripts (v1_n200_*, v2_*) still work — but
new scripts MUST use `bench.parallel.map_instances` unless the bench
is genuinely sequential (e.g., the result of step N feeds into step
N+1's input).
