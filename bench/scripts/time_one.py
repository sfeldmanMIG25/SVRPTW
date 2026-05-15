"""Hot timing of one auction solve, ignoring import / GART cold start."""
import sys
import time

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import auction_gart

inst = load_instance(sys.argv[1] if len(sys.argv) > 1 else "instances/v1/OSM-Manhattan-N100-I000.json")
# Warm GART singleton
auction_gart.solve(inst, Settings())
t0 = time.perf_counter()
sol = auction_gart.solve(inst, Settings())
dt = time.perf_counter() - t0
print(f"cost={sol.metrics['operational_cost']:.2f}  time={dt:.2f}s  miss={int(sol.metrics['missed_deliveries'])}")
