"""Profile a single auction_gart solve to find the hot path."""
import cProfile
import pstats
import sys
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import auction_gart

inst_path = sys.argv[1] if len(sys.argv) > 1 else "instances/v1/OSM-Manhattan-N100-I000.json"
inst = load_instance(inst_path)

profiler = cProfile.Profile()
profiler.enable()
auction_gart.solve(inst, Settings())
profiler.disable()

stats = pstats.Stats(profiler).sort_stats("cumulative")
stats.print_stats(20)
