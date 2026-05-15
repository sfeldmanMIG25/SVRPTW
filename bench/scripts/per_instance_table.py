"""Per-instance cost table from a bench JSON file."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "bench/runs/v1_smoke_with_relocate.json"
d = json.load(open(path))
by_inst = {}
solvers = []
for r in d["rows"]:
    by_inst.setdefault(r["instance_id"], {})[r["solver"]] = r.get("operational_cost")
    if r["solver"] not in solvers:
        solvers.append(r["solver"])

print(f"{'instance':<32}", " ".join(f"{s:>14}" for s in solvers))
for iid, row in sorted(by_inst.items()):
    vals = " ".join(f"{row.get(s, float('nan')):>14.2f}" for s in solvers)
    print(f"{iid:<32}", vals)

print(f"\n{'MEAN':<32}", " ".join(
    f"{sum(by_inst[i].get(s, 0) for i in by_inst) / len(by_inst):>14.2f}" for s in solvers
))
