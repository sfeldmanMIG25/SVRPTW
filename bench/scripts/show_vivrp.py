"""Quick formatter for ViVRP scores from a bench JSON."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "bench/runs/v1_smoke_vivrp_local.json"
d = json.load(open(path))
print(f"{'instance':<26} {'solver':<18} {'cost':>8} {'ovr':>4} {'clst':>4} {'geom':>4} {'intp':>4}  notes")
print("-" * 110)
for r in d["rows"]:
    inst = r["instance_id"]
    s = r["solver"]
    cost = r.get("operational_cost", float("nan"))
    ovr = r.get("vivrp_overall", "-")
    clst = r.get("vivrp_clustering", "-")
    geom = r.get("vivrp_geometry", "-")
    intp = r.get("vivrp_interpretability", "-")
    notes = (r.get("vivrp_notes") or "")[:50]
    print(f"{inst:<26} {s:<18} {cost:>8.1f} {ovr:>4} {clst:>4} {geom:>4} {intp:>4}  {notes}")
