"""Convert the iter-5m smoke instance from .npy to .npz format + fix paths."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "D:/SVRPTW")


def main() -> int:
    matrices_dir = Path("instances/v1_large/matrices")
    iid = "OSM-Manhattan-N0100-I000_smoke"
    # Convert .npy -> .npz with key 'm'
    for kind in ("dist", "time"):
        npy = matrices_dir / f"{iid}__{kind}.npy"
        npz = matrices_dir / f"{iid}__{kind}.npz"
        if npy.exists():
            arr = np.load(npy)
            np.savez_compressed(npz, m=arr)
            print(f"converted {npy.name} -> {npz.name} (shape {arr.shape})")
            npy.unlink()
    # Update JSON to point at .npz with proper relative paths
    jp = Path(f"instances/v1_large/{iid}.json")
    j = json.loads(jp.read_text())
    j["travel_time_matrix_path"] = f"matrices/{iid}__time.npz"
    j["travel_distance_matrix_path"] = f"matrices/{iid}__dist.npz"
    jp.write_text(json.dumps(j, indent=2))
    print(f"updated {jp}")

    # Verify load
    from svrptw.io import load_instance
    inst = load_instance(str(jp))
    cost_mi = float(inst.travel_dist[0, 1])
    time_min = float(inst.travel_time[0, 1])
    print(f"loaded: {inst.instance_id} N={inst.num_customers} K={inst.num_vehicles}")
    print(f"  travel_dist[0,1]: {cost_mi:.3f} mi")
    print(f"  travel_time[0,1]: {time_min:.3f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
