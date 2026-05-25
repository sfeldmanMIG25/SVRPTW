"""Build a v3 instance set: hetero fleet + EU561 breaks + chargers.

Generates 20 instances at N=100 and 20 at N=200 with deterministic seeds.
Writes to instances/v3/ with a manifest.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from svrptw.instances_gen.synthetic import generate
from svrptw.io import save_instance


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="instances/v3")
    p.add_argument("--per-n", type=int, default=20)
    p.add_argument("--n-list", default="100,200")
    p.add_argument("--master-seed", type=int, default=271828)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_list = [int(x) for x in args.n_list.split(",")]
    rows: list[str] = ["| instance_id | N | asym | n_classes | breaks | chargers |",
                       "|---|---|---|---|---|---|"]
    written: list[Path] = []
    for N in n_list:
        for i in range(args.per_n):
            seed = args.master_seed + N * 1000 + i
            inst = generate(
                N=N, seed=seed,
                hetero_fleet=True,
                breaks_regime="EU561",
                num_chargers=max(5, N // 30),
            )
            # Rename to clearly mark v3.
            inst.instance_id = f"V3-N{N:03d}-I{i:03d}"
            path = save_instance(inst, out_dir)
            written.append(path)
            n_classes = len({v.vclass for v in (inst.vehicles or [])})
            n_charge = len(inst.chargers or [])
            rows.append(
                f"| {inst.instance_id} | {N} | {inst.asymmetry_score:.3f} | "
                f"{n_classes} | {inst.breaks.name if inst.breaks else 'none'} | "
                f"{n_charge} |"
            )
            print(f"  wrote {inst.instance_id}")

    manifest = out_dir / "manifest.sha256"
    with open(manifest, "w", encoding="utf-8") as f:
        for jp in sorted(written):
            f.write(f"{_hash_file(jp)}  {jp.relative_to(out_dir)}\n")
    report = out_dir / "REPORT.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"# instances/v3 (SPEC-0-INST-03)\n\nWrote {len(written)} instances.\n\n")
        f.write("\n".join(rows))
        f.write("\n")
    print(f"\nmanifest: {manifest}\nreport:   {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
