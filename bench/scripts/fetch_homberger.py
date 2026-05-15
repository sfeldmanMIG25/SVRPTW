"""Fetch Gehring-Homberger extended Solomon (200/400 customer) instances.

Source: ML4VRP/ML4VRP2023 GitHub mirror (text format identical to Solomon).
Naming: <class>_<size_idx>_<seq>, where size_idx ∈ {2: 200, 4: 400}.
60 instances per scale (10 per Solomon-class subgroup).
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


CLASSES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
SIZES = {200: 2, 400: 4}
BASE = "https://raw.githubusercontent.com/ML4VRP/ML4VRP2023/main/Instances/text"


def main() -> int:
    total = 0; fail = 0
    for size, size_idx in SIZES.items():
        out_dir = Path(f"instances/homberger/{size}")
        out_dir.mkdir(parents=True, exist_ok=True)
        names = [f"{cls}_{size_idx}_{seq}" for cls in CLASSES for seq in range(1, 11)]
        for i, name in enumerate(names, 1):
            target = out_dir / f"{name}.txt"
            if target.exists() and target.stat().st_size > 1000:
                total += 1
                continue
            url = f"{BASE}/Customer{size}/{name}.txt"
            try:
                r = urllib.request.urlopen(url, timeout=15)
                body = r.read()
                target.write_bytes(body)
                total += 1
                sys.stdout.write(f"  [{size}/{i:>2}] {name} ok ({len(body)}b)\n")
            except Exception as e:
                fail += 1
                sys.stdout.write(f"  [{size}/{i:>2}] {name} FAIL: {e}\n")
            sys.stdout.flush()
    print(f"\nfetched {total} (failed {fail}) across {len(SIZES)} sizes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
