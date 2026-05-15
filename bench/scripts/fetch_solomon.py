"""Fetch the 56 Solomon N=100 CVRPTW benchmark instances.

Source: iRB-Lab/py-ga-VRPTW GitHub mirror (vrplib's CVRPLIB URL is dead).
Stores Solomon-format text files under instances/solomon/<NAME>.txt.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/iRB-Lab/py-ga-VRPTW/master/data/text/"
NAMES = [
    *[f"C10{i}" for i in range(1, 10)],     # C101..C109
    *[f"C20{i}" for i in range(1, 9)],      # C201..C208
    *[f"R10{i}" for i in range(1, 10)],     # R101..R109
    "R110", "R111", "R112",
    *[f"R20{i}" for i in range(1, 10)],     # R201..R209
    "R210", "R211",
    *[f"RC10{i}" for i in range(1, 9)],     # RC101..RC108
    *[f"RC20{i}" for i in range(1, 9)],     # RC201..RC208
]


def main() -> int:
    out_dir = Path("instances/solomon")
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = bad = 0
    for i, name in enumerate(NAMES, 1):
        target = out_dir / f"{name}.txt"
        if target.exists() and target.stat().st_size > 1000:
            ok += 1
            continue
        url = BASE + f"{name}.txt"
        try:
            r = urllib.request.urlopen(url, timeout=15)
            body = r.read()
            target.write_bytes(body)
            ok += 1
            sys.stdout.write(f"  [{i:>2}/{len(NAMES)}] {name} ok ({len(body)}b)\n")
        except Exception as e:
            bad += 1
            sys.stdout.write(f"  [{i:>2}/{len(NAMES)}] {name} FAIL: {e}\n")
        sys.stdout.flush()
    print(f"\nfetched {ok}/{len(NAMES)} (failed {bad})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
