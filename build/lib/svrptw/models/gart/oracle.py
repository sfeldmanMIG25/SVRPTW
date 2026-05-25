"""TSP/ATSP oracle wrappers for GART calibration.

LKH-3 for asymmetric TSP ground truth, Concorde (via WSL) for symmetric TSP.
Used to calibrate the asymmetry-correction beta in SPEC-1-GART-01.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

LKH_EXE = Path(os.environ.get("SVRPTW_LKH_EXE", r"C:\LKH\LKH-3.exe"))
WSL = "wsl.exe"


def lkh_atsp_tour_length(D: np.ndarray, time_limit: float = 5.0) -> float:
    """Optimal-ish ATSP tour length under LKH-3.  D is an (n, n) integer-ish
    asymmetric matrix; we scale by 100 internally for int representation."""
    n = D.shape[0]
    Di = (D * 100).round().astype(np.int64)
    np.fill_diagonal(Di, 0)
    with tempfile.TemporaryDirectory(prefix="lkh_atsp_") as td:
        work = Path(td)
        prob = work / "p.atsp"
        par  = work / "p.par"
        lines = [
            f"NAME : ATSP_{n}", "TYPE : ATSP", f"DIMENSION : {n}",
            "EDGE_WEIGHT_TYPE : EXPLICIT",
            "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
            "EDGE_WEIGHT_SECTION",
        ]
        for i in range(n):
            lines.append(" ".join(str(int(v)) for v in Di[i]))
        lines.append("EOF")
        prob.write_text("\n".join(lines))

        tour_path = work / "tour.txt"
        par.write_text("\n".join([
            f"PROBLEM_FILE = {prob}",
            f"TOUR_FILE = {tour_path}",
            f"TIME_LIMIT = {int(max(1, time_limit))}",
            "RUNS = 1",
            "TRACE_LEVEL = 0",
        ]) + "\n")
        try:
            r = subprocess.run([str(LKH_EXE), str(par)], cwd=work,
                               capture_output=True, text=True, timeout=time_limit * 3 + 30)
        except subprocess.TimeoutExpired:
            return float("nan")
        m = re.search(r"Cost\.min\s*=\s*(\d+)", r.stdout)
        if not m:
            return float("nan")
        return float(int(m.group(1)) / 100.0)


def concorde_tsp_tour_length(coords: np.ndarray) -> float:
    """Symmetric TSP via Concorde (WSL).  coords is (n, 2)."""
    n = coords.shape[0]
    with tempfile.TemporaryDirectory(prefix="concorde_") as td:
        work = Path(td)
        prob = work / "p.tsp"
        lines = [f"NAME : T{n}", "TYPE : TSP", f"DIMENSION : {n}",
                 "EDGE_WEIGHT_TYPE : EUC_2D", "NODE_COORD_SECTION"]
        for i, (x, y) in enumerate(coords, start=1):
            lines.append(f"{i} {x:.6f} {y:.6f}")
        lines.append("EOF")
        prob.write_text("\n".join(lines))
        win_path = str(prob).replace("\\", "/")
        # Map C: -> /mnt/c
        if len(win_path) > 1 and win_path[1] == ":":
            drive = win_path[0].lower()
            wsl_path = f"/mnt/{drive}{win_path[2:]}"
        else:
            wsl_path = win_path
        try:
            r = subprocess.run([WSL, "-e", "bash", "-c",
                                f"cd /tmp && cp '{wsl_path}' /tmp/p.tsp && /usr/local/bin/concorde /tmp/p.tsp 2>&1 | tail -20"],
                               capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return float("nan")
        m = re.search(r"Optimal Solution:\s*([\d.]+)", r.stdout)
        if not m:
            return float("nan")
        return float(m.group(1))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--mode", choices=["atsp", "tsp"], default="atsp")
    args = p.parse_args()
    rng = np.random.default_rng(0)
    if args.mode == "atsp":
        M = rng.uniform(1, 20, size=(args.n, args.n))
        np.fill_diagonal(M, 0)
        L = lkh_atsp_tour_length(M)
        print(f"LKH-3 ATSP optimal: {L:.2f}")
    else:
        coords = rng.uniform(0, 100, size=(args.n, 2))
        L = concorde_tsp_tour_length(coords)
        print(f"Concorde TSP optimal: {L:.2f}")
