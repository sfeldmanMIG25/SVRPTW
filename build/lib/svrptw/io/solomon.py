"""Solomon CVRPTW benchmark loader → svrptw.io.Instance.

Format reference: Solomon's text file is `VEHICLE / NUMBER CAPACITY` then a
`CUSTOMER` section with rows `id x y demand ready due service`. Customer 0
is the depot. Euclidean distance is the standard cost; travel time equals
Euclidean distance (Solomon's convention — speed = 1).

Reading a Solomon instance produces a SYMMETRIC Euclidean
travel matrix (asymmetry_score = 0.0). This makes it directly comparable to
PyVRP and LKH-3 published numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from svrptw.io.instance import Customer, Depot, Instance


def load_solomon(path: str | Path) -> Instance:
    """Parse a Solomon-format text file. Returns an Instance with Euclidean
    symmetric travel_time = travel_dist (the academic convention)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]

    # First non-empty line is the instance name (e.g. "C101").
    name = lines[0].strip()
    # Find VEHICLE section.
    i = next(k for k, ln in enumerate(lines) if "VEHICLE" in ln.upper())
    # VEHICLE line "NUMBER CAPACITY" header is i+1; data is i+2.
    parts = lines[i + 2].split()
    num_vehicles = int(parts[0])
    capacity = int(parts[1])

    # Find CUSTOMER section.
    j = next(k for k, ln in enumerate(lines) if "CUSTOMER" in ln.upper())
    # Data rows start after the column-header line (j+2).
    data_lines = [ln for ln in lines[j + 2:] if ln.strip() and ln.split()[0].isdigit()]

    depot_row = None
    customers: list[Customer] = []
    coords: list[tuple[float, float]] = []
    for row in data_lines:
        parts = row.split()
        cid     = int(parts[0])
        x       = float(parts[1])
        y       = float(parts[2])
        demand  = int(parts[3])
        ready   = int(parts[4])
        due     = int(parts[5])
        service = int(parts[6])
        coords.append((x, y))
        if cid == 0:
            depot_row = (x, y, ready, due)
        else:
            customers.append(Customer(
                id=cid, node_id=cid,
                x=x, y=y,
                demand=demand, ready=ready, due=due, service=service,
            ))

    if depot_row is None:
        raise ValueError(f"Solomon file {p} missing depot row (customer 0)")

    dx, dy, dready, ddue = depot_row
    depot = Depot(node_id=0, x=dx, y=dy, ready=dready, due=ddue)

    # Build symmetric Euclidean travel-time / dist matrices (Solomon's
    # convention: speed = 1, so time == distance).
    n = len(coords)
    pts = np.array(coords, dtype=np.float64)
    diff = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt((diff * diff).sum(-1))
    T = D.copy()

    inst = Instance(
        instance_id=f"Solomon-{name}",
        city="Solomon",
        num_customers=len(customers),
        num_vehicles=num_vehicles,
        vehicle_capacity=capacity,
        depot=depot,
        customers=customers,
        travel_time=T,
        travel_dist=D,
        asymmetry_score=0.0,
        seed=0,
    )
    return inst


__all__ = ["load_solomon"]
