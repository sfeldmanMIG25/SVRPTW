"""Asymmetric VRPTW instance loader/saver.

SPEC-0-INST-01 introduced the base schema; SPEC-0-INST-03 extended it
with optional heterogeneous-fleet, breaks, and renewable-charger
fields.  All v3 fields are optional so v1 instances still load
unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Customer:
    id: int
    node_id: int
    x: float
    y: float
    demand: int
    ready: int
    due: int
    service: int


@dataclass
class Depot:
    node_id: int
    x: float
    y: float
    ready: int
    due: int


# --------- SPEC-0-INST-03 additions ---------


@dataclass
class VehicleSpec:
    vclass: str = "van"
    capacity: int = 50
    fuel_capacity: float | None = None     # miles of range; None = ICE/unlimited
    recharge_rate: float | None = None     # miles/min at a charger
    allow_zones: tuple[str, ...] = ()      # empty = all zones OK


@dataclass
class BreakRegime:
    name: str = "none"                     # "none" | "EU561" | "FMCSA"
    drive_limit_min: int = 270
    break_duration_min: int = 45
    rest_node_ids: tuple[int, ...] = ()


@dataclass
class Charger:
    node_id: int
    rate: float = 5.0                      # miles recharged per minute
    types: tuple[str, ...] = ("ev",)


@dataclass
class Instance:
    instance_id: str
    city: str
    num_customers: int
    num_vehicles: int
    vehicle_capacity: int
    depot: Depot
    customers: list[Customer]
    travel_time: np.ndarray
    travel_dist: np.ndarray
    asymmetry_score: float
    seed: int
    schema_version: str = "1.0"
    # SPEC-0-INST-03 (all optional; None / empty = "use legacy homogeneous fleet")
    vehicles: list[VehicleSpec] | None = None
    breaks: BreakRegime | None = None
    chargers: list[Charger] | None = None
    zones: dict | None = None              # reserved for SPEC-0-INST-02 cross-pollination

    @property
    def n(self) -> int:
        return self.num_customers


def _resolve(base: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (base / p)


def load_instance(json_path: str | Path) -> Instance:
    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    base = json_path.parent
    tm = np.load(_resolve(base, raw["travel_time_matrix_path"]))["m"]
    dm = np.load(_resolve(base, raw["travel_distance_matrix_path"]))["m"]
    # SPEC-0-INST-03 optional fields (all missing in v1).
    # Tuple-typed dataclass fields round-trip through JSON as lists; coerce
    # back so identity tests like `tuple == tuple` work.
    vehicles = None
    if "vehicles" in raw and raw["vehicles"] is not None:
        vehicles = [
            VehicleSpec(
                vclass=v.get("vclass", "van"),
                capacity=v.get("capacity", 50),
                fuel_capacity=v.get("fuel_capacity"),
                recharge_rate=v.get("recharge_rate"),
                allow_zones=tuple(v.get("allow_zones") or ()),
            )
            for v in raw["vehicles"]
        ]
    breaks = None
    if raw.get("breaks"):
        rb = raw["breaks"]
        breaks = BreakRegime(
            name=rb.get("name", "none"),
            drive_limit_min=rb.get("drive_limit_min", 270),
            break_duration_min=rb.get("break_duration_min", 45),
            rest_node_ids=tuple(int(x) for x in (rb.get("rest_node_ids") or ())),
        )
    chargers = None
    if "chargers" in raw and raw["chargers"] is not None:
        chargers = [
            Charger(
                node_id=int(c["node_id"]),
                rate=float(c.get("rate", 5.0)),
                types=tuple(c.get("types") or ("ev",)),
            )
            for c in raw["chargers"]
        ]
    return Instance(
        instance_id=raw["instance_id"],
        city=raw["city"],
        num_customers=raw["num_customers"],
        num_vehicles=raw["num_vehicles"],
        vehicle_capacity=raw["vehicle_capacity"],
        depot=Depot(**raw["depot"]),
        customers=[Customer(**c) for c in raw["customers"]],
        travel_time=tm,
        travel_dist=dm,
        asymmetry_score=raw["asymmetry_score"],
        seed=raw["generator_seed"],
        schema_version=raw.get("schema_version", "1.0"),
        vehicles=vehicles,
        breaks=breaks,
        chargers=chargers,
        zones=raw.get("zones"),
    )


def save_instance(inst: Instance, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    matrices = out_dir / "matrices"
    matrices.mkdir(parents=True, exist_ok=True)
    tm_path = matrices / f"{inst.instance_id}.npz"
    dm_path = matrices / f"{inst.instance_id}_dist.npz"
    np.savez_compressed(tm_path, m=inst.travel_time)
    np.savez_compressed(dm_path, m=inst.travel_dist)
    body = {
        "instance_id": inst.instance_id,
        "city": inst.city,
        "num_customers": inst.num_customers,
        "num_vehicles": inst.num_vehicles,
        "vehicle_capacity": inst.vehicle_capacity,
        "depot": inst.depot.__dict__,
        "customers": [c.__dict__ for c in inst.customers],
        "travel_time_matrix_path": str(tm_path.relative_to(out_dir)),
        "travel_distance_matrix_path": str(dm_path.relative_to(out_dir)),
        "asymmetry_score": inst.asymmetry_score,
        "generator_seed": inst.seed,
        "schema_version": inst.schema_version,
    }
    if inst.vehicles is not None:
        body["vehicles"] = [{**v.__dict__, "allow_zones": list(v.allow_zones)} for v in inst.vehicles]
    if inst.breaks is not None:
        body["breaks"] = {**inst.breaks.__dict__, "rest_node_ids": list(inst.breaks.rest_node_ids)}
    if inst.chargers is not None:
        body["chargers"] = [{**c.__dict__, "types": list(c.types)} for c in inst.chargers]
    if inst.zones is not None:
        body["zones"] = inst.zones
    json_path = out_dir / f"{inst.instance_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
    return json_path
