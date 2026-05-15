"""Round-trip + back-compat for SPEC-0-INST-03 fields."""
import numpy as np

from svrptw.instances_gen.synthetic import generate
from svrptw.io import BreakRegime, Charger, VehicleSpec, load_instance, save_instance


def test_v3_roundtrip_full(tmp_path):
    inst = generate(N=20, seed=0,
                    hetero_fleet=True, breaks_regime="EU561", num_chargers=3)
    assert inst.vehicles is not None
    assert inst.breaks is not None and inst.breaks.name == "EU561"
    assert inst.chargers is not None and len(inst.chargers) == 3

    save_instance(inst, tmp_path)
    loaded = load_instance(tmp_path / f"{inst.instance_id}.json")
    assert loaded.vehicles is not None and len(loaded.vehicles) == len(inst.vehicles)
    assert all(isinstance(v, VehicleSpec) for v in loaded.vehicles)
    assert isinstance(loaded.breaks, BreakRegime)
    assert loaded.breaks.name == "EU561"
    assert loaded.breaks.rest_node_ids == inst.breaks.rest_node_ids
    assert len(loaded.chargers) == 3
    assert all(isinstance(c, Charger) for c in loaded.chargers)


def test_v3_backcompat_missing_fields(tmp_path):
    """An instance saved without v3 fields still loads cleanly."""
    inst = generate(N=10, seed=1)
    assert inst.vehicles is None
    assert inst.breaks is None
    assert inst.chargers is None

    save_instance(inst, tmp_path)
    loaded = load_instance(tmp_path / f"{inst.instance_id}.json")
    assert loaded.vehicles is None
    assert loaded.breaks is None
    assert loaded.chargers is None


def test_hetero_capacity_varies():
    inst = generate(N=40, seed=2, hetero_fleet=True, num_chargers=5)
    caps = [v.capacity for v in inst.vehicles]
    classes = {v.vclass for v in inst.vehicles}
    # Multiple distinct capacities AND at least 2 vehicle classes
    assert len(set(caps)) >= 2
    assert len(classes) >= 2
    # EV present when chargers were requested
    assert any(v.vclass == "ev" for v in inst.vehicles)


def test_breaks_rest_areas_in_customer_range():
    inst = generate(N=50, seed=3, breaks_regime="FMCSA")
    assert inst.breaks.name == "FMCSA"
    for nid in inst.breaks.rest_node_ids:
        assert 1 <= nid <= 50

    # Force-load + save again to double-check serialization tuple→list→tuple
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        save_instance(inst, Path(td))
        again = load_instance(Path(td) / f"{inst.instance_id}.json")
    assert again.breaks.rest_node_ids == inst.breaks.rest_node_ids


def test_evaluator_ignores_v3_fields():
    """Old solvers must keep working — evaluate() doesn't crash on v3 instances."""
    from svrptw.config import Settings
    from svrptw.solvers.classical import greedy as greedy_mod
    inst = generate(N=20, seed=4, hetero_fleet=True, breaks_regime="EU561", num_chargers=2)
    sol = greedy_mod.solve(inst, Settings())
    assert np.isfinite(sol.metrics["operational_cost"])
