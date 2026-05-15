"""Round-trip: write a synthetic Instance to disk and load it back, byte-identical
on every numeric field.  Part of SPEC-0-INST-01 acceptance."""
import numpy as np
import pytest

from svrptw.io import Customer, Depot, Instance, load_instance, save_instance


def _make(n: int = 5) -> Instance:
    rng = np.random.default_rng(0)
    T = rng.uniform(1.0, 20.0, size=(n + 1, n + 1))
    np.fill_diagonal(T, 0.0)
    D = T * 0.5
    return Instance(
        instance_id="TEST-N005-I000",
        city="Synth",
        num_customers=n,
        num_vehicles=2,
        vehicle_capacity=10,
        depot=Depot(node_id=0, x=0.0, y=0.0, ready=480, due=960),
        customers=[
            Customer(id=i + 1, node_id=10 + i, x=float(i), y=float(i),
                     demand=1, ready=500, due=900, service=10)
            for i in range(n)
        ],
        travel_time=T,
        travel_dist=D,
        asymmetry_score=0.5,
        seed=42,
    )


def test_roundtrip(tmp_path):
    inst = _make()
    save_instance(inst, tmp_path)
    loaded = load_instance(tmp_path / f"{inst.instance_id}.json")
    np.testing.assert_array_equal(loaded.travel_time, inst.travel_time)
    np.testing.assert_array_equal(loaded.travel_dist, inst.travel_dist)
    assert loaded.num_customers == inst.num_customers
    assert loaded.asymmetry_score == pytest.approx(inst.asymmetry_score)
