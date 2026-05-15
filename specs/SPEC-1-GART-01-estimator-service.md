# SPEC-1-GART-01 — Tour-length estimator service

```
ID:            SPEC-1-GART-01
Title:         Stable, fast tour-length estimator API wrapping GART v4
Owner role:    ML Engineer
Status:        FROZEN
Inputs:        Frozen LightGBM artifact at gart_artifact_path
Outputs:       svrptw.models.gart.TourLengthEstimator
```

## Behavior

Wrap the frozen `lgbm_alpha_model_v4.joblib` (from `D:/VRP-Advanced-Estimator-Integration/estimators/`) behind a stable Python interface used by every GART-consuming solver.

```python
class TourLengthEstimator(Protocol):
    def estimate(self, nodes: np.ndarray, dist_matrix: np.ndarray | None = None) -> float:
        """Expected closed TSP tour length (depot -> nodes -> depot).
           nodes: (k, d) feature array OR (k,) array of node indices into dist_matrix."""
    def estimate_batch(self, batch: list[np.ndarray], dist_matrix: np.ndarray | None = None) -> np.ndarray: ...
    def estimate_marginal(self, current: np.ndarray, candidate: np.ndarray | int,
                          dist_matrix: np.ndarray | None = None) -> float:
        """Marginal tour-length increase from adding `candidate` to `current`."""
```

The GART v4 estimator predicts an `alpha` parameter; tour length is reconstructed via the GART formula `L ≈ alpha · sqrt(n · A)` (Beardwood-Halton-Hammersley generalization). The feature pipeline lives in `D:/VRP-Advanced-Estimator-Integration/estimators/feature_creator_v3.py`; we **import** it (do not copy).

### Asymmetric extension

Base GART was trained on Euclidean (symmetric). For SVRPTW v1 instances the travel-time matrix is asymmetric. The asymmetric estimator emits:

  `L_asym ≈ alpha · sqrt(n · A) · (1 + beta · asymmetry_score)`

`beta` is fit by least squares against LKH-3 ATSP ground truth on a 5,000-node-set calibration set in `models/gart/calibration_asym.npz`. This calibration is part of Phase 1 setup, not Phase 3 stochastic training.

### Marginal computation

```python
def estimate_marginal(self, current, candidate, dist_matrix):
    if dist_matrix is not None:
        # Cheap: nearest-insertion cost into current's TSP cycle.
        return _nearest_insertion_delta(current, candidate, dist_matrix)
    return self.estimate(np.concatenate([current, [candidate]])) - self.estimate(current)
```

A nearest-insertion approximation is preferred when a distance matrix is available — it is O(k) and tracks LKH-3 marginals to within ~3%.

## Invariants

- `estimate(np.array([]))` returns 0.
- `estimate({single}, dist_matrix)` equals `2 · dist_matrix[depot, single]`.
- `estimate` is non-decreasing in node count under a fixed matrix (random property test, 100 trials).
- Cache (SPEC-1-GART-03) never serves results from a previous model load.

## Acceptance

- MAPE on the symmetric calibration set ≤ MAPE-reported-by-v4-paper + 0.5 pp.
- MAPE on the asymmetric calibration set (LKH-3 ground truth) ≤ 1.5 × symmetric MAPE.
- `estimate_batch` on 1000 sets of 20 nodes completes in < 50 ms on CPU.
- Invariant assertions in `tests/unit/test_gart_invariants.py` (100 random property tests).
- `python -m svrptw.models.gart estimate --instance instances/v1/OSM-Austin-N100-I003.json --customers 0,3,7,15` exits 0 with a numeric estimate.

## Non-goals

- Retraining the LightGBM model.
- Stochastic / distributional estimation (deferred per ADR-0001).

## Dependencies

- SPEC-0-CFG-01, SPEC-0-INST-01.
- Read-only access to `D:/VRP-Advanced-Estimator-Integration/`.
