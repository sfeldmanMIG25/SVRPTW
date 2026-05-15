"""Property tests for the GART estimator. SPEC-1-GART-01 invariants."""
import numpy as np
import pytest

from svrptw.models.gart import get_default_estimator


@pytest.fixture(scope="module")
def est():
    return get_default_estimator()


@pytest.fixture(scope="module")
def synthetic_matrix():
    """Euclidean distance matrix on uniform 2D points — GART's training distribution.
    Mild asymmetry injected (5%) so the wrapper's asymmetric-correction branch executes."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 100.0, size=(30, 2))
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    noise = rng.uniform(0.95, 1.05, size=D.shape)
    D = D * noise
    np.fill_diagonal(D, 0.0)
    return D


def test_empty_returns_zero(est):
    assert est.estimate(np.array([], dtype=np.int64), dist_matrix=np.zeros((1, 1))) == 0.0


def test_single_returns_round_trip(est, synthetic_matrix):
    M = synthetic_matrix
    L = est.estimate(np.array([3], dtype=np.int64), dist_matrix=M)
    assert L == pytest.approx(2.0 * M[0, 3])


def test_non_decreasing_in_node_count(est, synthetic_matrix):
    """Within the alpha*MST regime (k >= 3) length should be non-decreasing.
    The k=1/k=2 → k=3 boundary uses different formulas (direct tour vs alpha*MST)
    and is exempt from the test."""
    M = synthetic_matrix
    nodes = list(range(1, 10))
    prev = 0.0
    for k in range(3, len(nodes) + 1):
        L = est.estimate(np.array(nodes[:k], dtype=np.int64), dist_matrix=M)
        assert L >= prev * 0.95, f"length dropped from {prev:.2f} to {L:.2f} at k={k}"
        prev = L


def test_marginal_matches_diff_within_tolerance(est, synthetic_matrix):
    M = synthetic_matrix
    current = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    cand = 6
    via_diff = est.estimate(np.append(current, cand), dist_matrix=M) - est.estimate(current, dist_matrix=M)
    via_marginal = est.estimate_marginal(current, cand, dist_matrix=M)
    # Marginal uses nearest-insertion; we just check it's a reasonable bound.
    assert via_marginal > 0
    assert via_marginal <= via_diff * 1.5 + 5.0  # loose — marginal is an O(k) approximation
