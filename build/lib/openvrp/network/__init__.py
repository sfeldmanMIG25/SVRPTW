"""Network ingest + geometry reconstruction (SPEC-OPENVRP-06).

This subpackage lives behind the ``[network]`` extra. The core install
never imports it. Calling ``Problem.from_network(...)`` and then
``solve(problem)`` triggers ``ingest.load_network`` and
``geometry.build_od_with_predecessors``; the resulting predecessor
matrix is reused by ``geometry.reconstruct_route`` for the leg-by-leg
along-network polyline (D14).
"""
from __future__ import annotations

from openvrp.errors import MissingExtra


def _require_network() -> None:
    """Raise MissingExtra if [network] extras aren't installed."""
    try:
        import networkx        # noqa: F401
        import scipy.sparse    # noqa: F401
    except ImportError as e:
        raise MissingExtra(
            "network",
            reason=f"networkx + scipy needed: {e}",
        ) from e


__all__ = ["_require_network"]
