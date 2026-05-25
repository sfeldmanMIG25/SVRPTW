# OpenVRP

A pip-installable, dependency-honest Python library that solves rich
Vehicle Routing Problems with Time Windows and heterogeneous fleets.
The caller supplies a problem (an OD matrix, or a network plus points)
and a constraint configuration; OpenVRP returns structured route objects
containing the logical plan (stop sequence, event timeline, cost
decomposition) and, when a network was supplied, the actual
along-network geometry.

**Target user.** An open-source developer or analyst who must handle
real-world routing constraints — non-homogeneous fleets, driver-hour
law, zone access, shift rules, pickup-delivery — without buying a
commercial solver and without building a metaheuristic search engine.

## Documentation map

1. [Quickstart](quickstart.md) — solve an OD problem in 60 seconds
2. [Network tutorial](network.md) — city name → snapped routes → GeoJSON
3. [Constraints guide](constraints.md) — one runnable section per mechanism
4. [Heterogeneous fleet deep-dive](fleet.md) — mixed van + truck + refrigerated
5. [Objective control](objective.md) — quality terms, fleet minimization
6. [Schema reference](schema.md) — auto-generated from docstrings
7. [Output & serialization](serialization.md) — JSON + GeoJSON
8. [Performance & limits](performance.md) — measured, honest numbers
9. [Adding a constraint](contributing.md) — the contributor on-ramp
10. [FAQ / migration](faq.md) — common questions

The product thesis in three sentences: OpenVRP is a free, MIT-licensed
library that handles real-world routing constraints (heterogeneous
fleets, EU 561 / US HOS driver-hour rulesets, embargo zones, EV range,
multi-depot, pickup-delivery) without a commercial solver. It's
network-aware: give it an OSM city name and it returns real
along-network polylines that drop into QGIS/Leaflet via GeoJSON. The
caller controls the objective from the top — operational cost,
fleet size, and any of 8 measurable operational-quality metrics —
with no VLM/visual scoring anywhere.
