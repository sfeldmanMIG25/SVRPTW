# SPEC-OPENVRP-07 — Packaging & Dependency Layering

**Status:** FROZEN
**Owner role:** Orchestrator + Bench Engineer
**Depends on:** SPEC-OPENVRP-00 (D1, D2, D12, D13)
**Artifacts:** `pyproject.toml`, `openvrp/__init__.py`, CI config

The dependency tree is part of the product: an open-source dev can `pip install
openvrp`, solve a constraint-rich problem from their own OD, and never pull a heavy
native dependency.

---

## 1. Dependency layers

| Layer | Extra | Hard deps | Enables |
|---|---|---|---|
| core | (none) | numpy, scipy, pydantic≥2 | OD-only solving, full constraint catalog, full driver-hour rulesets, objective control, JSON I/O |
| network | `[network]` | osmnx, networkx, shapely, pyproj | `from_network`, geometry reconstruction, GeoJSON with real lines |
| accel | `[pyvrp]` | pyvrp | preferred construction when present |
| viz | `[viz]` | matplotlib, contextily | optional rendering helpers (never used by `solve`) |
| all | `[all]` | union | convenience |
| dev | `[dev]` | pytest, mypy, ruff, fiona, jsonschema, pip-licenses | tests, lint, GeoJSON validation, license audit |

Rules:
- `import openvrp` and an OD-only `solve` work on **core only**. A CI job in a
  core-deps-only environment runs the OD-only conformance subset and **fails if
  osmnx/pyvrp/matplotlib is importable on that path**.
- `construction="pyvrp"` without `[pyvrp]` ⇒ `MissingExtra("pyvrp", install="pip
  install openvrp[pyvrp]")` — never a raw ImportError.
- `from_network` / `to_geojson` without `[network]` ⇒ `MissingExtra("network", ...)`
  at call time.
- `auto` construction degrades silently pyvrp→fast when `[pyvrp]` is absent and
  records `construction_used="fast"`; the PyVRP-free path must independently meet the
  performance mandate (SPEC-OPENVRP-00 §1 / SPEC-OPENVRP-03 §4).

## 2. License (SPEC-OPENVRP-00 D2)

MIT. Every core and `[network]` dependency must be permissive-compatible (numpy/scipy
BSD, pydantic/osmnx/networkx MIT/BSD, shapely/pyproj BSD — all fine; PyVRP MIT, fine
as an optional extra). A CI license-audit job fails on any copyleft/unknown
dependency in core or `[network]`. The vendored research core is released under MIT
(ADR recorded) before first publication.

## 3. Versioning & API stability

SemVer. The public surface is exactly what `openvrp/__init__.py` re-exports (schema
types + `solve`/`solve_od`); everything else is private and may change without a major
bump. New optional schema fields with safe defaults are minor bumps; changing an
existing field's meaning/type is a major bump. Every `Solution` carries
`solver_version` and `problem_fingerprint` for audit and cache keying.

## 4. Layout

```
openvrp/
  __init__.py          # public re-exports ONLY
  api.py               # solve, solve_od
  schema/{input,output}.py     # mypy --strict
  engine/{constraints,fleet,objective,_adapter}.py
  engine/_vendor/      # vendored research search core
  network/{ingest,geometry}.py # [network]
  io/serialize.py
  errors.py
  diagnostics.py
tests/
  unit/                # incl. ported research guard suites
  conformance/         # SPEC-OPENVRP-04 §7 suite
  core_only/           # runs under core-deps-only env (gate)
docs/                  # SPEC-OPENVRP-09
pyproject.toml
```

## 5. Acceptance criteria

- H1: clean env `pip install .` then an OD-only constraint-rich solve works;
  dependency tree shows no osmnx/pyvrp/matplotlib.
- H2: `pip install .[network]` enables `from_network` + geometry + GeoJSON.
- H3: `pip install .[pyvrp]` ⇒ `auto` selects pyvrp; absent ⇒ selects fast and
  `construction_used` reflects it; the fast path still meets the performance mandate.
- H4: core-only CI job passes the OD-only conformance subset and fails if a heavy dep
  is importable on that path.
- H5: license-audit job green; a deliberately added copyleft dep fails it.
- H6: `mypy --strict openvrp/schema` clean; `mypy openvrp` clean; `ruff` clean.
- H7: built sdist+wheel installed fresh reproduces H1–H3.

## 6. Non-goals
- No conda recipe (PyPI first).
- No pinned-exact runtime deps (compatible ranges; lockfile only in `[dev]`).
