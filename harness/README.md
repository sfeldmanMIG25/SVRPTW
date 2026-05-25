# OpenVRP testing harness

**This directory is the testing harness, NOT part of the openvrp
distribution.** It lives outside the package and is excluded from
the wheel.

```
harness/
├── README.md              ← this file
├── bench/                 ← perf + memory comparisons across solvers
│   ├── all_solvers.py     ← canonical 5-solver shootout
│   ├── orchestrator.py    ← parallel run manager + explicit timings
│   └── outputs/           ← JSON results (auto-generated)
├── visual/                ← solution renderers (SVG, no GUI deps)
│   ├── render_svg.py      ← per-route colored, suitable for human/VLM review
│   └── compare_grid.py    ← side-by-side grid for solver comparison
├── judge/                 ← optional VLM/LLM review of pairwise comparisons
│   └── pairwise.py        ← stub; wires to a future VLM judge
└── reports/               ← generated HTML reports for the dashboard
    └── (auto-populated)
```

## Why separate

`pyproject.toml` excludes `harness/`, `tests*`, `examples*`,
`bench*`, `WorldsFinestVRP/` from the wheel. The harness depends on
optional things (psutil, matplotlib-ish features, etc) the published
package must never carry.

## How to run

```bash
# Full shootout (5 solvers × 3 N values, with peak-RSS sampling):
PYTHONPATH=. python harness/bench/orchestrator.py --n 50 100 250 --budget 30

# Visualize a single solve:
PYTHONPATH=. python harness/visual/render_svg.py --n 100 --solver openvrp_native \
  --out harness/reports/openvrp_native_n100.svg

# Side-by-side grid of all solvers at one N:
PYTHONPATH=. python harness/visual/compare_grid.py --n 100 \
  --out harness/reports/compare_n100.html
```

## What the harness verifies

1. **`bench/`** — wall, peak RSS, objective (shared evaluator), feasibility, K, crossings, load balance per solver per N.
2. **`visual/`** — renders each solver's solution as SVG so a human
   reviewer (or a VLM downstream) can see the geometric quality.
3. **`judge/`** — optional layer that asks a VLM to pick the better of
   two visualized solutions. Strictly testing infrastructure — the
   *package* never invokes a VLM (SPEC-OPENVRP-00 D9).
