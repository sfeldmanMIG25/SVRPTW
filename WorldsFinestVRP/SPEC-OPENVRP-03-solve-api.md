# SPEC-OPENVRP-03 — `solve()` API & Execution Model

**Status:** FROZEN
**Owner role:** Spec Author
**Depends on:** SPEC-OPENVRP-00, 01, 02, 06
**Module:** `openvrp/api.py` (re-exported at `openvrp.solve`)

---

## 1. Public surface

```python
from openvrp import solve, solve_od, Problem, SolveOptions, Solution

sol = solve(problem: Problem, options: SolveOptions = SolveOptions()) -> Solution
sol = solve_od(time_matrix, stops, depots, fleet, *,
                options=SolveOptions(), distance_matrix=None) -> Solution
```
`solve` is the whole callable contract. `solve_od` is sugar that builds an
`ODMatrix`/`Problem(mode="od")` and delegates.

---

## 2. Execution model (SPEC-OPENVRP-00 D10)

Synchronous and blocking, but cooperatively cancellable and progress-reporting:
- `SolveOptions.deadline` (absolute `time.monotonic()`) and `budget_seconds` — the
  sooner bound wins; checked every search-iteration boundary; returns best feasible
  found (`status="feasible"`, or `"partial"` if none yet).
- `on_progress(ProgressEvent)` where `ProgressEvent = {fraction, best_objective|None,
  vehicles_used|None, vehicles_minimum_found|None, elapsed_seconds,
  phase: ingest|construct|refine|finalize}` — at least once per phase, ≤~1/s in
  refine. A callback that raises the provided `StopSolve` sentinel ends the search
  cleanly with best-so-far; any other exception surfaces as `SolveAborted`.

No threads are exposed to the caller; internal parallelism is governed only by
`SolveOptions.threads`.

---

## 3. Dispatch

```
ingest    : network mode -> SPEC-OPENVRP-06 (snap + sparse shortest paths,
            predecessors retained for geometry); od mode -> use problem.od.
            Triangle-inequality precondition (D16) assumed; gross violations
            -> diagnostics warning, not a failure.
construct : construction == "pyvrp"  -> requires [pyvrp] extra or MissingExtra
            construction == "fast"   -> always available, no extra
            construction == "auto"   -> pyvrp if importable else fast
            build an initial hard-feasible solution.
refine    : adaptive operator search over the CONDITIONALLY REGISTERED operator
            pool (SPEC-OPENVRP-04 §5 — non-optional); objective is the caller's
            ObjectiveConfig including active fleet minimization (D7) and any
            opted-in quality terms (D8).
finalize  : reconstruct per-leg geometry (network path); compute the economics
            decomposition and the full QualityReport (all catalog metrics, whether
            or not weighted); minimize reported fleet; assemble Solution; verify
            invariants; fingerprint; diagnostics.
```
Conditional operator registration and the fleet-minimization pass are part of the
contract, not optimizations. Omitting either is a spec violation: the first silently
regresses solution magnitude (established in the research record), the second breaks
SPEC-OPENVRP-00 D7.

---

## 4. Performance mandate (SPEC-OPENVRP-00 §1, gate #6)

The shipped `solve` MUST, at matched wall-clock on the OSM-asymmetric evaluation
regime:
- be **strictly faster** than research `solve_auto` (lower wall to reach an
  equal-or-better objective), AND
- produce solutions of **equal or lower** objective cost, AND
- retain the established advantage over PyVRP, OR-Tools, and LKH-3.

This is a standing acceptance gate, re-measured by the benchmark harness whenever the
search path changes. A change that regresses it does not ship. The default
(PyVRP-free `fast` construction) must independently satisfy the mandate so the
core-only install is genuinely competitive (SPEC-OPENVRP-00 D13).

---

## 5. Determinism (SPEC-OPENVRP-00 D11)

Identical `Problem` + `seed` ⇒ identical `Route.visits` stop sequences across runs,
across `threads=1` vs `N`, and across repeated `fast` constructions; timings are a
pure function of sequence + input. `pyvrp` construction inherits PyVRP's
seed-determinism at the pinned version (documented).

---

## 6. Acceptance criteria

- C1: OD-only solve returns a valid `Solution` (SPEC-OPENVRP-02 B1).
- C2: network solve returns geometry-bearing `Solution` (B2).
- C3: `construction="pyvrp"` without the extra ⇒ `MissingExtra("pyvrp", install=...)`,
  never a raw ImportError.
- C4: a 1-second deadline returns within ~1 s + finalize; never hangs.
- C5: `on_progress` fires per phase and ≥1/s in refine; `StopSolve` yields clean
  best-so-far.
- C6: determinism gate passes.
- C7: **performance-mandate gate passes** — harness shows shipped `solve` strictly
  faster and not worse on cost than `solve_auto`, advantage over the three open
  solvers retained, on the OSM regime, including the PyVRP-free path.
- C8: a well-formed but hard-infeasible problem returns `status="infeasible"` with
  diagnostics (SPEC-OPENVRP-08), never raises.
- C9: the fleet-minimization pass runs in finalize; `vehicles_minimum_found` is
  populated and respected per SPEC-OPENVRP-02 invariants.

## 7. Non-goals
- No async/await API (cancellation covers the real need).
- No batch/multi-problem API (callers loop; the pattern is documented).
