# Iter 5f sub-agent: Unified scoring API + construction-technique trial bench

Date: 2026-05-15
Sub-agent: scoring API + bench builder
Parent task: expose cost-vs-quality tension as a first-class objective and
trial constructions on it.

## What was built

| Path | Purpose | Lines |
|------|---------|-------|
| `D:/SVRPTW/svrptw/scoring/__init__.py` | Re-exports `score`, `compare`, `score_batch`, `UnifiedScore`, `DEFAULT_WEIGHTS` | 25 |
| `D:/SVRPTW/svrptw/scoring/sub_api.py` | The unified-score "pseudo-API" sub-agent module | 290 |
| `D:/SVRPTW/bench/scripts/construction_trials.py` | CLI bench head-to-heading 7 constructions (x2 with bandit) on a stratified set | 285 |
| `D:/SVRPTW/tests/unit/test_metrics_smoke.py` | Smoke tests for `svrptw.metrics.score_solution` | 70 |
| `D:/SVRPTW/tests/unit/test_unified_score_smoke.py` | Smoke tests for `svrptw.scoring.sub_api` | 75 |


## Exit-criteria checklist

1. `python -c "from svrptw.scoring.sub_api import score, compare, UnifiedScore; print('OK')"`
   PASS — prints `OK {'operational_cost': 0.4, 'quality_index': 0.4, 'visual_score': 0.2}`.
2. `pytest tests/unit/test_metrics_smoke.py tests/unit/test_unified_score_smoke.py -q`
   PASS — 9/9 tests.
3. `pytest tests/unit -q` no regressions
   PASS — 137 passed, 1 skipped (pre-existing skip on `tests/unit/test_*` unaffected).
4. `python bench/scripts/construction_trials.py --help`
   PASS — argparse usage prints; PYTHONPATH note in module docstring.
5. End-to-end on Manhattan N=50 + 3 constructions, no bandit
   PASS — wrote `bench/runs/construction_trials_smoke.json` (3 rows).

## Smoke leaderboard (Manhattan N=50, I003)

```
rank construction         n   unified   cost($)     quality
1    pyvrp_4s             1   0.5419    801.29      0.7242
2    fast_construct_2s    1   0.4050    797.94      0.4478
3    greedy               1   0.3921    834.22      0.4509
```

Interpretation: PyVRP wins on the unified objective despite NOT being the
cheapest. fast_construct beats it on raw cost ($797.94 vs $801.29) but is
heavily penalised on `quality_index` (0.4478 vs 0.7242). Crossings tell
the same story: PyVRP=58, fast_construct=243, greedy=271. This matches
the Iter 5f motivating signal (PyVRP=0.724 vs solve_auto=0.538) the user
shared, confirming the metrics suite is detecting the same structural
gap from a different solver pair.


## API surface

```python
from svrptw.scoring.sub_api import score, compare, score_batch, UnifiedScore

# single-solution score, no VLM
us = score(inst, sol)                       # UnifiedScore(unified=..., breakdown=...)
us = score(inst, sol, weights={"quality_index": 1.0,
                                "operational_cost": 0.0})  # weight isolation

# pair compare with optional pair-image VLM
res = compare(inst, sol_a, sol_b,
              do_pair_visual=True,         # renders side-by-side + asks Qwen-VL
              judge_model="qwen/qwen3-vl-4b",
              judge_kind="lmstudio")
# res = {"a": UnifiedScore, "b": UnifiedScore,
#        "winner": "a" | "b" | "tie", "margin": float,
#        "dissent_flags": ["cost_prefers_a_but_quality_prefers_b", ...],
#        "pair_visual_score": float | None,
#        "cost_winner": ..., "quality_winner": ..., "visual_winner": ...}

# batch
scores = score_batch(paths, solver_fn=lambda inst: pv.solve(inst, settings))
```

The cost component is normalised against either a caller-supplied
`cost_max` or a deterministic per-instance bound `N * miss_penalty * 1.5`.
The bench derives a per-instance shared `cost_max = 1.5 * worst_finite`
across constructions on that instance, so all solvers on the same
instance are normalised against the same scale.

When `visual_image_path is None`, the `visual_score` weight is dropped
and the cost+quality weights are renormalised to sum to 1 — verified by
`test_visual_none_no_crash` and the leaderboard above (default weights
0.40/0.40 collapse to 0.50/0.50, as `test_weight_isolation_quality_only`
implicitly confirms).

## Caveats

- **No real VLM call in the smoke run.** The end-to-end command does NOT
  pass `visual_image_path`, so `visual_score` stays None. Tests rely on
  this branch — they explicitly DO NOT mock the LM Studio HTTP client
  because the `visual_image_path=None` branch is the documented
  no-network path.
- **PyVRP construction time clamp.** `pyvrp_4s` is implemented via
  `pv.solve(inst, settings, budget_seconds=4.0)`. Inside the warm
  variant, PyVRP construction is clamped to `0.30 * total_budget`, but
  the standalone `pv.solve` honours the full budget. Documented in the
  catalog comments.
- **Pair visual cost.** `compare(do_pair_visual=True)` writes a temp
  directory with two PNGs per pair and currently does NOT delete it. The
  pair-VLM path adds ~2-5s per pair on the local Qwen-4B. Use sparingly
  in batch contexts.
- **PYTHONPATH requirement.** The bench depends on `import svrptw` from
  the repo root (no editable install). The module docstring documents
  this. Tested with `$env:PYTHONPATH = "D:/SVRPTW"` from PowerShell.
- **No new metrics added.** Per the brief, `svrptw/metrics/quality.py`
  was not modified — `quality_index` is consumed as-is.


## Files (absolute paths)

- `D:/SVRPTW/svrptw/scoring/__init__.py`
- `D:/SVRPTW/svrptw/scoring/sub_api.py`
- `D:/SVRPTW/bench/scripts/construction_trials.py`
- `D:/SVRPTW/tests/unit/test_metrics_smoke.py`
- `D:/SVRPTW/tests/unit/test_unified_score_smoke.py`
- `D:/SVRPTW/bench/runs/construction_trials_smoke.json` (smoke output)

## Next steps the parent might want

1. Run the full default sweep
   `python bench/scripts/construction_trials.py` (8 instances x 14 arms;
   est. ~10-15 min on the laptop, dominated by 30s bandit per arm).
2. Wire `compare(do_pair_visual=True)` into the council loop so the
   live VLM judge folds into per-iteration arm scoring.
3. Add a `visual_image_path` arg to the bench (optional flag) so the
   Qwen-VL is called per row when LM Studio is up, and the leaderboard
   shows a real `visual_score` column instead of None.
4. Snapshot every per-row PNG to webui via `push_snapshot` so the live
   gallery shows the per-construction visual quality at a glance.
