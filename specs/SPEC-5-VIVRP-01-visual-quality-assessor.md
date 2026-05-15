# SPEC-5-VIVRP-01 — ViVRP visual quality assessor

```
ID:            SPEC-5-VIVRP-01
Title:         Vision-language solution quality assessor (ViVRP)
Owner role:    ML Engineer + Bench Engineer
Status:        FROZEN
Inputs:        Instance, Solution, PNG renderer for routes
Outputs:       svrptw.vivrp.assess(instance, solution) -> ViVRPReport
```

## Why

Operational cost is one signal. Human-interpretability — does the plan *look*
sensible, do clusters make geographic sense, are routes legible to a dispatcher
— is orthogonal and matters for adoption. ViVRP captures it as a single
quantitative score so it can be plotted on the same Pareto frontier as cost.

This replaces the originally-proposed ViTSP visual review (a single-route TSP
critic) with a VRP-native critic that judges multi-route partitions.

## Behavior

```python
@dataclass
class ViVRPReport:
    overall_score: int                  # 1 - 10
    clustering_score: int               # how spatially coherent are vehicle clusters
    interpretability_score: int         # could a dispatcher follow this at a glance
    geometry_score: int                 # are routes free of large crossings / detours
    notes: str                          # 2-4 sentence explanation, no fluff
    zoom_findings: list[ZoomFinding]    # 0..N regions the model zoomed into
    rubric_version: str
    model: str                          # "qwen2-vl-2b" | "gemini-1.5-flash" | ...
    latency_seconds: float
```

```python
@dataclass
class ZoomFinding:
    bbox: tuple[float, float, float, float]   # (xmin, ymin, xmax, ymax) in image coords
    finding: str                              # 1-2 sentence specific observation
    severity: Literal["info", "minor", "major"]
```

### Pipeline

1. **Render.** `svrptw.viz.render_solution(inst, sol, out_path, dpi=180)` draws
   the depot, customers (colored by served-by-vehicle), and route polylines.
2. **First pass.** VLM sees the full image plus a textual summary
   `{N, K, asym_score, missed, num_vehicles_used}` and emits an initial
   `ViVRPReport` with `zoom_findings = []`.
3. **Optional zoom.** If `overall_score < 7` or the first pass marks
   `clustering_score < 5`, the model selects 1–3 bounding boxes via a
   structured tool call; the renderer crops those regions at 2× and reissues
   the prompt for each. The model returns one `ZoomFinding` per crop.
4. **Aggregate.** The final report contains the first-pass scores plus zoom
   findings.

### Models

Tier 1 — **Local Qwen2-VL-2B-Instruct** via `transformers`. Runs on the
8 GB 3070 Ti. Quantized to 4-bit when not in use elsewhere. This is the
default.

Tier 2 — **Gemini 1.5 Flash** via `google-generativeai`. Behind
`SVRPTW_GEMINI_API_KEY` env var. Used when (a) local model unavailable,
(b) per-spec override `force_gemini=True`, or (c) local confidence flags
report `"unsure"`.

### Prompting

The model is given a fixed rubric prompt with the score definitions and
asked to emit a JSON object. Structured-output parsing uses `instructor`
or a JSON-schema constrained decoder when the backend supports it; the
fallback is `json.loads` with a one-retry repair.

Rubric (full text in `svrptw/vivrp/rubric.md`):

- 9–10: every vehicle's customers form a tight cluster, routes are nearly
  straight depot-out-back arcs, no obvious crossings.
- 7–8: clusters mostly coherent, minor detours acceptable.
- 5–6: one or two routes wander; some customers clearly mis-assigned.
- 3–4: significant interleaving between vehicles, several detours.
- 1–2: routes are visually random; a dispatcher would reject this.

## Invariants

- The score is deterministic per (image bytes, model, seed). Repeated calls
  with the same inputs return byte-identical scores.
- The model never sees the operational cost — it scores on visual evidence
  only so its signal is independent of the numerical evaluation.
- Latency: full pipeline ≤ 6 s on the 3070 Ti for an N ≤ 200 instance using
  Qwen2-VL-2B.

## Acceptance

- `python -m svrptw.vivrp assess --instance instances/v1/OSM-Austin-N100-I000.json
  --solution solutions/ortools/OSM-Austin-N100-I000.json` prints a valid
  `ViVRPReport` JSON and exits 0.
- A held-out 20-solution calibration set (10 obviously-good, 10 obviously-bad
  hand-crafted) yields Spearman ρ ≥ 0.7 between ViVRP score and human ranking.
- Test asserts: structured-output schema validates; failed parse triggers
  exactly one repair retry.

## Non-goals

- Real-time interactivity. Batch only.
- Replacing the operational-cost evaluator. ViVRP complements; it does not
  rank solvers by itself.
- Multi-image attention / video. Single static image per assessment.

## Dependencies

- `transformers >= 4.45`, `accelerate`, `Pillow`, `matplotlib`.
- Optional: `google-generativeai` for Gemini fallback.
- SPEC-0-CFG-01 (Settings carries the model choice).
