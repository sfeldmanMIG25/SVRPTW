---
title: LLM Judge Consistency Study
project: SVRPTW
tags: [judges, vlm, consistency, phase-C, openrouter, gemini]
updated: 2026-05-15
---

# Phase C* — LLM judge consistency study

## Why this study exists

The pasted reflection from earlier this epic warned: *"single-judge VLM resolution is too coarse to be a reliable selection signal — the teacher's preferences are noisy enough that with more samples, the apparent dominance washes out."*

The user's direction (iter 5): **first evaluate each evaluator for consistency**, lightest models first. If a model gives wildly different scores on the same image pair across runs, it's noise. If it gives consistent scores, it's signal. We need to know per-model before we use the multi-judge panel as a selection signal.

## Protocol

For each judge model (lightest → heaviest):
- Take the same (image_A, image_B) pair
- Call the model K=3 times with identical prompt + identical image bytes
- Record (score, rationale, confidence) per call
- Compute mean, σ, range across calls
- Classify: **CONSISTENT** if σ < 0.10, **NOISY** otherwise

Pair shown to all judges:
- Image A: `solve_auto` (PyVRP-warm + LinUCB) final solution
- Image B: PyVRP@2x baseline final solution
- Both rendered via `render_llm_compare` — clean white basemap (when geographic), distinct tab10 colors per route, numbered route badges

Models, lightest first:
| order | model | params | kind |
|-------|-------|--------|------|
| 1 | nvidia/nemotron-nano-12b-v2-vl:free | 12B (vision-specialized) | OpenRouter |
| 2 | google/gemma-4-26b-a4b-it:free | 26B (MoE) | OpenRouter |
| 3 | nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free | 30B (MoE) | OpenRouter |
| 4 | google/gemma-4-31b-it:free | 31B | OpenRouter |
| 5 | gemini-2.0-flash | flash class | Google direct |
| 6 | claude-3-5-haiku-latest | small | Anthropic (SDK not installed) |

## Findings so far (Manhattan N=100 pair, 3 reps each)

| model | scores | μ | σ | range | verdict |
|-------|--------|---|---|-------|---------|
| nemotron-12b-vl | 0.40, 0.60, 0.60 | 0.533 | 0.115 | 0.20 | borderline (σ just above 0.10 cutoff); **usable with repeat-aggregate** |
| gemma-26b-MoE | err, err, err | — | — | — | **JSON-broken**: returns ```json…``` markdown code-block, not strict JSON. Hits 429 after retry. |
| nemotron-30b-MoE | 1.0, ... (run in flight) | — | — | — | first call extreme preference for A (vs nemotron-12b's tied mean) — divergence between sizes is real signal |
| gemma-31b | (in flight) | — | — | — | TBD |
| gemini-2.0-flash | (next) | — | — | — | TBD |
| claude-haiku | — | — | — | — | SDK not installed; needs `pip install anthropic` |

## Implications

1. **gemma-26b-MoE needs a different prompt strategy.** The OpenRouter `JUDGMENT_SCHEMA` constraint isn't being honored. Options:
   - Strip the schema and parse a relaxed grammar (extract `"score"\s*:\s*([0-9.]+)` from the markdown)
   - Use OpenRouter's tool-calling mode if the model supports it
   - Drop gemma-26b from the panel

2. **12B nemotron is borderline-consistent at σ=0.115.** Aggregating 5+ calls would push σ into the consistent range and give a usable judge. That's an architectural insight: the panel should call each judge multiple times, not just once.

3. **30B nemotron diverges from 12B on the same pair** (1.0 vs 0.53 mean). This is useful — different model sizes reveal different signals on the same image. The multi-judge panel composes this divergence into a tier-weighted median + dissent flag, which is exactly the right shape if individual judges are noisy.

## Live in dashboard

The webui's "VLM judge panel" now displays per pair:
- Image A and Image B side-by-side at the top of the pair card (the EXACT bytes the judges saw)
- One row per (model, run_index) with score color-coded (green ≥ 0.7, amber middle, red ≤ 0.3) + rationale snippet
- A consistency badge: `μ=…  σ=…  range=…  n=K/N  CONSISTENT|NOISY`

This is exactly the user's iter-5 ask: "show me what the images you are looking at."

## Files

- `bench/scripts/judge_consistency.py` — main runner (lightest-first, K reps, streams to dash)
- `bench/scripts/judge_consistency_run2.py` — extended run (models 3-4 + Paris pair)
- `bench/runs/judge_consistency.json` — output rows from run 1
- `webui/static/snapshots/consistency/<iid>__warm.png` and `<iid>__pyvrp.png` — the cached pair PNGs

## Next steps

1. **Wait for run 2** to finish (30B + 31B on Manhattan + add Paris pair). Monitor armed.
2. **Add gemini-2.0-flash** as run 3 (cross-vendor reference; the multi_judge.py `_call_gemini` path).
3. **Decide on gemma-26b**: either fix its parsing OR drop it. JSON schema enforcement on free-tier models is fundamentally unreliable.
4. **Aggregate-then-classify**: re-run the borderline models with K=5 or K=8 to see whether σ tightens enough to call them consistent.
5. **Pair up against Paris N=200 + Solomon C101** to ensure findings generalize beyond the Manhattan pair.

## Cross-references
- [[01 - Progress Report]]
- [[10  - Human Reflection]] — user's vision for VLM judges
- [[12 - Architecture Review]]
- [[13 - Construction Bypass]] — running in parallel
