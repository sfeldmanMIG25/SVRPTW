# OpenRouter committee smoke — pipeline validated

Date: 2026-05-13
Spec: SPEC-6-LOGIC-02

## Pipeline validated end-to-end

- HTTP client with bearer auth ✓
- JSON-schema-constrained structured output ✓ (the model returns
  parseable `{score, rationale, confidence}` JSON)
- Sanitisation hook (Tier S only) ✓
- Tier-weighted median aggregation ✓
- Dead-model drop on 404 + rate-limit graceful degrade ✓
- `authoritative` flag fires correctly on n<6 ✓

## What broke + got fixed

1. **Wrong model slugs.** SPEC-6-LOGIC-02 listed ~12 free-vision
   models pulled from a research-style writeup; only 4 are
   actually free-vision-capable on OpenRouter live (verified via
   `/api/v1/models` 2026-05-13):
   - `google/gemma-4-31b-it:free`
   - `google/gemma-4-26b-a4b-it:free`
   - `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`
   - `nvidia/nemotron-nano-12b-v2-vl:free`
2. **`max_price=0` was invalid payload.** OpenRouter expects
   `max_price` as an object (per-token caps), not a scalar — 400
   error. Removed; the `:free` suffix is the actual paywall guard.

## One real judgment

On portfolio@10 / OSM-Manhattan-N050-I000:

| field | value |
|---|---|
| consensus_score | 0.50 |
| n_responders | 1 of 4 |
| authoritative | False |
| latency | 4.5 s |
| rationale | "The route map generally makes sense, but there is a noticeable zig-zagging route (purple) that could be optimized to reduce travel distance. The map covers all stops, and the routes are roughly simila…" |

`nvidia/nemotron-nano-12b-v2-vl:free` was the only responder; the
other 3 Tier-A models returned 429 (rate-limited on the free-tier
budget). The rationale flagged a *real* visual feature of the
solution — the zig-zag on one route — which is encouraging
evidence the judge isn't hallucinating.

## What this means for SPEC-6-LOGIC-01 distillation

- The teacher path works. **Capacity is the bottleneck**: 4 models
  × ~2 RPM effective ≈ 8 RPM total. Labelling 2 500 preference
  pairs at 8 RPM = ~5 hours wall-clock. Plus Gemini Flash Lite's
  500 RPD adds ~1 more day. Doable but not parallelisable beyond
  this without paying.
- The `authoritative` flag needs recalibration. With only 4 models
  available, the threshold of n>=6 is unreachable. Lower the
  default to 3 OR add Gemini Flash Lite as a 5th voter so that
  ensembled-score becomes reachable.
- Lifecycle is the spec's chronic risk. OpenRouter's free-tier
  catalog churns weekly; the committee config should be
  auto-queried from `/api/v1/models` on startup rather than
  hard-coded.

## Next

1. ~~Lower `authoritative_min_responders` default from 6 to 3~~ ✓ landed.
2. Add `_refresh_models_from_catalog()` that auto-queries free
   vision models at boot.
3. ~~Wire Gemini Flash Lite as a 5th voter~~ ✓ landed via `include_gemini=True`
   in `OpenRouterCommittee.__init__`; routed through the existing
   `svrptw.vivrp.assessor._GeminiBackend` so the same key + model
   config flow works.

## Updated smoke (with Gemini)

| model | score | latency |
|---|---:|---:|
| nvidia/nemotron-nano-12b-v2-vl:free | 0.50 | 5.4 s |
| nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free | 0.50 | 12.0 s |
| gemini-direct:gemini-3.1-flash-lite | **0.00** | 5.1 s |

3 responders, consensus 0.50, **score_std = 0.236 → authoritative=False**.
The disagreement is the win: Gemini is much harsher than the
Nemotrons. The committee correctly flags this as untrustworthy
rather than averaging to a misleadingly "confident" 0.33.

This is *exactly* the calibrated-uncertainty signal SPEC-6-LOGIC-01
asks for. The student-side ensemble σ will compose: pairs labelled
on `authoritative=False` data go into the dataset but never the
training loss, preventing the student from picking up cross-model
biases as if they were ground truth.
