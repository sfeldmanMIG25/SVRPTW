# SPEC-6-LOGIC-02 — Multi-model OpenRouter committee as teacher

```
ID:            SPEC-6-LOGIC-02
Title:         Upgrade the Track-4 teacher from single-Gemini to a
               tier-weighted free-tier OpenRouter committee (~11 VLMs +
               6 text reviewers). Consensus + variance feed the
               student's training labels and uncertainty signal.
Owner role:    ML Engineer
Status:        FROZEN
Inputs:        rendered solution PNGs, dispatcher rubric
Outputs:       svrptw.logic.committee  →  CommitteeLabel (consensus +
               per-model votes + variance) used by SPEC-6-LOGIC-01 student
Depends on:    SPEC-6-LOGIC-01 (teacher cache + student architecture
               survive intact; only the *label source* changes)
```

## Why

A single Gemini-Flash-Lite call is enough signal to make the
ViVRP-Gemini smoke result interesting (portfolio wins cost, LKH-3
wins logic — see `bench/figures/v1_vivrp_gemini_smoke.md`), but it
is not enough to *distill into a student*. One labeller's biases
become the student's biases. Two failure modes are already visible:

1. **Single-model bias** — Gemini may reward fewer routes / cleaner
   geometry even when those don't translate to dispatcher acceptance.
2. **No abstention signal** — when the judge is unsure, there is no
   ensemble σ to read; we infer "low confidence" only from rationale
   tone.

OpenRouter exposes ~11 free vision-capable models from 6 different
model families (Google Gemma, Alibaba Qwen, NVIDIA Nemotron,
Mistral, Meta Llama, Z.ai/Moonshot) at 200 RPD each. Theoretical
ceiling: ~2 200 vision judgments/day pure-free, before touching
Gemini or Groq. That is *more* than enough capacity to label our
2.5 k preference pairs in one weekend.

What the committee gives us that single-Gemini does not:

- **Decorrelated biases** across model families → the median is more
  trustworthy than any single vote.
- **Free σ signal** (variance across committee votes) → directly
  feeds the `authoritative` flag the student ensemble uses in
  SPEC-6-LOGIC-01.
- **Free rate budget** — Gemini's 500 RPD is the limiting reagent
  today; OpenRouter relieves it.

## Behaviour

```python
from svrptw.logic.committee import OpenRouterCommittee

c = OpenRouterCommittee()  # reads SVRPTW_OPENROUTER_API_KEY from .env
label = c.label(instance, solution)

# label.score:            float in [0,1]   (tier-weighted median)
# label.score_std:        float            (across all responding models)
# label.confidence:       float in [0,1]   (= 1 - score_std, clamped)
# label.authoritative:    bool             (n_responders >= 6 and std <= 0.15)
# label.per_model:        dict[str, ModelVote]   (raw votes, retained for audit)
# label.rationale:        str              (winning-tier rationale)
# label.rationale_review: str | None       (text-only reviewer's QA pass)
```

### Tiers (configurable; defaults below)

```python
TIER_S_STEALTH_LOGGED = [        # weight per-vote: 0.5  (logs prompts)
    "openrouter/owl-alpha",
]
TIER_A_PRIMARY_PRIVATE = [        # weight per-vote: 1.0
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "qwen/qwen3-vl-235b-a22b-thinking:free",
]
TIER_B_DIVERSITY = [              # weight per-vote: 0.6
    "qwen/qwen3-vl-30b-a3b-thinking:free",
    "qwen/qwen-2.5-vl-72b-instruct:free",
    "qwen/qwen-2.5-vl-7b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
]
TIER_C_TIEBREAKER = [             # weight per-vote: 0.4  (called only on disagreement)
    "z-ai/glm-4.5-air:free",
    "moonshotai/kimi-vl-a3b-thinking:free",
]
TEXT_RATIONALE_REVIEW = [         # weight 0  (sanity-check rationale, not score)
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "inclusionai/ring-2.6-1t:free",
]
```

### Aggregation

Per pair:
1. Call **Tier A** (5 models) and **Tier B** (7 models) in parallel
   with a 30 s per-model timeout.
2. Drop any model that errors, times out, or returns a non-conforming
   JSON. Log the drop; do not abort.
3. Compute the tier-weighted median over `score`. Tier weights are
   *per vote*: a Tier-A vote counts 1.0, Tier-B 0.6, Tier-S 0.5.
4. Compute σ across the responding scores (no tier weight).
5. If σ > 0.20 *and* fewer than 6 models responded, escalate to
   **Tier C** (2 models). Recompute median + σ.
6. Pick the rationale from the model whose score is closest to the
   weighted median *and* whose tier weight is highest. Tiebreaker:
   shortest rationale (less waffle).
7. Optionally hand the chosen rationale to one **TEXT_RATIONALE_REVIEW**
   model with a "is this rationale consistent with score X?" prompt.
   Stored in `rationale_review`; never affects the score.

### `authoritative` flag

```
authoritative = (n_responders >= 6) and (score_std <= 0.15)
```

This is what SPEC-6-LOGIC-01's ensemble σ pipes into. The student
trains on `authoritative=True` pairs only; `False` pairs are
preserved in the dataset for held-out calibration but never used
for gradient updates.

### Prompt sanitization for Tier S

Cloaked OpenRouter models (`openrouter/*-alpha`) explicitly state
"prompts may be logged for model improvement." Our instance data is
synthetic, but `instance_id` strings could correlate to public OSMnx
city names. Before sending to Tier S only:

- Replace `instance_id` with a `sha1(instance_id)[:12]` hash.
- Strip the city-name field from any rendered solution metadata.
- Round all coordinates to 2 decimal places (preserves topology,
  obscures exact locations).

Sanitization is per-tier in `_TierConfig.sanitize=True/False`.

### Liveness / failure handling

Cloaked models have a lifecycle measured in weeks. The committee
must:

- Catch `HTTP 404 model_not_found` and drop the model from the
  active pool for the rest of the process lifetime.
- Track per-model rolling success rate over the last 100 calls; if
  it falls below 0.5, log a warning and drop the model.
- Never let any one model failing block the rest. The committee
  returns a label as soon as ≥ 3 Tier-A or Tier-B models have
  responded successfully.

### Caching

Same sqlite as SPEC-6-LOGIC-01 (`cache/logic/teacher.sqlite`) but
add a `committee` table with one row per (instance_id, solution_hash,
model) and a materialised view `committee_consensus` that aggregates
votes back into a `CommitteeLabel`. This lets us re-run aggregation
without re-hitting OpenRouter when we tune tier weights.

## Acceptance gates

1. **Liveness**: a synthetic 404 from any single model does not abort
   `label()`; the remaining models still produce a label.
2. **Sanitization**: a unit test asserts that calls to any Tier-S
   model contain the hashed `instance_id`, never the raw form, and
   never the city name field.
3. **Rate-budget**: a 100-pair smoke run completes inside 30 minutes
   wall-clock against the live API. (Allows ~150 calls/min headroom
   across all models; comfortably under 200 RPD per model.)
4. **Agreement w/ Gemini**: on the existing 40-row Gemini-vivrp set,
   the committee's tier-weighted median agrees with Gemini's score
   within ±0.20 on ≥ 70 % of rows. Bigger divergence is a *win* (it
   means we have new signal), not a loss — but the floor protects
   against catastrophic decorrelation.
5. **Student quality lift**: when the SPEC-6-LOGIC-01 student is
   retrained on committee labels rather than Gemini-only labels,
   held-out preference-pair accuracy improves by ≥ 3 %. (If it
   doesn't, the committee is buying capacity but not quality, and
   we should revisit tier weights.)

## Non-goals

- **Not** running the committee in the solver hot loop. The committee
  is the *teacher*, called offline to label pairs; the *student* is
  what the solver calls. Per SPEC-6-LOGIC-01 the student target is
  <50 ms; the committee runs at ~30 s/pair.
- **Not** training a separate distilled model per OpenRouter model.
  One student, one ensemble, trained on consensus labels — that's
  the whole point.
- **Not** paying for any model. `:free` suffix is mandatory in the
  config; the HTTP client sets `max_price=0` in the request headers
  as a belt-and-braces guard.

## Files this spec creates / touches

| Path | Role |
|---|---|
| `svrptw/logic/openrouter_client.py` | thin HTTP wrapper, retry, rate limit, structured-output, max_price=0 guard |
| `svrptw/logic/committee.py` | tier config + parallel orchestrator + consensus aggregator + sanitization |
| `svrptw/logic/teacher.py` (existing) | now backed by `OpenRouterCommittee`; Gemini becomes one optional voter |
| `tests/unit/test_logic_committee.py` | sanitization, liveness fallback, aggregation math |
| `cache/logic/teacher.sqlite` | new `committee` table + `committee_consensus` view |
