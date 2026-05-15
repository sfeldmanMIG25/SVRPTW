# SPEC-6-LOGIC-01 — Distilled dispatcher-acceptance judge

```
ID:            SPEC-6-LOGIC-01
Title:         Gemini-teacher → small-student calibrated logic score with uncertainty
Owner role:    ML Engineer
Status:        FROZEN
Inputs:        svrptw.bench rows (instance + solution JSON), svrptw.viz.renderer PNGs
Outputs:       svrptw.logic.{teacher, student, ensemble, dataset}, logic ∈ [0,1] + σ
Depends on:    SPEC-5-VIVRP-01 (Gemini backend already wired), bench/baselines/v1.json
```

## Why

Academic VRPTW cost is the benchmark on those benchmarks — chasing it
to four decimals does not produce a commercial win. The thing a
dispatcher actually trades on is "would I ship this route sheet?":
coverage of tight windows, no obviously ugly crossings, balanced
load, plausible break placement. We already have a teacher that can
score this from a rendered map (Gemini 3.1 Flash Lite, ~5–10 s/call).
What we lack is a **cheap student** (<50 ms) that can be called every
move during local search and an **uncertainty signal** that prevents
the bandit from chasing the student off a cliff when it is unsure.

Three requirements for the logic axis:
1. **Distill** the teacher into a student small enough to call in the
   solver's inner loop.
2. **Calibrate** the student's scalar so it is interpretable as
   P(dispatcher ships it), not a raw ordinal score.
3. **Carry uncertainty** via a small ensemble; a high-variance prediction
   must downgrade the student's authority in downstream consumers.

## Behaviour

### Teacher (`svrptw.logic.teacher`)

```python
from svrptw.logic.teacher import GeminiTeacher

teacher = GeminiTeacher()  # reads SVRPTW_GEMINI_API_KEY / MODEL from .env
label = teacher.score(instance, solution)
# label.score: float in [0, 1]
# label.rationale: str (≤ 4 sentences from rubric)
# label.confidence: float in [0, 1] (teacher's self-reported confidence)
# label.basemap_path: Path  (cached PNG for reuse by the student)
```

- Wraps the existing `svrptw.vivrp.assessor._GeminiBackend` with a fixed
  rubric prompt focused on dispatcher acceptance, not visual aesthetics.
- Cached on disk keyed by `(instance.instance_id, solution.solution_hash)`
  in `cache/logic/teacher.sqlite`. Cache hit returns in <5 ms.
- Throttled to ≤ 15 RPM and ≤ 500 RPD (Gemini quota).
- Returns a structured `TeacherLabel` dataclass; never raises on Gemini
  errors — on failure returns `score=None` and the search treats this
  as "no logic signal."

### Preference-pair generator (`svrptw.logic.dataset`)

```python
from svrptw.logic.dataset import build_pairs

pairs = build_pairs(
    bench_path=Path("bench/baselines/v1.json"),
    n_pairs=2500,
    seed=0,
)
# Writes data/logic/pairs.parquet with columns:
#   instance_id, solution_a_hash, solution_b_hash,
#   teacher_label  (1 if A≻B, 0 if B≻A, 0.5 if abstain)
#   teacher_confidence_a, teacher_confidence_b
```

- Group bench rows by `instance_id`, sample within-instance pairs.
- Skew sampling toward pairs that disagree on cost ranking by >5 %
  (these are the interesting ones).
- Target dataset size: 2 000–3 000 labelled pairs from the 720-row
  baseline (≈ 4–6 pairs/instance). At 4 s/teacher-call, ~3 h to label.
- 80/20 train/val split by `instance_id` (no leakage across splits).

### Student (`svrptw.logic.student`)

Architecture (kept deliberately small — must hit <50 ms on CPU):

```
Inputs:
  solution_features  (32-dim engineered: n_routes, capacity_utilisation,
                       tw_tightness_p90, intra-route avg-crossing-count,
                       mean/std route length, breaks/charger compliance,
                       fleet-class mix, ...)
  image_embedding    (256-dim from a frozen small ViT or CLIP-base on the
                       128×128 basemap render; cached per-solution)

Body:
  3-layer MLP, hidden 128, GELU, dropout 0.1.
  Output: a single scalar logit s.

Loss: Bradley-Terry on pairs.
  P(A ≻ B) = sigmoid(s_A - s_B)
  L = - mean_{(A,B,y)} [ y log P(A≻B) + (1-y) log P(B≻A) ]

Calibration: temperature scaling on val set so that
  P(dispatcher_ships) ≈ sigmoid(s / T).
```

API:

```python
from svrptw.logic.student import LogicStudent

s = LogicStudent.load("models/logic/student_v1.pt")
p = s.score(instance, solution)            # float in [0,1]
p_batch = s.score_batch(solutions)         # vectorised
```

### Ensemble (`svrptw.logic.ensemble`)

```python
ens = LogicEnsemble.load("models/logic/ensemble_v1/")
score = ens.score(instance, solution)
# score.mean: float in [0,1]
# score.std:  float (across 5 heads with different seeds + render jitter)
# score.authoritative: bool  (True iff std < 0.08)
```

- 5 heads trained on bootstrap resamples of the pair set + per-head
  render-jitter seeds.
- `authoritative=False` means downstream consumers must treat the score
  as advisory only. Specifically: the bandit (SPEC-6-BANDIT-PLATEAU-01)
  must not basin-jump on logic when `authoritative=False`.

## Acceptance gates

1. **Teacher round-trips on bench**: scoring all 720 baseline rows
   completes in ≤ 3.5 h, with cache enabled second run completes in
   ≤ 30 s.
2. **Student agrees with held-out teacher labels** on val pairs
   ≥ 80 % (Bradley-Terry argmax accuracy).
3. **Student latency** P95 ≤ 50 ms on CPU at N=200 (image embedding
   cached); ≤ 200 ms cold-start.
4. **Calibration**: on val set, the expected calibration error (ECE,
   10 bins) of the temperature-scaled student is ≤ 0.05.
5. **Ensemble σ correlates with student error**: Spearman ≥ 0.4
   between per-pair |s_mean − y| and per-pair σ.

## Non-goals

- Not training a vision backbone end-to-end. The image embedding comes
  from a frozen small ViT/CLIP; only the head sees gradient.
- Not retraining the Gemini teacher. The teacher is a fixed oracle.
- Not building a human-labelled set. The teacher *is* the label
  source. (A small human spot-check on ~50 pairs lands later as
  SPEC-6-LOGIC-02 if we need it.)
- Not promising the student beats the teacher — it just has to be
  *good enough cheap enough* to live in the solver inner loop.

## Files this spec creates

| Path | Role |
|---|---|
| `svrptw/logic/__init__.py` | package |
| `svrptw/logic/teacher.py`  | `GeminiTeacher`, `TeacherLabel`, sqlite cache |
| `svrptw/logic/dataset.py`  | `build_pairs`, parquet writer |
| `svrptw/logic/features.py` | 32-dim solution_features extractor |
| `svrptw/logic/embed.py`    | frozen-ViT image embedder + cache |
| `svrptw/logic/student.py`  | `LogicStudent` MLP + BT loss + temp-scaling |
| `svrptw/logic/ensemble.py` | `LogicEnsemble` over 5 heads |
| `svrptw/logic/train.py`    | CLI: train students + ensemble, write models/logic/ |
| `tests/unit/test_logic_teacher_cache.py` | cache key + persistence |
| `tests/unit/test_logic_student_bt.py`    | BT loss + temp scaling math |

## Open questions (do not block the spec freeze)

- Image embedder choice: frozen `timm/vit_small_patch16_224` (22 M
  params) vs OpenCLIP `ViT-B-32`. Decide once we have the 2.5k pair
  set in hand; bench both for held-out accuracy and CPU latency.
- Whether to add a *third* uncertainty source beyond ensemble σ — e.g.
  rendering the same solution at 3 zoom levels and computing σ across
  views. Deferred to SPEC-6-LOGIC-03 if ensemble σ alone misses the
  Spearman gate.
