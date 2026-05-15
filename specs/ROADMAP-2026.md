# SVRPTW Solver Roadmap (May 2026)

This document is the master plan tying together the three parallel
research tracks the principal has greenlit. Each track has its own
spec(s); this file is the cross-track schedule, references, and
non-overlap contract.

## Goal

> Best solver possible on realistic instances where a good solution is
> almost impossible to define.

Four implications:
1. The instances must be realistic enough that "real-world" generalisation
   is plausible (Track 1).
2. The solver must be a state-of-the-art metaheuristic + RL hybrid, with
   no single component bottlenecking the others (Track 2).
3. "Good" must be a compound, defensible Pareto signal — not a single
   number (Track 3).
4. The headline number must measure what a dispatcher actually trades
   on — "would I ship this?" — not what the academic VRPTW cost model
   measures (Track 4).

### Headline metric (May 2026 reframe)

> **Fraction of v1 instances where our best solver Pareto-dominates
> PyVRP under (operational_cost, wall_clock_seconds, logic_score).**

This is the number we report. Academic VRPTW cost in isolation is
*not* the headline — those benchmarks ARE the cost model, and the
honest read is that pure RL + metaheuristic does not beat HGS/PyVRP
to four decimals at N=100 today. What we *can* win on is the
commercial benchmark: cost within a small gap, time orders of
magnitude better, and a dispatcher-shaped logic score that the cost
model is blind to. Track 4 builds that third axis.

## Track 1 — Realistic instance generation (SPEC-0-INST-02)

**Status:** Spec frozen; implementation queued after v1 finishes.

**Research distilled** (Track-1 agent, May 2026):

10 features ranked by discriminating power:
1. Mixed hard + soft TWs (SVRPBench '25 — TW alone inflates cost 5-6×)
2. Driver-zone compatibility (Amazon Last Mile '21 corpus)
3. Statutory breaks (EU 561/2006 + FMCSA HOS)
4. Heterogeneous fleet with vehicle-arc compatibility
5. Time-dependent travel times (peak-hour Gaussian mixture)
6. Service-time variability by stop archetype
7. Multi-trip / depot reload
8. Pattern-break stops (lunch, fueling, EV charging)
9. PDP precedence (Sartori-Buriol PDPTW '20 set is the closest reference)
10. Multi-day / periodic visits

Hard validation per instance: OR-Tools feasibility probe (60 s); LKH-3 + HGS-VRPTW with > 2 % cost diff (discriminating); LP set-partition gap > 1 % (non-trivial); fingerprint dedup.

External benchmark adapters to ship in `svrptw.io`: Solomon, Gehring-Homberger, Sartori-Buriol PDPTW, Amazon Last Mile, SVRPBench 2025.

## Track 2 — State-of-the-art solver (SPEC-3-PORTFOLIO-01 + sub-specs)

**Status:** Spec frozen; SISR + granular neighbourhood are next implementations.

**Research distilled** (Track-2 agent, May 2026):

The brutal verdict: **no pure RL beats HGS/PyVRP on N=100 asymmetric today.** Our hybrid path is the right one.

Top SOTA references:
- **PyVRP 0.9+ (Wouda 2024-25)** — won 2024 DIMACS VRPTW; C++ kernels + Python orchestration. Our most direct benchmark.
- **HGS-VRPTW (Vidal / Kool 2022, DIMACS '24 variant)** — granular neighbourhood + SREX + penalty oscillation.
- **SISRs (Christiaens & Vanden Berghe '20)** — string removal + blink-greedy insertion. **Highest ROI addition** for asymmetric tight-TW cases.
- **FILO2 (Accorsi & Vigo '24)** — SA-driven RVND, strongest at N ≥ 10k.
- **POMO + EAS (Hottung et al. '22-24)** — strongest RL baseline; needs asym retraining.
- **NeuOpt / L2D (Ma et al. '24)** — learned operator selection over a fixed pool.

8 implementation tricks ranked by speedup:
1. Doubly-linked-list routes + forward/backward TW accumulators → 50-200× over naïve Python. (SPEC-3-DLL-01, new sub-spec.)
2. Granular neighbourhood (k=20 time-aware nearest neighbours).
3. C++ / Cython / numba on inner loops only (PyVRP's pattern).
4. SISR string-removal operator (50 lines; replaces ejection chain on tight TWs).
5. Regret-k insertion (k=2 or 3) after ruin operators.
6. Don't-look bits on customers — 3-5× speedup.
7. Lazy inter-route 2-opt* with cached segment reversal.
8. Penalty-adjusted feasibility oscillation (HGS-DIMACS).

RL plan (three coordinated tracks):
- **RL-1 Construction warm-start:** POMO-style attention model retrained on `instances/v2` asymmetric distribution. Produces 5-10 diverse warm starts (SPEC-4-POMO-01 extends to this).
- **RL-2 Operator selection:** Contextual bandit (LinUCB) over the operator pool, conditioned on instance + state features. Promotes to small MLP policy (SPEC-3-OPSEL-01) once we have episode data.
- **RL-3 Hyperparameter tuning:** Optuna sweep on operator-selection temperatures, penalty oscillation rates, regret-k choice. Offline; per instance class.

GART's role becomes **a critic for gap reporting and beam-search scoring**, not a primary cost model.

The LKH-3 customer-drop trick that beats us at N=100 needs to be added explicitly as a **soft-drop operator** (drop customers whose insertion-cost > `hard_late_penalty`).

## Track 3 — Compound evaluation (SPEC-5-VIVRP-02 + sub-specs)

**Status:** Spec frozen; constrained decoding + multi-pass zoom are next implementations.

**Research distilled** (Track-3 agent, May 2026):

VLM-as-judge stack to adopt:
- **Structured output via constrained decoding** — Outlines for local, Gemini `response_schema` for cloud. Eliminates ~90 % of score clumping. Already partially in place; needs grammar enforcement.
- **Multi-pass with zoom ROI** — pass 1 full map → bbox of "worst region" → pass 2 crop → pass 3 (optional) 2×2 tile. Biggest quality lever per V*/SEAL/Visual-CoT 2025-26 lit.
- **N=3-5 jittered renders + median** — cuts published VLM-judge variance roughly in half.
- **Isotonic calibration** on a ~50-pair human-labelled set. Without it, raw VLM scores are ordinal.
- Honest caveat: no public benchmark cleanly measures VLM-vs-human on VRPTW specifically; ViTSP and GPT-4V combinatorial papers report Spearman ~0.55-0.70. **Treat VLM score as a noisy objective, never as the objective.**

Pareto/multi-objective stack (5 objectives: cost, missed, time-violation, cluster-coherence, VLM-score):
- **Non-dominated sorting** (NSGA-III for ≥ 4 objectives).
- **Hypervolume (WFG)** as the primary frontier metric.
- **IGD+** vs. a reference front (e.g. long Gurobi run on small instances).
- **R2** as a cheap HV proxy when |front| explodes.
- **Pareto pruning** via k-medoids (k=20-30) for human review.
- **Final ranking for single decision** — lexicographic on hard constraints, then HV-contribution. Never weighted sums.

VLM feedback as RL signal (the highest-probability win):
- **Contextual bandit** (LinUCB / Thompson) over destroy-repair operators.
- **Context** = VLM `worst_region_bbox` + critique scalars.
- **Action** = (operator, region).
- **Reward** = post-move Pareto HV-contribution improvement, + VLM-delta as ≤ 20 % auxiliary term.
- Converges in hundreds of steps. No GPU training. Precedent: Liu et al. "LLM-guided LNS" 2024, ICLR '25 Vision-Guided Heuristics workshop.
- **Distillation Gemini → Qwen LoRA**: 5-10 k preference pairs, ~$50-150 Gemini cost, 4-8 h on one 4090. Premature now; queue after the bandit ships.

## Track 4 — Logic axis (SPEC-6-LOGIC-01 + SPEC-6-PARETO-3AXIS-01 + SPEC-6-BANDIT-PLATEAU-01)

**Status:** Specs frozen; implementation queued behind in-flight POMO/bench runs.

Three components, in implementation order:

1. **Teacher → student distillation** (SPEC-6-LOGIC-01 + SPEC-6-LOGIC-02).
   Teacher is no longer a single Gemini call — it is now a
   tier-weighted **OpenRouter free-tier committee** of ~11 vision
   models across 6 model families (Google Gemma, Alibaba Qwen,
   NVIDIA Nemotron, Mistral, Meta Llama, Z.ai/Moonshot), plus a
   cloaked Tier-S "stealth" tier with prompt sanitisation.
   Consensus = tier-weighted median, σ = variance across responding
   models, `authoritative` iff ≥ 6 responders and σ ≤ 0.15. Gemini
   3.1 Flash Lite remains as one optional voter via the existing
   ViVRP backend. Free-tier capacity: ~2 200 vision judgments/day
   from OpenRouter alone; ~6 000/day with Gemini + Groq.
   Preference pairs (2-3 k) generated from existing 720-row bench.
   Bradley-Terry-trained 3-layer MLP over (32-d solution features,
   256-d frozen-ViT image embedding). Latency target <50 ms on CPU.
   Ensemble of 5 heads + variance threshold for "authoritative"
   predictions on the *student* side too.

2. **Three-axis Pareto reporting** (SPEC-6-PARETO-3AXIS-01).
   `svrptw.bench.pareto3.dominance_report` returns the fraction of
   instances where the challenger strictly dominates the baseline
   under (cost, time, logic). The logic axis drops out for instances
   where the ensemble is non-authoritative — this is the
   uncertainty-aware reporting contract.

3. **Plateau-driven basin-jump** (SPEC-6-BANDIT-PLATEAU-01).
   When LinUCB hits its plateau exit, redirect one perturbation
   under the logic objective, then resume cost-driven search. Hard
   no-regret guarantee on the *returned* solution; the *trajectory*
   is allowed to climb.

Why this is honest:

- The teacher does not become a load-bearing oracle. The student is
  *distilled* and *ensembled* before any downstream consumer uses it.
- The bandit never optimises logic directly. It only consults logic
  at plateau, and only when the ensemble is confident.
- The headline metric (Track 4 reframe above) drops the logic axis on
  any instance where the student is unsure — we never claim a win
  on noise.

The reason this is a separate track and not a sub-spec under Track 3:
Track 3's VLM front (5-objective NSGA-III) is for *analysis*; Track 4
is for the *reported headline*. They consume the same Gemini backend
but answer different questions. Track 3 = "show me the frontier";
Track 4 = "is this shipped?"

## Track 5 — Cost model + new operators + 2-stage pipeline + network-OD data (May 2026 expansion)

**Status:** Specs frozen; implementation gated on completion of in-flight
POMO curriculum + PyVRP SOTA bench (do not invalidate cost model mid-run).

Four threads that together close the cost/logic divergence surfaced in
`bench/figures/v1_vivrp_gemini_smoke.md` (portfolio wins cost -18.7 %
vs LKH-3 but LKH wins Gemini judge 4.50 vs 1.70):

1. **Exponential underutilisation penalty** (SPEC-7-COST-01). The
   solver currently gets new routes "for free" — there is no fixed
   cost or utilisation penalty per route. New `Economics` fields
   `underutil_penalty_per_route` (default 0 = back-compat),
   exponent (default 2.0), target_util (default 0.70), plus a
   cross-route symmetry penalty. Defaults activated once
   in-flight benches complete.

2. **Three new destroy operators** (SPEC-7-OPS-DESTROY-01).
   `destroy_island` (Voronoi/Thiessen cluster removal),
   `drop_leg` (lowest-load-utilisation edge),
   `drop_route` (entire under-utilised route). These are the moves
   the new cost gradient wants — SISR alone cannot delete a route,
   only shorten it.

3. **Two-stage pipeline with POMO as both constructor and online
   hyperparameter controller** (SPEC-7-ARCH-01). POMO builds, a
   bandit-driven improvement phase fixes, and POMO *also* predicts
   continuous hyperparameters (destruction strength, regret-k,
   plateau-window, temperature, termination probability) every 25
   improvement iterations. To our knowledge no current paper ships
   this combination — operator-selection bandits are common,
   continuous online hyperparam control during search is not.

4. **Network-OD synthetic generator** (SPEC-4-DATA-02). The current
   synthetic path is Euclidean + multiplicative edge noise — the
   exact reason POMO v4 transferred poorly from synthetic to OSM
   Manhattan. New generator produces directed graphs (grid-with-
   arterials, fractal-radial, mesh-with-bridges, osm-like-sampled)
   and emits T/D as shortest-path matrices, naturally asymmetric
   from one-ways and congestion profiles rather than from i.i.d.
   noise. POMO v5 retrains on this.

## Sub-specs to write (in order)

| ID | Title | Owner | When |
|---|---|---|---|
| SPEC-3-DLL-01 | Doubly-linked-list route reps + TW accumulators | OR | First — 50-200× win |
| SPEC-3-SISR-01 | SISR string-removal + blink-greedy insertion | OR | Second — highest single-operator ROI |
| SPEC-3-GRAN-01 | Granular neighbourhood (k=20 time-aware) | OR | Third — speeds every existing operator |
| SPEC-3-PENOSC-01 | Penalty-adjusted feasibility oscillation | OR | Fourth — needed for soft-drop trick |
| SPEC-3-SOFTDROP-01 | Soft customer-drop operator (close N=100 gap to LKH-3) | OR | Fifth |
| SPEC-3-OPSEL-02 | Contextual bandit over operators with VLM region context | ML | After SISR+granular ship |
| SPEC-5-VIVRP-03 | Multi-pass zoom + jittered render + isotonic calibration | ML | Parallel with bandit |
| SPEC-5-PARETO-01 | NSGA-III front + WFG hypervolume + IGD+ | Bench | Once SPEC-5-VIVRP-03 ships |
| SPEC-4-POMO-02 | POMO warm-start trained on instances/v2 asymmetric | ML | After v2 lands |
| SPEC-3-LKHCOMPARE-01 | PyVRP 0.9 + HGS-DIMACS + FILO2 adapters | Bench | First parallelisable |

## Non-overlap contract

- Track 1 and Track 2 share `svrptw.io.instance` (Track 1 owns the schema;
  Track 2 reads it).
- Track 2 and Track 3 share `svrptw.solvers.common.solution` (Track 2
  produces; Track 3 evaluates).
- The bandit context interface (`worst_region_bbox`) lives in
  `svrptw.vivrp.assessor` (Track 3) and is consumed by
  `svrptw.solvers.learning.op_select` (Track 2).

## What this roadmap is not

- A commitment to ship everything in the next two weeks. The honest
  estimate: SPEC-3-DLL-01 + SISR + granular = 1 week. POMO retraining
  = 1 week. v2 instance set + adapters = 1 week. Pareto + multi-pass
  VLM = 1 week. RL-from-VLM-feedback = 2 weeks. Total: ~6 calendar
  weeks of focused work.

- Promising to "beat OR-Tools / LKH-3 / HGS across all instance classes
  at all budgets." Honest: we will beat OR-Tools and LKH-3 on most
  classes; PyVRP / HGS-DIMACS is the real bar. The Pareto framing
  matters because at any single class one solver will win — what
  generalises is the *frontier*.
