# Decisions & next steps

## Decision log (this session, in order)

1. **Composability bench spec amendment** (SPEC-8-COUNCIL-02) — user signed off. Tests "13 arms + new arm vs 13 alone at same wall budget" instead of "operator beats portfolio@10s in one shot."
2. **Drop the unit-test gate** from council proposal validation — LLM-generated tests hallucinated constructor signatures. Smoke gate is authoritative.
3. **Add API CHEAT SHEET** to LLM proposer prompt — fixed n=10 batch from 0/10 to 2/10 schema pass.
4. **Paired seeding in composability bench** — threaded `seed=` through `portfolio.solve()` → `LinUCBBandit` → `SISR`. Removed the ~$8 exploration-variance noise floor.
5. **Bumped composability subset 8 → 16 instances** — defeats selection bias.
6. **Wrote off logic-axis** — single-judge labelling lacks resolution to discriminate strong solvers.
7. **Wrote off POMO/EAS as warmstart** (fusion) — checkpoint structurally weaker than auction_gart's.
8. **Adopted PyVRP-warmstart** as the architectural win. Validated across N×instance regime.
9. **Tuned cb=8 + plateaus_to_stop=20** — captures full 30s budget; +$16 per N=200 instance over old defaults.
10. **Refined N-recipe to 2-tier** (was 3-tier): vanilla N<100, warm N≥100. Earlier "vanilla at N≥300" was a pre-tuning artifact.
11. **Mandated parallel-bench harness** — `bench/parallel.py` with max_workers=4; bench wall time cut 3-4×.
12. **External benchmark validation** — Solomon C101 hits published optimum, Homberger N=200 wins 83% at half-budget.

## Open questions / next decision points

### Highest-value next moves
- [ ] **Homberger N=400 result** (in flight) — closes the academic-scale story.
- [ ] **C-class operator gap** — Solomon C is 29% wins (warm tied), Homberger C is 87% wins. Why the inversion at scale? Likely the clustered geography at N=100 falls within PyVRP's local-search neighborhood; at N=200+ the global structure helps the bandit. Worth a focused investigation.
- [ ] **Re-bench v1 leaderboard headline** with `solve_auto()` (2-tier refined recipe) — would update the original report's 95/65/45/72.5% Pareto numbers.
- [ ] **Publish-ready writeup** — the methodology + result are now solid enough for a paper draft.

### Open infrastructure questions
- [ ] **Larger Solomon-Homberger mirrors**: ML4VRP has only 48 of 120 instances; need to find the missing 72.
- [ ] **GART warmstart experiment**: GART estimator is ML-based, like POMO, but produces marginal-length estimates rather than full tours. Could it be used to bias the bandit's *operator choice*, not warmstart? Untested.
- [ ] **Better LLM proposer**: 25 proposals, 0 production wins. Either swap to a larger model (Claude/GPT-5) or pivot the proposer to *instance-generation* (find pathological inputs PyVRP fails on, then operate on those).

### Considered + parked
- LKH-3 as warmstart — tested negative.
- EAS-as-bandit-arm — tested negative.
- LLM committee with multi-model voting — OpenRouter free tier rate-limited too aggressively.
- POMO retraining at larger N — would take days of GPU time; weak evidence of payoff given current POMO checkpoint's structural limitations.

## What's working (don't change without strong reason)

- The 13-arm LinUCB bandit + PyVRP construction composition.
- The paired-seed composability bench at 16+ instances.
- Solomon + Homberger loaders.
- The parallel bench harness with max_workers=4.

## What's not working (revisit)

- The LLM proposer's yield-to-validation ratio (0/25 production-meaningful).
- POMO v3 checkpoint as a competitive baseline at N=200+.
- Logic-axis evaluation under single-judge VLM labelling.
