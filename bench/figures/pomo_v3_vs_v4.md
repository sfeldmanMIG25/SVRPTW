# POMO v3 vs v4 vs curriculum — N=50 Manhattan inference

Date: 2026-05-12 → updated 2026-05-13 with curriculum result

| ckpt       | layers | epochs | train source                  | mean cost (3 inst) | mean wall |
|------------|-------:|-------:|-------------------------------|-------------------:|----------:|
| **v3**     |      4 |     80 | v1 OSM (Manhattan + 7 cities) |          **822.1** |    3.3 s |
| v4         |      6 |    150 | synthetic random asymmetric   |            881.1   |    3.3 s |
| curriculum |      4 |     60 | synthetic, mix N=50/100/200   |            900.6   |    3.3 s |

Inference: K=16 POMO greedy decode, per-instance, on
OSM-Manhattan-N050-I{000,001,002}.

## Read

POMO v4 was meant to be a strict upgrade: more capacity (6 layers vs
4), more training (150 epochs vs 80), temperature annealing from 1.0
to 0.3. It is **7 % worse** on the in-distribution test set.

Two confounds, ranked by likely cause:

1. **Training distribution mismatch dominates.** v4 trained on
   synthetic random-asymmetric instances; v3 trained on v1 OSM
   (real road-network). On v1 evaluation, the model that saw v1 at
   train time wins, regardless of capacity. This is the boring,
   expected outcome.
2. **Bigger model needs more data, not just more epochs.** Final
   training cost on v4 cycled in [1180, 1300] for the last 30
   epochs — converged, no more learning happening, just oscillation.

## Decision

- **v3 remains the canonical POMO checkpoint** (`models/pomo_v3/`).
- v4 and curriculum are retained but not promoted. Both used
  synthetic training distributions and both transfer worse than v3
  to v1 OSM.
- **Two confirmations of the same lesson**: SPEC-4-DATA-02
  (network-OD synthetic generator) is now the gating prerequisite
  for any further POMO retrain. Until we have OSM-like training
  data at scale, more compute on synthetic is wasted.
- POMO v5 plan: retrain on v1 OSM directly (small dataset, 160
  instances) for 80 epochs with v3's architecture. Should at
  least match v3; with luck and the additional cities beyond
  Manhattan, exceed it.
