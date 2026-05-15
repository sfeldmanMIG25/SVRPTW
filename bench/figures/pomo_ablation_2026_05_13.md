# POMO ablation series (2026-05-13)

Five POMO variants trained and benched today (all with vectorised env;
each training run ~3-5 min).

| variant | epochs | source | N config | mean cost @N=50 (5 inst) | mean cost @N=100 (5 inst) |
|---------|------:|--------|----------|-----------------------:|-------------------------:|
| **v3** (canonical) | 80 | v1 | N=50 only |          **1127.2** |          **2069.9** |
| v5 | 40 | v1 | curr 50/100/200 |   1153.3 |          2143.6 |
| v6 | 80 | v1 | curr 50/100/200 |   1161.4 |          2090.2 |
| v7 | 80 | v1 | N=50 only (repro) |   1129.2 |          2107.6 |
| N=100 specialist | 80 | v1 | N=100 only |   —      |          2097.5 |

(reference: greedy @ N=100 = 1634.50, portfolio @ N=100 = 1474.47)

## Five tested hypotheses

1. **Curriculum helps generalisation.** ❌ — v5/v6 worse than v3 at every N.
2. **More epochs (80 vs 40).** ✓ slight: v6 (80) beats v5 (40), but
   v6 still loses to v3 (80, no curriculum).
3. **v3 was a lucky seed.** ❌ — v7 (same config, fresh seed) lands
   within 0.2 % of v3 at N=50.
4. **Per-N specialist beats N=50 generalist.** ❌ — N=100 specialist
   scored 2097 vs v3's 2069 on N=100 instances.
5. **POMO competitive past N=50 with any tested setup.** ❌ — every
   variant is ~25-30 % worse than greedy at N=100.

## What's left to try

a. **POMO+EAS** (Hottung 2022) — inference-time gradient search over
   the policy logits. Increases POMO wall by 5-10× but produces
   tours significantly better than greedy decode.
b. **Larger encoder** with width, not depth (v4 deeper hurt).
c. **Drop POMO from the bench** and accept that the portfolio's
   12-arm bandit is the SOTA we publish.

(c) is the honest move for the current paper. POMO v3 is preserved
as the cheap-but-mediocre arm of the (cost, wall) frontier; we don't
overclaim. Path to better POMO is real research, not a quick fix.

## v3 stays canonical

The bench-default checkpoint (`DEFAULT_CKPT` in `infer.py`) remains
`models/pomo_v3/pomo_N50_e80.pt`.
