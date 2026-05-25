---
title: Architecture Review (2026-05-15 epic, end of iter 4)
project: SVRPTW
tags: [architecture, review, options, current-state]
updated: 2026-05-15
---

# Architecture Review — what's running, what works, what's next

## TL;DR

The "architectural win" from the prior session (`portfolio_pyvrp_warm` beating PyVRP@60s by +$110/inst on held-out OSM instances at N=100–500) is now **operationalized**:

1. **N=400 Homberger reversal is fixed** in production via `scaled_cb()` in `solve_auto`.
2. **Live web UI** at `http://127.0.0.1:8765/` shows benches, snapshots, judges, agent dashboard, cost curves, and the canonical Progress Report all on one page.
3. **Multi-judge VLM panel** (OpenRouter committee + Gemini direct + Anthropic Haiku, tier-weighted-median + dissent) is wired and tested.
4. **RL replacement for the LinUCB bandit** is implemented (MLP policy + reward shaping + offline trainer + A/B harness) and ships behind `solve_auto(rl_neighborhood=True)`.

You can drive routine work entirely from the browser; bench scripts stream their progress + per-iteration snapshots + cost curves into the page in real time.

## Layer-by-layer architecture

### Solver core
```
solve_auto(inst, settings, budget_seconds, ..., on_accept=, rl_neighborhood=)
  ├─ N <  100: pm.solve(...)   [vanilla portfolio + LinUCB]
  └─ N >= 100: pyw.solve(...)
        ├─ pv.solve(inst, ...)   [PyVRP construction; budget = scaled_cb(N) capped at 30%·budget_s]
        └─ pm.solve(inst, ..., initial_solution=warm)   [LinUCB bandit refinement]
                ├─ on every accepted move: bandit.update + on_accept(op, Δ, i, sol)
                └─ if rl_neighborhood=True → bandit_kind="mlp" + shape_reward=True
```

- **`scaled_cb(N, base_cb=8, n_pivot=200)`** — at N≤200 returns 8.0; at N=400 returns 16.0; at N=500 returns 20.0. Downstream cap = `min(cb, 0.30*budget_s)` keeps it sane.
- **`on_accept(op_name, improvement, ops_applied, new_sol)`** — fires only when bandit accepts a move. Used by the web UI to push live snapshots + cost curves.
- **`bandit_kind in {"linucb", "logging", "mlp"}`** — `linucb` is default (unchanged behavior); `logging` wraps LinUCB and dumps transitions to JSONL; `mlp` loads a trained policy artifact.
- **`shape_reward=True`** — adds island/isolated-stop/leg-disposal terms to the reward (per Human Reflection 10 item 3).

### Web UI architecture
```
PowerShell terminal A:                        PowerShell terminal B:
  pwsh scripts\launch_webui.ps1                $env:SVRPTW_WEBUI_URL = "http://127.0.0.1:8765"
    ↓ spawns (DETACHED_PROCESS)                python bench/scripts/cb_scaled_rebench.py --webui
    python -m webui.app                          ↓ workers inherit env var
       ↓ binds 127.0.0.1:8765                   ↓
       opens browser                            POST /event {type, payload} on every
                                                   stage / snapshot / progress / judge / log / agent / series
                                                ↓
        ┌──────────────────────┐                ↓
        │ EventBus singleton   │ ←─── HTTP ─────┘
        │  - stage             │
        │  - snapshots         │ ─── PNG ───→  webui/static/snapshots/*.png
        │  - bench progress    │              ↑ rendered by render_llm_compare
        │  - judges            │                (basemap when geographic)
        │  - log (last 400)    │
        │  - agents            │
        │  - series (cost-curve) ← polled by browser (1.5s)
        └──────────────────────┘
                ↓ /status
            browser
              renderAgents / renderBenches / renderSnapshots /
              renderJudges / renderLog / renderCharts +
              <iframe src=/reports/progress>
```

Singleton lives **per process**. Cross-process push goes through `/event` HTTP. The `webui.client` module auto-detects `SVRPTW_WEBUI_URL` and routes to HTTP or in-process bus accordingly.

### Multi-judge panel
```
judge_pair(image_a, image_b, prompt, ...)
  ├─ OpenRouter committee (Tier-A free vision pool, 4 models)
  ├─ Gemini direct (gemini-2.0-flash via genai SDK, same auth as proposer)
  └─ Anthropic Haiku (claude-3-5-haiku-latest, lazy-imports `anthropic`)
        ↓
  ThreadPoolExecutor fan-out, per-call try/except
        ↓
  aggregate_verdicts → tier-weighted median (S=0.5/A=1.0/B=0.6/C=0.4)
                       + dissent flag (max-min > 0.3)
        ↓
  webui.client.push_judges(pair_id, judges, consensus)
        ↓
  page renders judges panel (per-model row + consensus + dissent badge)
```

Anthropic SDK is not installed; Haiku currently returns a single `error="anthropic-not-available"` verdict. Install with `pip install anthropic` to enable.

## What's running RIGHT NOW

| component | status | where |
|-----------|--------|-------|
| webui server | running, pid varies | `http://127.0.0.1:8765/` |
| BUI sub-agent | in flight | building `/control/*` + `/viewer` + integration test |
| D sub-agent | **completed** | RL impl done; 122 tests pass |
| demo-N100-Manhattan-warm series | 40 fake pts | chart panel demo |
| warm-OSM-Manhattan-N100-I003 | 5 real pts, 2 frames | live bandit accepts |
| warm-OSM-Paris-N200-I003 | 6 real pts, 2 frames | live bandit accepts |

## What you can DO right now

### From the browser (no command line)
- **Watch the agents panel** — sub-agents I dispatch appear as rows with status + elapsed time.
- **Click around the snapshots grid** — each tile is a real route plot from a bandit accept (basemap underneath; LLM-comp safe colors).
- **Read the cost-over-iteration sparklines** — green = improving, amber = flat. One per stream.
- **Scroll to the bottom** — the canonical `Progress_Report.html` is embedded inline as an iframe.
- **Click the "reports →" header link** — opens `/reports` (just the index of available HTML reports).

When the BUI sub-agent finishes, additional buttons appear at top: ▶ Run bench, ■ Stop, ⛶ Viewer (full-page side-by-side time-lapse), 📄 Publish report, Reset bus, Clear snapshots.

### From PowerShell (current capabilities)

```powershell
# bench: 72 tasks, parallel, live-streams to UI
python bench/scripts/cb_scaled_rebench.py --webui --workers 4

# bench: v1 leaderboard refresh
python bench/scripts/v1_leaderboard_solve_auto.py --webui --workers 4

# RL: collect bandit transitions from solve_auto runs
python bench/scripts/collect_bandit_logs.py

# RL: train MLP policy offline on those logs
python bench/scripts/train_policy_offline.py

# RL: A/B the trained policy against LinUCB
python bench/scripts/rl_vs_linucb_ab.py --webui --rl-artifact bench/runs/policy_v1.pt

# multi-judge: re-judge an existing rebench run via VLM panel
python bench/scripts/judge_solutions_multi.py --input bench/runs/cb_scaled_rebench.json --limit 8 --judges openrouter gemini --webui

# time-lapse: GIF of any operator stream
python -m webui.timelapse --list
python -m webui.timelapse --stream warm-OSM-Manhattan-N100-I003

# one-shot demo solve (the one that populated the dashboard)
python scripts/demo_solve.py instances/v1/OSM-Paris-N200-I003.json 20
```

## Strategic options — pick where to push next

The pasted reflection from earlier this epic said: **crystallize first, push second**. Phases A1–A3 (the cb-scaling + bench scripts) are done; Phase A4 (the writeup) needs A2/A3 to *actually run end-to-end* on the real instance set first. Three concrete directions you can pick from:

### Option 1 — RUN THE BENCHES (closes the crystallization loop)
- Start `cb_scaled_rebench.py --webui --workers 4` and let it run 25-40 min.
- Watch the dashboard fill up: 72 tasks × (auto + PyVRP@2x baseline) with live snapshots and cost curves.
- Check whether the cb-scaling restores parity at Homberger N=400 (the open question from iter 0). The dashboard shows you per-instance deltas as they complete.
- **Then** start `v1_leaderboard_solve_auto.py --webui` to refresh the headline 95/65/45/72.5% numbers in `Progress_Report.html`.
- **Then** A4 writeup with the fresh numbers. This is the lowest-effort, highest-confidence path.

### Option 2 — TRAIN THE RL POLICY
- Run `collect_bandit_logs.py` for a couple hours over a diverse instance subset → `bench/runs/bandit_transitions.jsonl`.
- Run `train_policy_offline.py` → `bench/runs/policy_v1.pt`.
- Run `rl_vs_linucb_ab.py --webui` to compare. Watch the cost curves in the dashboard.
- This is the highest-payoff-if-it-works direction (the user's vision in Human Reflection 10 was specifically "stochastic trained model that chooses what neighborhoods to apply"). But it's research — could go several ways.

### Option 3 — STRESS THE MULTI-JUDGE
- Set `SVRPTW_OPENROUTER_API_KEY` + `SVRPTW_GEMINI_API_KEY`.
- Run `judge_solutions_multi.py --input bench/runs/cb_scaled_rebench.json --limit 16 --judges openrouter gemini --webui`.
- The judges panel populates in real time with model-by-model scores + dissent flags.
- The pasted reflection suggested *parking* this; Human Reflection 10 + your explicit ask said *build it*. The path is now stress-test it on real pairs and decide whether multi-judge agreement is high enough to be a useful selection signal.

### Option 4 — INSTANCE GENERATION (per pasted reflection)
- The LLM proposer is built and has plumbing. Repurpose it to generate **instance distributions** (large-N stress instances) — the Homberger N=400 reversal would have been caught earlier if such instances existed in the held-out set.
- This is a small spec + small bench: ~half-day of work.

### Option 5 — COST-MODEL EXPLORATION
- `per_route_fixed_cost` is the existing structural-advantage lever (PyVRP can't see it). Add more such terms: peak-hour penalty, fairness across drivers, route-shape regularizers.
- Each new term that LinUCB can optimize but PyVRP cannot widens our lead.
- Tractable: ~1-2 days per term + re-bench.

## My recommendation

**Option 1 first**, because it costs ~30 min of bench time and locks in the architectural-win narrative that's been built for two iterations. The dashboard will SHOW you whether the cb-scaling fix worked at N=400 — that's the closing answer for the crystallization phase.

**Option 2 second**, because the RL infra is fresh in head and the user explicitly asked for it. Even a negative result ("MLP behavior-cloned policy can't beat tuned LinUCB") closes a question.

**Options 3-5 later**, only after 1-2 give clean numbers worth comparing against.

## Cross-references
- [[01 - Progress Report]] — epic snapshot (kept current per iter)
- [[05 - Decisions & Next Steps]] — epic pivot log
- [[10  - Human Reflection]] — user's vision
- [[11 - RL Roadmap]] — full RL design spec (D2-D5)
- `Sessions/2026-05-15_iter2.md`, `_iter3.md`, `_iter4.md`, `_iter3_subagent_C.md`, `_iter4_subagent_D.md` — per-iteration logs
