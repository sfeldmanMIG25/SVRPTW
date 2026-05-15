# SPEC-7-ARCH-01 — POMO construction → bandit improvement → POMO hyperparam controller

```
ID:            SPEC-7-ARCH-01
Title:         Two-stage pipeline with a novel third controller: POMO
               builds, a beefy improvement phase fixes, and POMO ALSO
               sets the improvement phase's hyperparameters online
Owner role:    Architecture Lead
Status:        FROZEN
Depends on:    SPEC-4-POMO-01 (POMO works), SPEC-3-PORTFOLIO-01
               (bandit framework exists), SPEC-7-OPS-DESTROY-01
               (new destroy ops are arms), SPEC-7-COST-01 (the cost
               gradient that justifies splitting construction from
               improvement)
```

## Why

Empirical evidence from this repo:

- POMO v3 (4-layer attention, 80 epochs, v1 OSM) constructs N=50
  Manhattan in 3 s @ cost 822. Strong starts, but it cannot fix
  itself — its rollout is one-shot.
- Portfolio@10 (LinUCB over 11 operators) improves on greedy by
  ~5 % at N=100 — but it spends most of its time on routes POMO
  would have built better in the first place.
- SISR + SwapStar at N=200 dominate when they have a *strong start*
  to improve. They are mediocre cold-start (no signal to follow).

So the right split is exactly the one the user proposed:

```
            POMO          →           Improvement phase
       (one-shot,         →   (bandit over destroy/repair,
        cheap start)      →    SISR + SwapStar + new destroy ops)
```

This is the conventional half. The **novel half** is:

> POMO does not just hand off a start — it stays online and *tunes
> the improvement phase's hyperparameters* (destruction strength,
> regret-k, plateau-window, basin-jump trigger) while the
> improvement phase is running.

Most current learning-augmented metaheuristics (NeuOpt, L2D,
LLM-LNS) learn to *select* operators. POMO is far better-suited to
**continuous parameter prediction** than to discrete arm selection,
because its decoder is already trained to output a calibrated
softmax over actions — repurposing it to output a calibrated
distribution over hyperparameter settings is a straightforward
fine-tune. To our knowledge no one has shipped this exact
combination.

## Behaviour

### Stage 1 — construction

```python
sol_init, encoder_state = pomo.construct(inst, settings, n_starts=16)
```

`encoder_state` is the encoder embedding of the instance — kept
in memory; the improvement phase queries it.

### Stage 2 — improvement with online hyperparam control

```python
controller = PomoHyperController(encoder_state, decoder_weights)
opt = ImprovementLoop(
    sol_init,
    bandit=LinUCB(...),
    hyperparam_controller=controller,   # <- the novel piece
    ops=DESTROY_REPAIR_OPS,
    plateau_basin_jump=True,            # SPEC-6-BANDIT-PLATEAU-01
)
sol_final = opt.run(time_budget_s=10.0)
```

`controller.predict(state)` returns a tuple of continuous
hyperparameters every K iterations (K ≈ 25):

- `destruction_strength` ∈ [0.05, 0.40]   (fraction of customers removed)
- `regret_k`            ∈ {1, 2, 3}        (categorical via argmax)
- `plateau_window`      ∈ [3, 10]
- `op_temperature`      ∈ [0.2, 2.0]       (softmax over bandit means)

`state` is the same 16-dim state vector the LinUCB bandit already
sees, concatenated with `encoder_state` mean-pooled.

Training signal for the controller: REINFORCE on the
improvement-phase *cost delta* over a 25-iteration window, baseline
= moving average. The same loss machinery as POMO construction —
just a different decoder head.

### Stage 3 — RL termination

The controller has one more output: `terminate ∈ [0, 1]`, the
predicted probability that the improvement phase has converged.
The improvement loop exits when this exceeds 0.8 *and* the
classical plateau detector also fires (defence-in-depth).

## What this gives us that the alternatives don't

| Approach | Construction | Improvement | Hyperparam tuning |
|---|---|---|---|
| Optuna sweep | n/a | n/a | offline, per instance class |
| NeuOpt | learned operator order | manual | none |
| L2D | learned | manual | none |
| LLM-LNS | learned | learned arm select | none |
| **This spec** | learned | learned arm select | **learned continuous, online** |

Online continuous hyperparam learning during search is the only
genuinely novel claim. The rest is well-trodden.

## Risks honestly

1. **Two RL loops on one budget.** Controller updates compete with
   bandit updates for the small data each instance produces.
   Mitigation: the controller updates every 25 iterations, the
   bandit every 1; the controller's gradient is much rarer.
2. **Distillation creep.** If POMO is also doing logic-axis
   distillation (Track 4), it has three jobs. Mitigation: the
   controller is a separate decoder head with its own optimizer;
   POMO's primary loss does not see the hyperparam reward.
3. **Cold-start.** First few hundred steps the controller is
   untrained. Mitigation: bootstrap with hand-tuned defaults from
   the current portfolio config; controller's output is mixed with
   the default via a learned `alpha` until validation shows it
   helps.

## Acceptance gates

1. With `PomoHyperController` disabled, the pipeline is bit-identical
   to the existing POMO+portfolio composition.
2. After 500 v1 N=100 episodes of online updates, the controller's
   chosen hyperparams beat the fixed-default portfolio on ≥ 60 %
   of held-out instances.
3. The controller's chosen `terminate` signal saves ≥ 20 % of the
   improvement-phase wall-clock on instances where it fires, with
   no cost regression on those instances.
4. Total inference overhead of the controller ≤ 50 ms per 25-step
   window at N=200 (small MLP head, cheap).

## Files this spec creates

| Path | Role |
|---|---|
| `svrptw/solvers/learning/pomo/hyper_controller.py` | decoder head + REINFORCE update |
| `svrptw/solvers/classical/improvement_loop.py` | thin orchestrator wrapping bandit + controller |
| `svrptw/solvers/composite_pomo_portfolio.py` | end-to-end pipeline solver registered in bench |
| `tests/unit/test_hyper_controller_default.py` | bit-equality with controller disabled |
| `bench/figures/pomo_then_portfolio.md` | bench writeup once it lands |
