# SPEC-0-CFG-01 — Typed config

```
ID:            SPEC-0-CFG-01
Title:         Typed pydantic config replacing config.py constants
Owner role:    OR Engineer
Status:        FROZEN
Inputs:        YAML file path (default: svrptw/config/defaults.yaml)
Outputs:       svrptw.config.Settings (pydantic BaseSettings)
```

## Behavior

Replace the scattered `from config import ...` pattern with a single pydantic-validated `Settings` model. All numeric constants currently in `config.py` and `best_solver_params*.json` are sourced from this. Solvers receive a `Settings` object via constructor; they never import `svrptw.config` constants directly.

```python
class Economics(BaseModel):
    wage_per_hour: float = 14.50
    cost_per_mile: float = 0.50
    hard_late_penalty: float = 1000.0
    early_wait_penalty_per_minute: float | None = None  # defaults to wage/60

class TimeBudget(BaseModel):
    day_start_minute: int = 480       # 08:00
    day_end_minute:   int = 960       # 16:00

class Stochastic(BaseModel):
    enabled: bool = False              # default-OFF per ADR-0001
    travel_lognormal_sigma: float = 0.6
    service_normal_sigma:   float = 8.0
    service_mean:           float = 10.0
    days_per_instance:      int   = 30

class SolverCfg(BaseModel):
    name: Literal["greedy","ortools","dqn","auction","mcts","pomo"]
    params: dict[str, Any] = Field(default_factory=dict)

class Settings(BaseSettings):
    economics:  Economics  = Economics()
    time:       TimeBudget = TimeBudget()
    stochastic: Stochastic = Stochastic()
    solver:     SolverCfg  = SolverCfg(name="greedy")
    seed:       int        = 0
    gart_artifact_path: str = "D:/VRP-Advanced-Estimator-Integration/estimators/lgbm_alpha_model_v4.joblib"
```

## Invariants

- No file under `svrptw/solvers/` imports from `svrptw.config` except for the `Settings` type itself. (`grep "from svrptw.config import" svrptw/solvers/` returns no constant imports — only `Settings`.)
- Loading `defaults.yaml` produces a `Settings` whose numerics match the current `config.py` to the byte. This is a regression assertion in `tests/unit/test_config_parity.py`.
- Unknown fields in YAML are a hard error (pydantic `extra="forbid"`).
- Stochastic block defaults to disabled (ADR-0001 D4).

## Acceptance

- `grep -rn "from config import" svrptw/ specs/ tests/` returns zero hits.
- `python -m svrptw.cli show-config` prints the resolved `Settings` and exits 0.
- `pytest tests/unit/test_config_parity.py` passes — every numeric in `config.py` is reproduced by `Settings.default()`.
- `pytest tests/unit/test_config_strict.py` passes — invalid YAML keys raise `pydantic.ValidationError`.

## Non-goals

- Migrating runtime experiment-tracking config (W&B, Optuna study names). Those stay in their own files.
- Auto-discovering instance paths. The instance manifest is a separate concern (SPEC-0-INST-01).

## Dependencies

None. This is the first concrete code change.
