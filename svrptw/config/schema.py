"""Typed configuration. See SPEC-0-CFG-01."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Economics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wage_per_hour: float = 14.50
    cost_per_mile: float = 0.50
    hard_late_penalty: float = 1000.0
    early_wait_penalty_per_minute: float | None = None
    # SPEC-7-COST-01 — opt-in underutilisation + symmetry penalties.
    # Defaults are 0.0 so existing bench results stay bit-identical.
    underutil_penalty_per_route: float = 0.0
    underutil_exponent: float = 2.0
    underutil_target_util: float = 0.70
    symmetry_penalty_coef: float = 0.0
    # SPEC-7-COST-02 — opt-in per-route fixed cost (vehicle-day rental,
    # driver-shift overhead, etc.). Default 0.0 leaves cost identical.
    per_route_fixed_cost: float = 0.0

    @property
    def wage_per_minute(self) -> float:
        return self.wage_per_hour / 60.0

    @property
    def early_wait_per_minute(self) -> float:
        return self.early_wait_penalty_per_minute if self.early_wait_penalty_per_minute is not None else self.wage_per_minute


class TimeBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    day_start_minute: int = 480
    day_end_minute: int = 960


class Stochastic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    travel_lognormal_sigma: float = 0.6
    service_normal_sigma: float = 8.0
    service_mean: float = 10.0
    days_per_instance: int = 30


class SolverCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["greedy", "ortools", "lkh3", "dqn", "auction", "mcts", "pomo"] = "greedy"
    params: dict[str, Any] = Field(default_factory=dict)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", env_prefix="SVRPTW_")
    economics: Economics = Economics()
    time: TimeBudget = TimeBudget()
    stochastic: Stochastic = Stochastic()
    solver: SolverCfg = SolverCfg()
    seed: int = 0
    gart_artifact_path: str = "D:/VRP-Advanced-Estimator-Integration/estimators/lgbm_alpha_model_v4.joblib"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)
