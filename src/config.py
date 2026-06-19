from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ScorecardConfig:
    base_score: int = 680
    base_odds: int = 50
    pdo: int = 20
    min_score: int = 300
    max_score: int = 850


@dataclass(frozen=True)
class PolicyConfig:
    min_age: int = 18
    max_debt_to_income: float = 0.45
    max_crb_defaults: int = 0
    min_monthly_income_kes: float = 15_000
    require_crb_above_limit_kes: float = 100_000


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    version: str
    random_state: int
    data: dict[str, Any]
    training: dict[str, Any]
    scorecard: ScorecardConfig
    decision_bands: dict[str, int]
    policy: PolicyConfig
    channel_minimums: dict[str, dict[str, Any]]
    loan_limits: dict[str, Any]
    model_dir: Path
    reports_dir: Path


def load_config(config_path: str | Path | None = None) -> AppConfig:
    root = Path(__file__).resolve().parents[1]
    path = Path(config_path) if config_path else root / "config" / "scoring.yaml"

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    paths = raw["paths"]
    return AppConfig(
        project_name=raw["project"]["name"],
        version=raw["project"]["version"],
        random_state=raw["project"]["random_state"],
        data=raw["data"],
        training=raw["training"],
        scorecard=ScorecardConfig(**raw["scorecard"]),
        decision_bands=raw["decision_bands"],
        policy=PolicyConfig(**raw["policy"]),
        channel_minimums=raw["channel_minimums"],
        loan_limits=raw["loan_limits"],
        model_dir=root / paths["model_dir"],
        reports_dir=root / paths["reports_dir"],
    )
