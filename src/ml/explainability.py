from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import AppConfig
from src.domain import CreditDecision


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    raw_value: float
    shap_value: float
    impact: str


@dataclass(frozen=True)
class ShapExplanation:
    base_value: float
    predicted_log_odds: float
    contributions: tuple[FeatureContribution, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_value": self.base_value,
            "predicted_log_odds": self.predicted_log_odds,
            "summary": self.summary,
            "contributions": [asdict(item) for item in self.contributions],
        }


@dataclass(frozen=True)
class RegulatoryAuditTrail:
    audit_id: str
    scored_at: str
    model_version: str
    applicant_id: str
    channel: str
    probability_of_default: float
    credit_score: int
    decision: str
    policy_passed: bool
    policy_reasons: tuple[str, ...]
    shap: dict[str, Any]
    request_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ShapExplainerService:
    """Model-agnostic SHAP wrapper for tree-based credit pipelines."""

    def __init__(self, pipeline: Pipeline, feature_names: list[str]) -> None:
        self.pipeline = pipeline
        self.feature_names = feature_names
        self._explainer = None

    def _base_estimator(self):
        classifier = self.pipeline.named_steps["classifier"]
        if hasattr(classifier, "calibrated_classifiers_"):
            return classifier.calibrated_classifiers_[0].estimator
        return classifier

    def _transform(self, feature_matrix: pd.DataFrame):
        return self.pipeline.named_steps["preprocessor"].transform(feature_matrix)

    def _get_explainer(self):
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self._base_estimator())
        return self._explainer

    def explain(self, feature_matrix: pd.DataFrame) -> ShapExplanation:
        import shap

        transformed = self._transform(feature_matrix)
        shap_values = self._get_explainer().shap_values(transformed)
        if isinstance(shap_values, list):
            row_values = shap_values[1][0]
        else:
            row_values = shap_values[0]

        expected = self._get_explainer().expected_value
        base_value = float(expected[1] if isinstance(expected, (list, tuple)) else expected)
        raw_row = feature_matrix.iloc[0]

        contributions: list[FeatureContribution] = []
        for feature, shap_value in zip(self.feature_names, row_values):
            raw_value = float(raw_row[feature])
            if raw_value == 0.0 and abs(float(shap_value)) < 1e-8:
                continue
            impact = "increases_default_risk" if shap_value > 0 else "decreases_default_risk"
            contributions.append(
                FeatureContribution(
                    feature=feature,
                    raw_value=round(raw_value, 4),
                    shap_value=round(float(shap_value), 6),
                    impact=impact,
                )
            )

        contributions.sort(key=lambda item: abs(item.shap_value), reverse=True)
        top = contributions[:10]
        predicted_log_odds = base_value + float(sum(row_values))
        increases = [item.feature for item in top if item.shap_value > 0][:3]
        decreases = [item.feature for item in top if item.shap_value < 0][:3]
        summary = (
            f"Primary drivers increasing default risk: {', '.join(increases) or 'none'}. "
            f"Primary drivers reducing risk: {', '.join(decreases) or 'none'}."
        )

        return ShapExplanation(
            base_value=round(base_value, 6),
            predicted_log_odds=round(predicted_log_odds, 6),
            contributions=tuple(top),
            summary=summary,
        )


class AuditTrailWriter:
    def __init__(self, config: AppConfig, model_version: str) -> None:
        self.config = config
        self.model_version = model_version
        self.audit_dir = config.reports_dir / "audit_trails"
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        decision: CreditDecision,
        shap_explanation: ShapExplanation,
        request_snapshot: dict[str, Any],
    ) -> RegulatoryAuditTrail:
        audit_id = f"{decision.applicant_id}-{uuid.uuid4().hex[:8]}"
        trail = RegulatoryAuditTrail(
            audit_id=audit_id,
            scored_at=datetime.now(timezone.utc).isoformat(),
            model_version=self.model_version,
            applicant_id=decision.applicant_id,
            channel=decision.channel.value,
            probability_of_default=decision.probability_of_default,
            credit_score=decision.credit_score,
            decision=decision.decision.value,
            policy_passed=decision.policy.passed,
            policy_reasons=decision.policy.reasons,
            shap=shap_explanation.to_dict(),
            request_snapshot=request_snapshot,
        )
        path = self.audit_dir / f"{audit_id}.json"
        path.write_text(json.dumps(trail.to_dict(), indent=2), encoding="utf-8")
        return trail
