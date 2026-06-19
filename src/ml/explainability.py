from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import BaseEnsemble
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import BaseDecisionTree

from src.config import AppConfig
from src.domain import Channel, CreditDecision
from src.features.engineering import active_features_for_channel


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
    explanation_scope: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_value": self.base_value,
            "predicted_log_odds": self.predicted_log_odds,
            "summary": self.summary,
            "explanation_scope": self.explanation_scope,
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
    """SHAP explanations for tree and linear credit models."""

    _SCOPE_NOTE = (
        "Attributions reflect the underlying base classifier (pre-calibration). "
        "The credit score uses the calibrated probability of default."
    )

    def __init__(self, pipeline: Pipeline, feature_names: list[str]) -> None:
        self.pipeline = pipeline
        self.feature_names = feature_names
        self._explainer = None
        self._explainer_kind: str | None = None

    def _classifier_step(self):
        return self.pipeline.named_steps["classifier"]

    def _base_estimator(self):
        classifier = self._classifier_step()
        if hasattr(classifier, "calibrated_classifiers_"):
            return classifier.calibrated_classifiers_[0].estimator
        return classifier

    def _transform(self, feature_matrix: pd.DataFrame) -> np.ndarray:
        return self.pipeline.named_steps["preprocessor"].transform(feature_matrix)

    @staticmethod
    def _is_tree_estimator(estimator) -> bool:
        if isinstance(estimator, (BaseDecisionTree, BaseEnsemble)):
            return True
        return hasattr(estimator, "estimators_") or hasattr(estimator, "tree_")

    def _build_explainer(self, background: np.ndarray):
        import shap

        base = self._base_estimator()
        if self._is_tree_estimator(base):
            self._explainer_kind = "tree"
            return shap.TreeExplainer(base)
        if isinstance(base, LogisticRegression):
            self._explainer_kind = "linear"
            return shap.LinearExplainer(base, background)
        self._explainer_kind = "kernel"
        return shap.Explainer(base.predict_proba, background)

    def _get_explainer(self, background: np.ndarray):
        if self._explainer is None:
            self._explainer = self._build_explainer(background)
        return self._explainer

    @staticmethod
    def _default_class_shap_values(shap_values, row_index: int = 0) -> np.ndarray:
        if isinstance(shap_values, list):
            return np.asarray(shap_values[1][row_index])
        values = np.asarray(shap_values)
        if values.ndim == 3:
            return values[1, row_index]
        return values[row_index]

    @staticmethod
    def _default_class_base_value(explainer) -> float:
        expected = explainer.expected_value
        if isinstance(expected, (list, tuple, np.ndarray)):
            expected_array = np.asarray(expected)
            if expected_array.size > 1:
                return float(expected_array.flat[1])
        return float(np.asarray(expected).flat[0])

    def explain(self, feature_matrix: pd.DataFrame, channel: Channel) -> ShapExplanation:
        transformed = self._transform(feature_matrix)
        explainer = self._get_explainer(transformed)
        shap_values = explainer.shap_values(transformed)
        row_values = self._default_class_shap_values(shap_values)
        base_value = self._default_class_base_value(explainer)
        raw_row = feature_matrix.iloc[0]
        allowed_features = active_features_for_channel(channel)

        contributions: list[FeatureContribution] = []
        for feature, shap_value in zip(self.feature_names, row_values):
            if feature not in allowed_features:
                continue
            raw_value = float(raw_row[feature])
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
        filtered_sum = sum(item.shap_value for item in contributions)
        predicted_log_odds = base_value + float(filtered_sum)
        increases = [item.feature for item in top if item.shap_value > 0][:3]
        decreases = [item.feature for item in top if item.shap_value < 0][:3]
        summary = (
            f"Primary drivers increasing default risk: {', '.join(increases) or 'none'}. "
            f"Primary drivers reducing risk: {', '.join(decreases) or 'none'}. "
            f"{self._SCOPE_NOTE}"
        )

        return ShapExplanation(
            base_value=round(base_value, 6),
            predicted_log_odds=round(predicted_log_odds, 6),
            contributions=tuple(top),
            summary=summary,
            explanation_scope=self._SCOPE_NOTE,
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
