from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import AppConfig
from src.domain import ApplicantProfile, Channel, CreditDecision, Decision
from src.features.engineering import ALTERNATIVE_DATA_FEATURES, applicant_to_frame, build_feature_matrix
from src.lending.limit_engine import LoanLimitEngine
from src.ml.explainability import AuditTrailWriter, ShapExplainerService, ShapExplanation
from src.policy.engine import PolicyEngine


class CreditScorer:
    def __init__(
        self,
        config: AppConfig,
        model: Pipeline,
        feature_columns: list[str],
        model_version: str | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.feature_columns = feature_columns
        self.model_version = model_version or config.version
        self.policy_engine = PolicyEngine(config)
        self.limit_engine = LoanLimitEngine(config)
        self.shap_service = ShapExplainerService(model, feature_columns)
        self.audit_writer = AuditTrailWriter(config, self.model_version)

    @classmethod
    def from_latest(cls, config: AppConfig) -> "CreditScorer":
        latest_file = config.model_dir / "latest_model.txt"
        if not latest_file.exists():
            raise FileNotFoundError(
                f"No trained model found in {config.model_dir}. Run train.py first."
            )

        model_name = latest_file.read_text(encoding="utf-8").strip()
        metadata_path = config.model_dir / model_name.replace(".joblib", ".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = joblib.load(config.model_dir / model_name)
        return cls(
            config,
            model,
            metadata["feature_columns"],
            model_version=metadata.get("version", config.version),
        )

    def probability_to_score(self, probability_of_default: float) -> int:
        scorecard = self.config.scorecard
        odds = max(probability_of_default, 1e-6) / max(1 - probability_of_default, 1e-6)
        factor = scorecard.pdo / math.log(2)
        offset = scorecard.base_score - factor * math.log(scorecard.base_odds)
        raw_score = offset + factor * math.log(1 / odds)
        return int(np.clip(round(raw_score), scorecard.min_score, scorecard.max_score))

    def _decision_from_score(self, score: int, policy_passed: bool) -> Decision:
        if not policy_passed:
            return Decision.DECLINE
        if score >= self.config.decision_bands["approve"]:
            return Decision.APPROVE
        if score >= self.config.decision_bands["review"]:
            return Decision.REVIEW
        return Decision.DECLINE

    def _rule_based_factors(
        self,
        feature_row: pd.Series,
        probability: float,
        channel: Channel,
    ) -> list[tuple[str, float]]:
        explanations: list[tuple[str, float]] = []
        if feature_row["crb_defaults"] > 0:
            explanations.append(("crb_defaults", 0.35))
        if feature_row["debt_to_income"] > 0.35:
            explanations.append(("debt_to_income", 0.25))
        if feature_row["loan_to_income"] > 3:
            explanations.append(("loan_to_income", 0.20))
        if channel == Channel.MPESA and feature_row.get("fuliza_utilization", 0) > 0.7:
            explanations.append(("fuliza_utilization", 0.18))
        if channel == Channel.SACCO and feature_row.get("prior_loan_repayment_rate", 1.0) < 0.8:
            explanations.append(("prior_loan_repayment_rate", 0.18))
        if channel == Channel.BANK and feature_row.get("bounced_cheques_12m", 0) > 0:
            explanations.append(("bounced_cheques_12m", 0.16))
        if feature_row.get("lifetime_repayment_rate", 1.0) < 0.8 and feature_row.get("lifetime_loans_count", 0) > 0:
            explanations.append(("lifetime_repayment_rate", 0.22))
        if feature_row.get("on_time_repayment_streak", 0) >= 5:
            explanations.append(("on_time_repayment_streak", -0.15))
        if float(feature_row.get("alternative_data_consent", 0)) >= 0.5:
            if feature_row.get("apps_lending_app_count", 0) > 3:
                explanations.append(("apps_lending_app_count", 0.18))
            if feature_row.get("sms_gambling_ratio", 0) > 0.25:
                explanations.append(("sms_gambling_ratio", 0.16))
        if channel == Channel.MOBILE_LENDER:
            if feature_row.get("mpesa_active_lender_count", 0) > 2:
                explanations.append(("mpesa_active_lender_count", 0.22))
            if feature_row.get("mpesa_inferred_repayment_rate", 1.0) < 0.8:
                explanations.append(("mpesa_inferred_repayment_rate", 0.20))
            if feature_row.get("mpesa_late_repayment_events_12m", 0) > 2:
                explanations.append(("mpesa_late_repayment_events_12m", 0.18))
        if not explanations:
            explanations.append(("model_ensemble_score", float(probability)))
        return sorted(explanations, key=lambda item: item[1], reverse=True)[:3]

    def score_applicant(self, applicant: ApplicantProfile) -> CreditDecision:
        decision, _, _ = self.score_with_audit(
            applicant,
            include_shap=False,
            persist_audit_trail=False,
        )
        return decision

    def score_with_audit(
        self,
        applicant: ApplicantProfile,
        include_shap: bool = True,
        persist_audit_trail: bool = True,
        request_snapshot: dict | None = None,
    ) -> tuple[CreditDecision, ShapExplanation | None, str | None]:
        frame = applicant_to_frame([applicant])
        features = build_feature_matrix(frame).iloc[0]
        matrix = build_feature_matrix(frame)
        probability = float(self.model.predict_proba(matrix)[0, 1])
        credit_score = self.probability_to_score(probability)
        policy = self.policy_engine.evaluate(applicant)
        decision = CreditDecision(
            applicant_id=applicant.applicant_id,
            channel=applicant.channel,
            probability_of_default=round(probability, 4),
            credit_score=credit_score,
            decision=self._decision_from_score(credit_score, policy.passed),
            policy=policy,
            top_risk_factors=self._rule_based_factors(features, probability, applicant.channel),
            metadata={
                "requested_amount_kes": applicant.requested_amount_kes,
                "monthly_income_kes": applicant.monthly_income_kes,
            },
        )
        decision.loan_limit = self.limit_engine.assign(applicant, decision)

        shap_explanation = None
        audit_id = None
        if include_shap:
            shap_explanation = self.shap_service.explain(matrix, applicant.channel)

        if persist_audit_trail and shap_explanation is not None:
            trail = self.audit_writer.write(
                decision,
                shap_explanation,
                request_snapshot or {},
            )
            audit_id = trail.audit_id

        return decision, shap_explanation, audit_id

    def score_batch(self, applicants: list[ApplicantProfile]) -> list[CreditDecision]:
        return [self.score_applicant(applicant) for applicant in applicants]
