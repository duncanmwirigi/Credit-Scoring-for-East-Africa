from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.domain import ApplicantProfile, Channel


class MpesaFeatures(BaseModel):
    kyc_tier: float = Field(ge=1, le=3)
    wallet_activity_days_90d: float = Field(ge=0, le=90)
    avg_monthly_txn_count: float = Field(ge=0)
    avg_txn_amount_kes: float = Field(ge=0)
    cash_in_out_ratio: float = Field(ge=0)
    merchant_spend_ratio: float = Field(ge=0, le=1)
    fuliza_utilization: float = Field(ge=0, le=1)
    wallet_balance_volatility: float = Field(ge=0, le=1)
    days_since_last_txn: float = Field(ge=0)


class SaccoFeatures(BaseModel):
    membership_months: float = Field(ge=0)
    share_capital_kes: float = Field(ge=0)
    monthly_savings_kes: float = Field(ge=0)
    savings_consistency_score: float = Field(ge=0, le=1)
    prior_loan_repayment_rate: float = Field(ge=0, le=1)
    guarantor_count: float = Field(ge=0)
    guarantor_avg_score: float = Field(ge=300, le=850)
    dividend_years: float = Field(ge=0)


class BankFeatures(BaseModel):
    account_age_months: float = Field(ge=0)
    avg_monthly_balance_kes: float = Field(ge=0)
    salary_deposit_regularity: float = Field(ge=0, le=1)
    bounced_cheques_12m: float = Field(ge=0)
    overdraft_usage_ratio: float = Field(ge=0, le=1)
    credit_card_utilization: float = Field(ge=0, le=1)
    existing_loan_count: float = Field(ge=0)
    branch_relationship_score: float = Field(ge=0, le=1)


class MobileLenderFeatures(BaseModel):
    """Features for app-based digital lenders (Tala, Branch, Zenka, Okash, etc.)."""

    platform_tenure_months: float = Field(ge=0)
    prior_loans_on_platform: float = Field(ge=0)
    platform_repayment_rate: float = Field(ge=0, le=1)
    days_since_last_repayment: float = Field(ge=0)
    active_digital_loans_count: float = Field(ge=0)
    avg_historical_loan_kes: float = Field(ge=0)
    rollover_count_12m: float = Field(ge=0)
    app_engagement_score: float = Field(ge=0, le=1)
    mpesa_disbursement_linked: float = Field(ge=0, le=1)
    alternative_data_score: float = Field(ge=0, le=1)


class ScoreRequest(BaseModel):
    applicant_id: str = Field(min_length=1, max_length=64)
    channel: Channel
    age: int = Field(ge=18, le=100)
    monthly_income_kes: float = Field(gt=0)
    requested_amount_kes: float = Field(gt=0)
    existing_debt_kes: float = Field(ge=0)
    crb_defaults: int = Field(ge=0)
    crb_inquiries_6m: int = Field(ge=0)
    mpesa_features: MpesaFeatures | None = None
    sacco_features: SaccoFeatures | None = None
    bank_features: BankFeatures | None = None
    mobile_lender_features: MobileLenderFeatures | None = None
    include_shap: bool = True
    persist_audit_trail: bool = True

    @model_validator(mode="after")
    def validate_channel_features(self) -> "ScoreRequest":
        if self.channel == Channel.MPESA and self.mpesa_features is None:
            raise ValueError("mpesa_features are required when channel is mpesa.")
        if self.channel == Channel.SACCO and self.sacco_features is None:
            raise ValueError("sacco_features are required when channel is sacco.")
        if self.channel == Channel.BANK and self.bank_features is None:
            raise ValueError("bank_features are required when channel is bank.")
        if self.channel == Channel.MOBILE_LENDER and self.mobile_lender_features is None:
            raise ValueError(
                "mobile_lender_features are required when channel is mobile_lender."
            )
        return self

    def to_applicant(self) -> ApplicantProfile:
        features: dict[str, float] = {}
        if self.mpesa_features:
            features.update(self.mpesa_features.model_dump())
        if self.sacco_features:
            features.update(self.sacco_features.model_dump())
        if self.bank_features:
            features.update(self.bank_features.model_dump())
        if self.mobile_lender_features:
            features.update(self.mobile_lender_features.model_dump())

        return ApplicantProfile(
            applicant_id=self.applicant_id,
            channel=self.channel,
            age=self.age,
            monthly_income_kes=self.monthly_income_kes,
            requested_amount_kes=self.requested_amount_kes,
            existing_debt_kes=self.existing_debt_kes,
            crb_defaults=self.crb_defaults,
            crb_inquiries_6m=self.crb_inquiries_6m,
            features=features,
        )

    def snapshot(self) -> dict[str, Any]:
        return self.model_dump()


class BatchScoreRequest(BaseModel):
    applicants: list[ScoreRequest] = Field(min_length=1, max_length=100)
    include_shap: bool = False
    persist_audit_trail: bool = False


class FeatureContributionResponse(BaseModel):
    feature: str
    raw_value: float
    shap_value: float
    impact: str


class ShapExplanationResponse(BaseModel):
    base_value: float
    predicted_log_odds: float
    summary: str
    explanation_scope: str
    contributions: list[FeatureContributionResponse]


class PolicyResponse(BaseModel):
    passed: bool
    reasons: list[str]


class ScoreResponse(BaseModel):
    applicant_id: str
    channel: Channel
    probability_of_default: float
    credit_score: int
    decision: str
    policy: PolicyResponse
    top_risk_factors: list[list[float | str]]
    shap: ShapExplanationResponse | None = None
    audit_id: str | None = None
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None


class ModelInfoResponse(BaseModel):
    project: str
    version: str
    model_file: str
    feature_count: int
    decision_bands: dict[str, int]
    scorecard: dict[str, int]
