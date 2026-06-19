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
    """M-Pesa statement-derived signals — no third-party lender API required."""

    mpesa_statement_days_covered: float = Field(ge=0, le=365)
    mpesa_lender_disbursement_count_12m: float = Field(ge=0)
    mpesa_lender_repayment_count_12m: float = Field(ge=0)
    mpesa_inferred_repayment_rate: float = Field(ge=0, le=1)
    mpesa_active_lender_count: float = Field(ge=0)
    mpesa_avg_inferred_loan_kes: float = Field(ge=0)
    mpesa_late_repayment_events_12m: float = Field(ge=0)
    mpesa_loan_rollover_signals_12m: float = Field(ge=0)
    mpesa_net_cashflow_kes_90d: float = Field(default=0)


class LendingHistoryFeatures(BaseModel):
    lifetime_loans_count: float = Field(ge=0, default=0)
    lifetime_loans_repaid_on_time: float = Field(ge=0, default=0)
    lifetime_default_count: float = Field(ge=0, default=0)
    lifetime_repayment_rate: float = Field(ge=0, le=1, default=1.0)
    on_time_repayment_streak: float = Field(ge=0, default=0)
    avg_days_past_due: float = Field(ge=0, default=0)
    days_since_last_loan: float = Field(ge=0, default=9999)
    days_since_last_default: float = Field(ge=0, default=9999)
    current_outstanding_kes: float = Field(ge=0, default=0)
    highest_prior_limit_kes: float = Field(ge=0, default=0)
    months_customer_relationship: float = Field(ge=0, default=0)


class PhoneSmsData(BaseModel):
    sms_salary_detected: float = Field(ge=0, le=1, default=0)
    sms_inferred_monthly_income_kes: float = Field(ge=0, default=0)
    sms_mpesa_txn_count_30d: float = Field(ge=0, default=0)
    sms_total_count_30d: float = Field(ge=0, default=0)
    sms_bill_pay_regularity: float = Field(ge=0, le=1, default=0)
    sms_other_lender_repayment_count: float = Field(ge=0, default=0)
    sms_collection_message_count_30d: float = Field(ge=0, default=0)
    sms_lender_promo_count_30d: float = Field(ge=0, default=0)
    sms_gambling_ratio: float = Field(ge=0, le=1, default=0)
    income_declared_vs_sms_ratio: float = Field(ge=0, default=1.0)


class PhoneCallData(BaseModel):
    call_total_count_30d: float = Field(ge=0, default=0)
    call_unique_contacts_30d: float = Field(ge=0, default=0)
    call_avg_duration_seconds: float = Field(ge=0, default=0)
    call_incoming_ratio: float = Field(ge=0, le=1, default=0.5)
    call_missed_ratio: float = Field(ge=0, le=1, default=0)
    call_collection_agency_count_30d: float = Field(ge=0, default=0)
    call_night_activity_ratio: float = Field(ge=0, le=1, default=0)


class PhoneDeviceData(BaseModel):
    device_tenure_days: float = Field(ge=0, default=0)
    contacts_count: float = Field(ge=0, default=0)
    contacts_saved_ratio: float = Field(ge=0, le=1, default=0)
    apps_lending_app_count: float = Field(ge=0, default=0)
    apps_gambling_app_count: float = Field(ge=0, default=0)
    device_os_android: float = Field(ge=0, le=1, default=1)
    device_os_version_score: float = Field(ge=0, le=1, default=0.5)
    device_tier: float = Field(ge=1, le=3, default=2)
    device_ram_gb: float = Field(ge=0, default=2)
    device_storage_free_ratio: float = Field(ge=0, le=1, default=0.5)
    device_dual_sim: float = Field(ge=0, le=1, default=1)
    device_network_4g_plus: float = Field(ge=0, le=1, default=1)
    device_model_age_months: float = Field(ge=0, default=24)


class DataSourcesFeatures(BaseModel):
    """Flags for which scoring data sources are available for this applicant."""

    has_mpesa_wallet: float = Field(ge=0, le=1, default=0)
    has_phone_consent: float = Field(ge=0, le=1, default=0)
    has_bank_account: float = Field(ge=0, le=1, default=0)
    has_sacco_membership: float = Field(ge=0, le=1, default=0)
    has_crb_record: float = Field(ge=0, le=1, default=0)


class PhoneDataFeatures(BaseModel):
    """On-device phone permissions: SMS inbox, call log, contacts, installed apps."""

    alternative_data_consent: float = Field(ge=0, le=1, default=0)
    sms: PhoneSmsData = Field(default_factory=PhoneSmsData)
    calls: PhoneCallData = Field(default_factory=PhoneCallData)
    device: PhoneDeviceData = Field(default_factory=PhoneDeviceData)

    def flatten(self) -> dict[str, float]:
        payload = {"alternative_data_consent": self.alternative_data_consent}
        payload.update(self.sms.model_dump())
        payload.update(self.calls.model_dump())
        payload.update(self.device.model_dump())
        return payload


# Backward-compatible alias for API clients using the old block name.
AlternativeDataFeatures = PhoneDataFeatures


class ScoreRequest(BaseModel):
    applicant_id: str = Field(min_length=1, max_length=64)
    channel: Channel
    age: int = Field(ge=18, le=100)
    monthly_income_kes: float = Field(gt=0)
    requested_amount_kes: float = Field(gt=0)
    existing_debt_kes: float = Field(ge=0)
    crb_defaults: int = Field(ge=0)
    crb_inquiries_6m: int = Field(ge=0)
    lending_history: LendingHistoryFeatures = Field(default_factory=LendingHistoryFeatures)
    data_sources: DataSourcesFeatures = Field(default_factory=DataSourcesFeatures)
    phone_data: PhoneDataFeatures = Field(default_factory=PhoneDataFeatures)
    alternative_data: PhoneDataFeatures | None = Field(
        default=None,
        description="Deprecated alias for phone_data.",
    )
    mpesa_features: MpesaFeatures | None = None
    sacco_features: SaccoFeatures | None = None
    bank_features: BankFeatures | None = None
    mobile_lender_features: MobileLenderFeatures | None = None
    include_shap: bool = True
    persist_audit_trail: bool = True

    @model_validator(mode="after")
    def validate_channel_features(self) -> "ScoreRequest":
        if self.channel in {Channel.UNBANKED, Channel.MPESA} and self.mpesa_features is None:
            raise ValueError("mpesa_features are required for unbanked/mpesa channels.")
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
        features.update(self.lending_history.model_dump())
        features.update(self.data_sources.model_dump())
        phone_payload = self.alternative_data if self.alternative_data is not None else self.phone_data
        features.update(phone_payload.flatten())
        if features.get("has_phone_consent", 0) >= 0.5:
            features["alternative_data_consent"] = max(
                features.get("alternative_data_consent", 0),
                features["has_phone_consent"],
            )
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


class LoanLimitResponse(BaseModel):
    approved_limit_kes: float
    min_limit_kes: float
    max_limit_kes: float
    prior_limit_kes: float
    requested_limit_kes: float
    adjustment: str
    adjustment_pct: float
    tier: str
    reasons: list[str]
    next_review_days: int


class ScoreResponse(BaseModel):
    applicant_id: str
    channel: Channel
    probability_of_default: float
    credit_score: int
    decision: str
    policy: PolicyResponse
    loan_limit: LoanLimitResponse
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
