#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.domain import ApplicantProfile, Channel
from src.ml.scorer import CreditScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("credit_scoring.score")


def _alt_data(income: float, *, good: bool) -> dict[str, float]:
    if good:
        sms_income = income * 0.98
        return {
            "alternative_data_consent": 1.0,
            "sms_salary_detected": 1.0,
            "sms_inferred_monthly_income_kes": sms_income,
            "sms_mpesa_txn_count_30d": 72,
            "sms_bill_pay_regularity": 0.91,
            "sms_other_lender_repayment_count": 1,
            "sms_gambling_ratio": 0.04,
            "apps_lending_app_count": 1,
            "apps_gambling_app_count": 0,
            "income_declared_vs_sms_ratio": sms_income / income,
            "device_tenure_days": 540,
        }
    return {
        "alternative_data_consent": 1.0,
        "sms_salary_detected": 0.0,
        "sms_inferred_monthly_income_kes": income * 0.55,
        "sms_mpesa_txn_count_30d": 9,
        "sms_bill_pay_regularity": 0.18,
        "sms_other_lender_repayment_count": 6,
        "sms_gambling_ratio": 0.38,
        "apps_lending_app_count": 5,
        "apps_gambling_app_count": 2,
        "income_declared_vs_sms_ratio": 0.55,
        "device_tenure_days": 40,
    }


def _history(*, good: bool, prior_limit: float = 0) -> dict[str, float]:
    if good:
        return {
            "lifetime_loans_count": 10,
            "lifetime_loans_repaid_on_time": 10,
            "lifetime_default_count": 0,
            "lifetime_repayment_rate": 1.0,
            "on_time_repayment_streak": 7,
            "avg_days_past_due": 0.5,
            "days_since_last_loan": 20,
            "days_since_last_default": 9999,
            "current_outstanding_kes": 8_000,
            "highest_prior_limit_kes": prior_limit or 35_000,
            "months_customer_relationship": 24,
        }
    return {
        "lifetime_loans_count": 4,
        "lifetime_loans_repaid_on_time": 2,
        "lifetime_default_count": 2,
        "lifetime_repayment_rate": 0.5,
        "on_time_repayment_streak": 0,
        "avg_days_past_due": 22,
        "days_since_last_loan": 15,
        "days_since_last_default": 45,
        "current_outstanding_kes": 42_000,
        "highest_prior_limit_kes": prior_limit or 12_000,
        "months_customer_relationship": 5,
    }


def sample_applicants() -> list[ApplicantProfile]:
    return [
        ApplicantProfile(
            applicant_id="MPESA-001",
            channel=Channel.MPESA,
            age=34,
            monthly_income_kes=85_000,
            requested_amount_kes=50_000,
            existing_debt_kes=5_000,
            crb_defaults=0,
            crb_inquiries_6m=0,
            features={
                **_history(good=True, prior_limit=40_000),
                **_alt_data(85_000, good=True),
                "kyc_tier": 3,
                "wallet_activity_days_90d": 88,
                "avg_monthly_txn_count": 95,
                "avg_txn_amount_kes": 4_200,
                "cash_in_out_ratio": 0.95,
                "merchant_spend_ratio": 0.35,
                "fuliza_utilization": 0.05,
                "wallet_balance_volatility": 0.08,
                "days_since_last_txn": 0,
            },
        ),
        ApplicantProfile(
            applicant_id="SACCO-014",
            channel=Channel.SACCO,
            age=42,
            monthly_income_kes=120_000,
            requested_amount_kes=180_000,
            existing_debt_kes=15_000,
            crb_defaults=0,
            crb_inquiries_6m=0,
            features={
                **_history(good=True, prior_limit=150_000),
                **_alt_data(120_000, good=True),
                "membership_months": 72,
                "share_capital_kes": 95_000,
                "monthly_savings_kes": 18_000,
                "savings_consistency_score": 0.97,
                "prior_loan_repayment_rate": 1.0,
                "guarantor_count": 4,
                "guarantor_avg_score": 760,
                "dividend_years": 6,
            },
        ),
        ApplicantProfile(
            applicant_id="BANK-203",
            channel=Channel.BANK,
            age=39,
            monthly_income_kes=210_000,
            requested_amount_kes=350_000,
            existing_debt_kes=25_000,
            crb_defaults=0,
            crb_inquiries_6m=0,
            features={
                **_history(good=True, prior_limit=300_000),
                **_alt_data(210_000, good=True),
                "account_age_months": 60,
                "avg_monthly_balance_kes": 280_000,
                "salary_deposit_regularity": 0.98,
                "bounced_cheques_12m": 0,
                "overdraft_usage_ratio": 0.03,
                "credit_card_utilization": 0.12,
                "existing_loan_count": 0,
                "branch_relationship_score": 0.91,
            },
        ),
        ApplicantProfile(
            applicant_id="TALA-001",
            channel=Channel.MOBILE_LENDER,
            age=31,
            monthly_income_kes=48_000,
            requested_amount_kes=15_000,
            existing_debt_kes=8_000,
            crb_defaults=0,
            crb_inquiries_6m=1,
            features={
                **_history(good=True, prior_limit=20_000),
                **_alt_data(48_000, good=True),
                "mpesa_statement_days_covered": 270,
                "mpesa_lender_disbursement_count_12m": 8,
                "mpesa_lender_repayment_count_12m": 9,
                "mpesa_inferred_repayment_rate": 0.96,
                "mpesa_active_lender_count": 1,
                "mpesa_avg_inferred_loan_kes": 12_000,
                "mpesa_late_repayment_events_12m": 0,
                "mpesa_loan_rollover_signals_12m": 1,
                "mpesa_net_cashflow_kes_90d": 18_500,
            },
        ),
        ApplicantProfile(
            applicant_id="TALA-RISK-99",
            channel=Channel.MOBILE_LENDER,
            age=23,
            monthly_income_kes=14_000,
            requested_amount_kes=20_000,
            existing_debt_kes=38_000,
            crb_defaults=1,
            crb_inquiries_6m=6,
            features={
                **_history(good=False, prior_limit=10_000),
                **_alt_data(14_000, good=False),
                "mpesa_statement_days_covered": 45,
                "mpesa_lender_disbursement_count_12m": 6,
                "mpesa_lender_repayment_count_12m": 2,
                "mpesa_inferred_repayment_rate": 0.45,
                "mpesa_active_lender_count": 5,
                "mpesa_avg_inferred_loan_kes": 4_500,
                "mpesa_late_repayment_events_12m": 8,
                "mpesa_loan_rollover_signals_12m": 7,
                "mpesa_net_cashflow_kes_90d": -12_000,
            },
        ),
        ApplicantProfile(
            applicant_id="MPESA-RISK-77",
            channel=Channel.MPESA,
            age=22,
            monthly_income_kes=18_000,
            requested_amount_kes=120_000,
            existing_debt_kes=55_000,
            crb_defaults=1,
            crb_inquiries_6m=5,
            features={
                **_history(good=False, prior_limit=8_000),
                **_alt_data(18_000, good=False),
                "kyc_tier": 1,
                "wallet_activity_days_90d": 11,
                "avg_monthly_txn_count": 8,
                "avg_txn_amount_kes": 450,
                "cash_in_out_ratio": 4.2,
                "merchant_spend_ratio": 0.03,
                "fuliza_utilization": 0.92,
                "wallet_balance_volatility": 0.78,
                "days_since_last_txn": 21,
            },
        ),
    ]


def main() -> None:
    config = load_config()
    scorer = CreditScorer.from_latest(config)

    payload = []
    for applicant in sample_applicants():
        decision, shap_explanation, audit_id = scorer.score_with_audit(
            applicant,
            include_shap=True,
            persist_audit_trail=True,
            request_snapshot={"source": "score.py"},
        )
        entry = {
            "applicant_id": decision.applicant_id,
            "channel": decision.channel.value,
            "probability_of_default": decision.probability_of_default,
            "credit_score": decision.credit_score,
            "decision": decision.decision.value,
            "policy_passed": decision.policy.passed,
            "policy_reasons": list(decision.policy.reasons),
            "loan_limit": decision.loan_limit.to_dict() if decision.loan_limit else {},
            "top_risk_factors": decision.top_risk_factors,
            "audit_id": audit_id,
            "metadata": decision.metadata,
        }
        if shap_explanation:
            entry["shap"] = shap_explanation.to_dict()
        payload.append(entry)
        limit = decision.loan_limit
        logger.info(
            "%s | channel=%s | score=%s | limit=KES %s | %s | decision=%s",
            decision.applicant_id,
            decision.channel.value,
            decision.credit_score,
            f"{limit.approved_limit_kes:,.0f}" if limit else "0",
            limit.adjustment.value.upper() if limit else "N/A",
            decision.decision.value.upper(),
        )

    output_path = config.reports_dir / "sample_decisions.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved sample decisions to %s", output_path)


if __name__ == "__main__":
    main()
