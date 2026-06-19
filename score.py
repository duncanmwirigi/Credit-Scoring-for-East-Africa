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
            applicant_id="MPESA-RISK-77",
            channel=Channel.MPESA,
            age=22,
            monthly_income_kes=18_000,
            requested_amount_kes=120_000,
            existing_debt_kes=55_000,
            crb_defaults=1,
            crb_inquiries_6m=5,
            features={
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
    decisions = scorer.score_batch(sample_applicants())

    payload = []
    for decision in decisions:
        payload.append(
            {
                "applicant_id": decision.applicant_id,
                "channel": decision.channel.value,
                "probability_of_default": decision.probability_of_default,
                "credit_score": decision.credit_score,
                "decision": decision.decision.value,
                "policy_passed": decision.policy.passed,
                "policy_reasons": list(decision.policy.reasons),
                "top_risk_factors": decision.top_risk_factors,
                "metadata": decision.metadata,
            }
        )
        logger.info(
            "%s | channel=%s | score=%s | pd=%.2f%% | decision=%s",
            decision.applicant_id,
            decision.channel.value,
            decision.credit_score,
            decision.probability_of_default * 100,
            decision.decision.value.upper(),
        )

    output_path = config.reports_dir / "sample_decisions.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved sample decisions to %s", output_path)


if __name__ == "__main__":
    main()
