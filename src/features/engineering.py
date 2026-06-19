from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.domain import Channel

COMMON_FEATURES = [
    "age",
    "monthly_income_kes",
    "requested_amount_kes",
    "existing_debt_kes",
    "debt_to_income",
    "loan_to_income",
    "crb_defaults",
    "crb_inquiries_6m",
]

RAW_APPLICANT_FEATURES = [
    "age",
    "monthly_income_kes",
    "requested_amount_kes",
    "existing_debt_kes",
    "crb_defaults",
    "crb_inquiries_6m",
]

MPESA_FEATURES = [
    "kyc_tier",
    "wallet_activity_days_90d",
    "avg_monthly_txn_count",
    "avg_txn_amount_kes",
    "cash_in_out_ratio",
    "merchant_spend_ratio",
    "fuliza_utilization",
    "wallet_balance_volatility",
    "days_since_last_txn",
]

SACCO_FEATURES = [
    "membership_months",
    "share_capital_kes",
    "monthly_savings_kes",
    "savings_consistency_score",
    "prior_loan_repayment_rate",
    "guarantor_count",
    "guarantor_avg_score",
    "dividend_years",
]

BANK_FEATURES = [
    "account_age_months",
    "avg_monthly_balance_kes",
    "salary_deposit_regularity",
    "bounced_cheques_12m",
    "overdraft_usage_ratio",
    "credit_card_utilization",
    "existing_loan_count",
    "branch_relationship_score",
]

MOBILE_LENDER_FEATURES = [
    "platform_tenure_months",
    "prior_loans_on_platform",
    "platform_repayment_rate",
    "days_since_last_repayment",
    "active_digital_loans_count",
    "avg_historical_loan_kes",
    "rollover_count_12m",
    "app_engagement_score",
    "mpesa_disbursement_linked",
    "alternative_data_score",
]

ALL_CHANNEL_FEATURES = (
    MPESA_FEATURES + SACCO_FEATURES + BANK_FEATURES + MOBILE_LENDER_FEATURES
)

FEATURE_COLUMNS = COMMON_FEATURES + ALL_CHANNEL_FEATURES

_CHANNEL_FEATURE_MAP: dict[Channel, list[str]] = {
    Channel.MPESA: MPESA_FEATURES,
    Channel.SACCO: SACCO_FEATURES,
    Channel.BANK: BANK_FEATURES,
    Channel.MOBILE_LENDER: MOBILE_LENDER_FEATURES,
}


def active_features_for_channel(channel: Channel) -> frozenset[str]:
    """Common + channel-specific columns that may appear in explanations."""
    return frozenset(COMMON_FEATURES + _CHANNEL_FEATURE_MAP[channel])


def enrich_common_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["debt_to_income"] = enriched["existing_debt_kes"] / enriched[
        "monthly_income_kes"
    ].clip(lower=1)
    enriched["loan_to_income"] = enriched["requested_amount_kes"] / enriched[
        "monthly_income_kes"
    ].clip(lower=1)
    return enriched


def mask_channel_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Zero out channel-specific columns that do not apply to each row."""
    masked = frame.copy()
    channel = masked["channel"]

    for feature in MPESA_FEATURES:
        masked.loc[channel != Channel.MPESA.value, feature] = 0.0
    for feature in SACCO_FEATURES:
        masked.loc[channel != Channel.SACCO.value, feature] = 0.0
    for feature in BANK_FEATURES:
        masked.loc[channel != Channel.BANK.value, feature] = 0.0
    for feature in MOBILE_LENDER_FEATURES:
        masked.loc[channel != Channel.MOBILE_LENDER.value, feature] = 0.0

    return masked


def build_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = enrich_common_features(frame)
    masked = mask_channel_features(enriched)
    return masked[FEATURE_COLUMNS]


def applicant_to_frame(applicants: Iterable) -> pd.DataFrame:
    rows = []
    for applicant in applicants:
        row = {
            "applicant_id": applicant.applicant_id,
            "channel": applicant.channel.value,
            "age": applicant.age,
            "monthly_income_kes": applicant.monthly_income_kes,
            "requested_amount_kes": applicant.requested_amount_kes,
            "existing_debt_kes": applicant.existing_debt_kes,
            "crb_defaults": applicant.crb_defaults,
            "crb_inquiries_6m": applicant.crb_inquiries_6m,
        }
        for feature in ALL_CHANNEL_FEATURES:
            row[feature] = applicant.features.get(feature, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)
