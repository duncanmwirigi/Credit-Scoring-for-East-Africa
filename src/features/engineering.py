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

LENDING_HISTORY_FEATURES = [
    # Own-institution loan ledger only — not shared by other banks/lenders/SACCOs.
    "lifetime_loans_count",
    "lifetime_loans_repaid_on_time",
    "lifetime_default_count",
    "lifetime_repayment_rate",
    "on_time_repayment_streak",
    "avg_days_past_due",
    "days_since_last_loan",
    "days_since_last_default",
    "current_outstanding_kes",
    "highest_prior_limit_kes",
    "months_customer_relationship",
]

# Which data sources were actually available for this applicant (unbanked-first design).
DATA_SOURCE_FEATURES = [
    "has_mpesa_wallet",
    "has_phone_consent",
    "has_bank_account",
    "has_sacco_membership",
    "has_crb_record",
]

# Phone permissions data — SMS, call log, contacts, apps, device tech (consent-gated).
PHONE_DATA_FEATURES = [
    "alternative_data_consent",
    "sms_salary_detected",
    "sms_inferred_monthly_income_kes",
    "sms_mpesa_txn_count_30d",
    "sms_total_count_30d",
    "sms_bill_pay_regularity",
    "sms_other_lender_repayment_count",
    "sms_collection_message_count_30d",
    "sms_lender_promo_count_30d",
    "sms_gambling_ratio",
    "income_declared_vs_sms_ratio",
    "call_total_count_30d",
    "call_unique_contacts_30d",
    "call_avg_duration_seconds",
    "call_incoming_ratio",
    "call_missed_ratio",
    "call_collection_agency_count_30d",
    "call_night_activity_ratio",
    "device_tenure_days",
    "contacts_count",
    "contacts_saved_ratio",
    "apps_lending_app_count",
    "apps_gambling_app_count",
    "device_os_android",
    "device_os_version_score",
    "device_tier",
    "device_ram_gb",
    "device_storage_free_ratio",
    "device_dual_sim",
    "device_network_4g_plus",
    "device_model_age_months",
]

# Backward-compatible alias used across the codebase.
ALTERNATIVE_DATA_FEATURES = PHONE_DATA_FEATURES

MPESA_WALLET_CHANNELS = frozenset({Channel.MPESA.value, Channel.UNBANKED.value})

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

# Inferred from customer M-Pesa statement — no third-party lender API access.
MOBILE_LENDER_FEATURES = [
    "mpesa_statement_days_covered",
    "mpesa_lender_disbursement_count_12m",
    "mpesa_lender_repayment_count_12m",
    "mpesa_inferred_repayment_rate",
    "mpesa_active_lender_count",
    "mpesa_avg_inferred_loan_kes",
    "mpesa_late_repayment_events_12m",
    "mpesa_loan_rollover_signals_12m",
    "mpesa_net_cashflow_kes_90d",
]

ALL_CHANNEL_FEATURES = (
    MPESA_FEATURES + SACCO_FEATURES + BANK_FEATURES + MOBILE_LENDER_FEATURES
)

SHARED_BEHAVIOR_FEATURES = (
    LENDING_HISTORY_FEATURES + DATA_SOURCE_FEATURES + PHONE_DATA_FEATURES
)

FEATURE_COLUMNS = COMMON_FEATURES + SHARED_BEHAVIOR_FEATURES + ALL_CHANNEL_FEATURES

_CHANNEL_FEATURE_MAP: dict[Channel, list[str]] = {
    Channel.UNBANKED: MPESA_FEATURES,
    Channel.MPESA: MPESA_FEATURES,
    Channel.SACCO: SACCO_FEATURES,
    Channel.BANK: BANK_FEATURES,
    Channel.MOBILE_LENDER: MOBILE_LENDER_FEATURES,
}

DEFAULT_FEATURE_VALUES: dict[str, float] = {
    **{feature: 0.0 for feature in LENDING_HISTORY_FEATURES},
    **{feature: 0.0 for feature in DATA_SOURCE_FEATURES},
    **{feature: 0.0 for feature in PHONE_DATA_FEATURES},
    "lifetime_repayment_rate": 1.0,
    "income_declared_vs_sms_ratio": 1.0,
    "call_incoming_ratio": 0.5,
}


def active_features_for_channel(channel: Channel) -> frozenset[str]:
    """Features that may appear in explanations for a channel."""
    return frozenset(COMMON_FEATURES + SHARED_BEHAVIOR_FEATURES + _CHANNEL_FEATURE_MAP[channel])


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
    """Mask channel blocks; bank/SACCO also activate when optional data is present."""
    masked = frame.copy()
    channel = masked["channel"]

    for feature in MPESA_FEATURES:
        masked.loc[~channel.isin(MPESA_WALLET_CHANNELS), feature] = 0.0

    has_bank = (channel == Channel.BANK.value) | (masked["has_bank_account"] >= 0.5)
    for feature in BANK_FEATURES:
        masked.loc[~has_bank, feature] = 0.0

    has_sacco = (channel == Channel.SACCO.value) | (masked["has_sacco_membership"] >= 0.5)
    for feature in SACCO_FEATURES:
        masked.loc[~has_sacco, feature] = 0.0

    for feature in MOBILE_LENDER_FEATURES:
        masked.loc[channel != Channel.MOBILE_LENDER.value, feature] = 0.0

    return masked


def mask_alternative_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Zero phone-derived features when the user has not granted consent."""
    masked = frame.copy()
    no_consent = masked["alternative_data_consent"] < 0.5
    for feature in PHONE_DATA_FEATURES:
        if feature == "alternative_data_consent":
            continue
        masked.loc[no_consent, feature] = 0.0
    masked.loc[no_consent, "income_declared_vs_sms_ratio"] = 1.0
    masked.loc[no_consent, "call_incoming_ratio"] = 0.5
    return masked


def build_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = enrich_common_features(frame)
    masked = mask_channel_features(enriched)
    masked = mask_alternative_data(masked)
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
        for feature, default in DEFAULT_FEATURE_VALUES.items():
            row[feature] = applicant.features.get(feature, default)
        for feature in ALL_CHANNEL_FEATURES:
            row[feature] = applicant.features.get(feature, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)
