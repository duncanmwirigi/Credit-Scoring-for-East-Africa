from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain import Channel
from src.features.engineering import (
    ALTERNATIVE_DATA_FEATURES,
    BANK_FEATURES,
    DATA_SOURCE_FEATURES,
    LENDING_HISTORY_FEATURES,
    MOBILE_LENDER_FEATURES,
    MPESA_FEATURES,
    RAW_APPLICANT_FEATURES,
    SACCO_FEATURES,
)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _assign_channels(n_samples: int, distribution: dict[str, float], seed: int) -> np.ndarray:
    channels = list(distribution.keys())
    weights = np.array([distribution[c] for c in channels], dtype=float)
    weights /= weights.sum()
    return _rng(seed).choice(channels, size=n_samples, p=weights)


def _lending_history_good(rng: np.random.Generator) -> dict[str, float]:
    loans = rng.integers(3, 18)
    repaid = int(loans * rng.uniform(0.88, 1.0))
    return {
        "lifetime_loans_count": loans,
        "lifetime_loans_repaid_on_time": repaid,
        "lifetime_default_count": rng.integers(0, 1),
        "lifetime_repayment_rate": repaid / max(loans, 1),
        "on_time_repayment_streak": rng.integers(2, 10),
        "avg_days_past_due": rng.uniform(0, 2),
        "days_since_last_loan": rng.integers(5, 120),
        "days_since_last_default": rng.integers(400, 1200),
        "current_outstanding_kes": rng.uniform(0, 25_000),
        "highest_prior_limit_kes": rng.uniform(8_000, 80_000),
        "months_customer_relationship": rng.integers(6, 48),
    }


def _lending_history_risk(rng: np.random.Generator) -> dict[str, float]:
    loans = rng.integers(1, 8)
    repaid = int(loans * rng.uniform(0.25, 0.65))
    return {
        "lifetime_loans_count": loans,
        "lifetime_loans_repaid_on_time": repaid,
        "lifetime_default_count": rng.integers(1, 4),
        "lifetime_repayment_rate": repaid / max(loans, 1),
        "on_time_repayment_streak": 0,
        "avg_days_past_due": rng.uniform(8, 45),
        "days_since_last_loan": rng.integers(1, 60),
        "days_since_last_default": rng.integers(10, 200),
        "current_outstanding_kes": rng.uniform(20_000, 120_000),
        "highest_prior_limit_kes": rng.uniform(3_000, 20_000),
        "months_customer_relationship": rng.integers(1, 8),
    }


def _alternative_data_good(rng: np.random.Generator, income: float) -> dict[str, float]:
    sms_income = income * rng.uniform(0.9, 1.1)
    return {
        "alternative_data_consent": 1.0,
        "sms_salary_detected": 1.0,
        "sms_inferred_monthly_income_kes": sms_income,
        "sms_mpesa_txn_count_30d": rng.uniform(25, 120),
        "sms_total_count_30d": rng.uniform(40, 180),
        "sms_bill_pay_regularity": rng.uniform(0.7, 1.0),
        "sms_other_lender_repayment_count": rng.integers(0, 2),
        "sms_collection_message_count_30d": rng.integers(0, 1),
        "sms_lender_promo_count_30d": rng.integers(0, 2),
        "sms_gambling_ratio": rng.uniform(0.0, 0.08),
        "income_declared_vs_sms_ratio": sms_income / max(income, 1),
        "call_total_count_30d": rng.uniform(20, 90),
        "call_unique_contacts_30d": rng.uniform(12, 45),
        "call_avg_duration_seconds": rng.uniform(45, 240),
        "call_incoming_ratio": rng.uniform(0.45, 0.75),
        "call_missed_ratio": rng.uniform(0.05, 0.2),
        "call_collection_agency_count_30d": 0.0,
        "call_night_activity_ratio": rng.uniform(0.02, 0.12),
        "device_tenure_days": rng.integers(120, 900),
        "contacts_count": rng.uniform(80, 350),
        "contacts_saved_ratio": rng.uniform(0.65, 0.95),
        "apps_lending_app_count": rng.integers(0, 2),
        "apps_gambling_app_count": rng.integers(0, 1),
        "device_os_android": 1.0,
        "device_os_version_score": rng.uniform(0.6, 1.0),
        "device_tier": rng.integers(2, 4),
        "device_ram_gb": rng.uniform(3, 8),
        "device_storage_free_ratio": rng.uniform(0.25, 0.8),
        "device_dual_sim": 1.0,
        "device_network_4g_plus": 1.0,
        "device_model_age_months": rng.uniform(6, 36),
    }


def _alternative_data_risk(rng: np.random.Generator, income: float) -> dict[str, float]:
    sms_income = income * rng.uniform(0.4, 0.75)
    return {
        "alternative_data_consent": 1.0,
        "sms_salary_detected": rng.choice([0.0, 1.0]),
        "sms_inferred_monthly_income_kes": sms_income,
        "sms_mpesa_txn_count_30d": rng.uniform(2, 18),
        "sms_total_count_30d": rng.uniform(5, 35),
        "sms_bill_pay_regularity": rng.uniform(0.05, 0.35),
        "sms_other_lender_repayment_count": rng.integers(3, 10),
        "sms_collection_message_count_30d": rng.integers(4, 14),
        "sms_lender_promo_count_30d": rng.integers(5, 18),
        "sms_gambling_ratio": rng.uniform(0.2, 0.55),
        "income_declared_vs_sms_ratio": sms_income / max(income, 1),
        "call_total_count_30d": rng.uniform(3, 25),
        "call_unique_contacts_30d": rng.uniform(2, 12),
        "call_avg_duration_seconds": rng.uniform(5, 40),
        "call_incoming_ratio": rng.uniform(0.15, 0.45),
        "call_missed_ratio": rng.uniform(0.35, 0.75),
        "call_collection_agency_count_30d": rng.uniform(2, 8),
        "call_night_activity_ratio": rng.uniform(0.25, 0.55),
        "device_tenure_days": rng.integers(5, 60),
        "contacts_count": rng.uniform(8, 45),
        "contacts_saved_ratio": rng.uniform(0.1, 0.4),
        "apps_lending_app_count": rng.integers(3, 7),
        "apps_gambling_app_count": rng.integers(1, 4),
        "device_os_android": rng.choice([1.0, 0.0]),
        "device_os_version_score": rng.uniform(0.1, 0.45),
        "device_tier": rng.integers(1, 2),
        "device_ram_gb": rng.uniform(1, 3),
        "device_storage_free_ratio": rng.uniform(0.02, 0.2),
        "device_dual_sim": rng.choice([1.0, 0.0]),
        "device_network_4g_plus": rng.choice([1.0, 0.0]),
        "device_model_age_months": rng.uniform(48, 96),
    }


def _data_sources_for_channel(channel: str, rng: np.random.Generator, *, good: bool) -> dict[str, float]:
    has_bank = channel == Channel.BANK.value or (good and channel == Channel.UNBANKED.value and rng.random() < 0.15)
    has_sacco = channel == Channel.SACCO.value or (good and rng.random() < 0.2)
    return {
        "has_mpesa_wallet": 1.0 if channel in {Channel.MPESA.value, Channel.UNBANKED.value} else float(rng.random() < 0.7),
        "has_phone_consent": 1.0,
        "has_bank_account": 1.0 if has_bank else 0.0,
        "has_sacco_membership": 1.0 if has_sacco else 0.0,
        "has_crb_record": 1.0 if channel == Channel.BANK.value or rng.random() < 0.35 else 0.0,
    }


def _mobile_lender_good(rng: np.random.Generator) -> dict[str, float]:
    disbursements = rng.integers(3, 14)
    return {
        "mpesa_statement_days_covered": rng.integers(90, 365),
        "mpesa_lender_disbursement_count_12m": disbursements,
        "mpesa_lender_repayment_count_12m": rng.integers(disbursements, disbursements + 4),
        "mpesa_inferred_repayment_rate": rng.uniform(0.90, 1.0),
        "mpesa_active_lender_count": rng.integers(0, 2),
        "mpesa_avg_inferred_loan_kes": rng.uniform(5_000, 45_000),
        "mpesa_late_repayment_events_12m": rng.integers(0, 2),
        "mpesa_loan_rollover_signals_12m": rng.integers(0, 2),
        "mpesa_net_cashflow_kes_90d": rng.uniform(5_000, 80_000),
    }


def _mobile_lender_risk(rng: np.random.Generator) -> dict[str, float]:
    disbursements = rng.integers(2, 8)
    return {
        "mpesa_statement_days_covered": rng.integers(15, 75),
        "mpesa_lender_disbursement_count_12m": disbursements,
        "mpesa_lender_repayment_count_12m": rng.integers(0, max(disbursements, 1)),
        "mpesa_inferred_repayment_rate": rng.uniform(0.25, 0.72),
        "mpesa_active_lender_count": rng.integers(3, 7),
        "mpesa_avg_inferred_loan_kes": rng.uniform(1_000, 8_000),
        "mpesa_late_repayment_events_12m": rng.integers(4, 12),
        "mpesa_loan_rollover_signals_12m": rng.integers(4, 10),
        "mpesa_net_cashflow_kes_90d": rng.uniform(-40_000, 5_000),
    }


def _good_profile(channel: str, rng: np.random.Generator) -> dict[str, float]:
    income = rng.uniform(35_000, 180_000)
    base = {
        "age": rng.integers(24, 55),
        "monthly_income_kes": income,
        "requested_amount_kes": rng.uniform(20_000, 250_000),
        "existing_debt_kes": rng.uniform(0, 80_000),
        "crb_defaults": 0,
        "crb_inquiries_6m": rng.integers(0, 2),
    }
    base.update(_lending_history_good(rng))
    base.update(_alternative_data_good(rng, income))
    base.update(_data_sources_for_channel(channel, rng, good=True))

    if channel in {Channel.MPESA.value, Channel.UNBANKED.value}:
        base.update(
            {
                "kyc_tier": rng.integers(2, 4),
                "wallet_activity_days_90d": rng.integers(45, 90),
                "avg_monthly_txn_count": rng.uniform(25, 120),
                "avg_txn_amount_kes": rng.uniform(800, 8_000),
                "cash_in_out_ratio": rng.uniform(0.7, 1.3),
                "merchant_spend_ratio": rng.uniform(0.15, 0.45),
                "fuliza_utilization": rng.uniform(0.0, 0.35),
                "wallet_balance_volatility": rng.uniform(0.05, 0.25),
                "days_since_last_txn": rng.integers(0, 7),
            }
        )
    elif channel == Channel.SACCO.value:
        base.update(
            {
                "membership_months": rng.integers(12, 96),
                "share_capital_kes": rng.uniform(15_000, 120_000),
                "monthly_savings_kes": rng.uniform(3_000, 25_000),
                "savings_consistency_score": rng.uniform(0.75, 1.0),
                "prior_loan_repayment_rate": rng.uniform(0.92, 1.0),
                "guarantor_count": rng.integers(2, 5),
                "guarantor_avg_score": rng.uniform(650, 820),
                "dividend_years": rng.integers(1, 8),
            }
        )
    elif channel == Channel.MOBILE_LENDER.value:
        base.update(_mobile_lender_good(rng))
        base["requested_amount_kes"] = rng.uniform(2_000, 50_000)
        base["monthly_income_kes"] = rng.uniform(18_000, 85_000)
    else:
        base.update(
            {
                "account_age_months": rng.integers(18, 120),
                "avg_monthly_balance_kes": rng.uniform(25_000, 350_000),
                "salary_deposit_regularity": rng.uniform(0.8, 1.0),
                "bounced_cheques_12m": 0,
                "overdraft_usage_ratio": rng.uniform(0.0, 0.25),
                "credit_card_utilization": rng.uniform(0.05, 0.35),
                "existing_loan_count": rng.integers(0, 2),
                "branch_relationship_score": rng.uniform(0.6, 1.0),
            }
        )
    return base


def _risk_profile(channel: str, rng: np.random.Generator) -> dict[str, float]:
    income = rng.uniform(12_000, 35_000)
    base = {
        "age": rng.integers(18, 30),
        "monthly_income_kes": income,
        "requested_amount_kes": rng.uniform(80_000, 400_000),
        "existing_debt_kes": rng.uniform(60_000, 220_000),
        "crb_defaults": rng.integers(1, 3),
        "crb_inquiries_6m": rng.integers(3, 8),
    }
    base.update(_lending_history_risk(rng))
    base.update(_alternative_data_risk(rng, income))
    base.update(_data_sources_for_channel(channel, rng, good=False))

    if channel in {Channel.MPESA.value, Channel.UNBANKED.value}:
        base.update(
            {
                "kyc_tier": rng.integers(1, 2),
                "wallet_activity_days_90d": rng.integers(5, 25),
                "avg_monthly_txn_count": rng.uniform(3, 18),
                "avg_txn_amount_kes": rng.uniform(150, 900),
                "cash_in_out_ratio": rng.uniform(2.5, 6.0),
                "merchant_spend_ratio": rng.uniform(0.0, 0.08),
                "fuliza_utilization": rng.uniform(0.65, 1.0),
                "wallet_balance_volatility": rng.uniform(0.45, 0.95),
                "days_since_last_txn": rng.integers(14, 60),
            }
        )
    elif channel == Channel.SACCO.value:
        base.update(
            {
                "membership_months": rng.integers(1, 8),
                "share_capital_kes": rng.uniform(1_000, 8_000),
                "monthly_savings_kes": rng.uniform(200, 2_500),
                "savings_consistency_score": rng.uniform(0.15, 0.55),
                "prior_loan_repayment_rate": rng.uniform(0.35, 0.75),
                "guarantor_count": rng.integers(0, 2),
                "guarantor_avg_score": rng.uniform(420, 580),
                "dividend_years": 0,
            }
        )
    elif channel == Channel.MOBILE_LENDER.value:
        base.update(_mobile_lender_risk(rng))
        base["requested_amount_kes"] = rng.uniform(5_000, 30_000)
        base["monthly_income_kes"] = rng.uniform(8_000, 22_000)
        base["existing_debt_kes"] = rng.uniform(15_000, 60_000)
    else:
        base.update(
            {
                "account_age_months": rng.integers(1, 10),
                "avg_monthly_balance_kes": rng.uniform(500, 12_000),
                "salary_deposit_regularity": rng.uniform(0.1, 0.45),
                "bounced_cheques_12m": rng.integers(2, 6),
                "overdraft_usage_ratio": rng.uniform(0.55, 1.0),
                "credit_card_utilization": rng.uniform(0.65, 1.0),
                "existing_loan_count": rng.integers(2, 5),
                "branch_relationship_score": rng.uniform(0.05, 0.35),
            }
        )
    return base


def _zero_fill_channel_features(row: dict[str, float], channel: str) -> dict[str, float]:
    all_features = (
        LENDING_HISTORY_FEATURES
        + DATA_SOURCE_FEATURES
        + ALTERNATIVE_DATA_FEATURES
        + MPESA_FEATURES
        + SACCO_FEATURES
        + BANK_FEATURES
        + MOBILE_LENDER_FEATURES
    )
    for feature in all_features:
        row.setdefault(feature, 0.0)

    if channel not in {Channel.MPESA.value, Channel.UNBANKED.value}:
        for feature in MPESA_FEATURES:
            row[feature] = 0.0
    if channel != Channel.SACCO.value:
        for feature in SACCO_FEATURES:
            row[feature] = 0.0
    if channel != Channel.BANK.value:
        for feature in BANK_FEATURES:
            row[feature] = 0.0
    if channel != Channel.MOBILE_LENDER.value:
        for feature in MOBILE_LENDER_FEATURES:
            row[feature] = 0.0
    return row


def generate_synthetic_portfolio(
    n_samples: int,
    default_rate: float,
    channel_distribution: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    rng = _rng(seed)
    channels = _assign_channels(n_samples, channel_distribution, seed)
    n_defaults = int(round(n_samples * default_rate))
    default_flags = np.zeros(n_samples, dtype=int)
    default_flags[:n_defaults] = 1
    rng.shuffle(default_flags)

    all_features = (
        LENDING_HISTORY_FEATURES
        + DATA_SOURCE_FEATURES
        + ALTERNATIVE_DATA_FEATURES
        + MPESA_FEATURES
        + SACCO_FEATURES
        + BANK_FEATURES
        + MOBILE_LENDER_FEATURES
    )
    records: list[dict[str, float | str | int]] = []
    for idx, (channel, is_default) in enumerate(zip(channels, default_flags)):
        flip_profile = rng.random() < 0.08
        profile_is_default = is_default if not flip_profile else not is_default
        profile = _risk_profile(channel, rng) if profile_is_default else _good_profile(channel, rng)
        profile = _zero_fill_channel_features(profile, channel)
        record: dict[str, float | str | int] = {
            "applicant_id": f"APP-{seed}-{idx:05d}",
            "channel": channel,
            "default_12m": int(is_default),
        }
        for key in RAW_APPLICANT_FEATURES:
            record[key] = profile[key]
        for key in all_features:
            record[key] = profile[key]
        records.append(record)

    frame = pd.DataFrame(records)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
