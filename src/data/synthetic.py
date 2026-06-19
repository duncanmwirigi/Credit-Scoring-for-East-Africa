from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain import Channel
from src.features.engineering import (
    BANK_FEATURES,
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


def _good_profile(channel: str, rng: np.random.Generator) -> dict[str, float]:
    base = {
        "age": rng.integers(24, 55),
        "monthly_income_kes": rng.uniform(35_000, 180_000),
        "requested_amount_kes": rng.uniform(20_000, 250_000),
        "existing_debt_kes": rng.uniform(0, 80_000),
        "crb_defaults": 0,
        "crb_inquiries_6m": rng.integers(0, 2),
    }

    if channel == Channel.MPESA.value:
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
    base = {
        "age": rng.integers(18, 30),
        "monthly_income_kes": rng.uniform(12_000, 35_000),
        "requested_amount_kes": rng.uniform(80_000, 400_000),
        "existing_debt_kes": rng.uniform(60_000, 220_000),
        "crb_defaults": rng.integers(1, 3),
        "crb_inquiries_6m": rng.integers(3, 8),
    }

    if channel == Channel.MPESA.value:
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
    for feature in MPESA_FEATURES + SACCO_FEATURES + BANK_FEATURES:
        row.setdefault(feature, 0.0)

    if channel != Channel.MPESA.value:
        for feature in MPESA_FEATURES:
            row[feature] = 0.0
    if channel != Channel.SACCO.value:
        for feature in SACCO_FEATURES:
            row[feature] = 0.0
    if channel != Channel.BANK.value:
        for feature in BANK_FEATURES:
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

    records: list[dict[str, float | str | int]] = []
    for idx, (channel, is_default) in enumerate(zip(channels, default_flags)):
        # Inject ~8% label noise to mimic real-world overlap.
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
        for key in MPESA_FEATURES + SACCO_FEATURES + BANK_FEATURES:
            record[key] = profile[key]
        records.append(record)

    frame = pd.DataFrame(records)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
