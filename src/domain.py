from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Channel(str, Enum):
    MPESA = "mpesa"
    SACCO = "sacco"
    BANK = "bank"
    MOBILE_LENDER = "mobile_lender"  # Tala, Branch, Zenka, Okash, etc.


class Decision(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"


@dataclass(frozen=True)
class ApplicantProfile:
    applicant_id: str
    channel: Channel
    age: int
    monthly_income_kes: float
    requested_amount_kes: float
    existing_debt_kes: float
    crb_defaults: int
    crb_inquiries_6m: int
    features: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyOutcome:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass
class CreditDecision:
    applicant_id: str
    channel: Channel
    probability_of_default: float
    credit_score: int
    decision: Decision
    policy: PolicyOutcome
    top_risk_factors: list[tuple[str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)
