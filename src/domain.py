from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Channel(str, Enum):
    MPESA = "mpesa"
    SACCO = "sacco"
    BANK = "bank"
    MOBILE_LENDER = "mobile_lender"


class Decision(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"


class LimitAdjustment(str, Enum):
    FIRST_TIME = "first_time"
    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"
    SUSPENDED = "suspended"


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


@dataclass(frozen=True)
class LoanLimitAssignment:
    approved_limit_kes: float
    min_limit_kes: float
    max_limit_kes: float
    prior_limit_kes: float
    requested_limit_kes: float
    adjustment: LimitAdjustment
    adjustment_pct: float
    tier: str
    reasons: tuple[str, ...]
    next_review_days: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved_limit_kes": round(self.approved_limit_kes, 2),
            "min_limit_kes": self.min_limit_kes,
            "max_limit_kes": self.max_limit_kes,
            "prior_limit_kes": self.prior_limit_kes,
            "requested_limit_kes": self.requested_limit_kes,
            "adjustment": self.adjustment.value,
            "adjustment_pct": round(self.adjustment_pct, 4),
            "tier": self.tier,
            "reasons": list(self.reasons),
            "next_review_days": self.next_review_days,
        }


@dataclass
class CreditDecision:
    applicant_id: str
    channel: Channel
    probability_of_default: float
    credit_score: int
    decision: Decision
    policy: PolicyOutcome
    top_risk_factors: list[tuple[str, float]]
    loan_limit: LoanLimitAssignment | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
