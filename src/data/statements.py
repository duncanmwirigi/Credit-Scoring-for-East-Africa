from __future__ import annotations

"""Derive credit features from customer-provided financial statements.

Third-party lenders (Tala, Branch, Zenka, etc.) do not share internal APIs.
This module parses uploaded M-Pesa, bank, and SACCO statements into model features.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

# Known digital-lender paybill / till keywords seen on M-Pesa statements (illustrative).
_DIGITAL_LENDER_KEYWORDS = (
    "tala",
    "branch",
    "zenka",
    "okash",
    "mshwari loan",
    "fuliza",
    "timiza",
    "kcb mpesa",
    "equity eazzy",
)

_ROLLOVER_KEYWORDS = ("rollover", "extension", "renewal fee", "late fee", "penalty")


@dataclass(frozen=True)
class MpesaStatementLine:
    """One parsed row from an M-Pesa mini-statement export."""

    date: datetime
    amount_kes: float
    direction: str  # "in" | "out"
    counterparty: str
    description: str = ""


def _matches_lender(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _DIGITAL_LENDER_KEYWORDS)


def _matches_rollover(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _ROLLOVER_KEYWORDS)


def derive_mpesa_statement_features(
    lines: Iterable[MpesaStatementLine],
    *,
    lookback_days: int = 365,
) -> dict[str, float]:
    """Build mobile-lender channel features from an M-Pesa statement only."""
    rows = list(lines)
    if not rows:
        return {
            "mpesa_statement_days_covered": 0.0,
            "mpesa_lender_disbursement_count_12m": 0.0,
            "mpesa_lender_repayment_count_12m": 0.0,
            "mpesa_inferred_repayment_rate": 1.0,
            "mpesa_active_lender_count": 0.0,
            "mpesa_avg_inferred_loan_kes": 0.0,
            "mpesa_late_repayment_events_12m": 0.0,
            "mpesa_loan_rollover_signals_12m": 0.0,
            "mpesa_net_cashflow_kes_90d": 0.0,
        }

    latest = max(row.date for row in rows)
    earliest = min(row.date for row in rows)
    days_covered = max(1, (latest - earliest).days + 1)

    disbursements: list[float] = []
    repayments: list[float] = []
    active_lenders: set[str] = set()
    late_events = 0
    rollover_signals = 0
    inflow_90d = 0.0
    outflow_90d = 0.0

    for row in rows:
        age_days = (latest - row.date).days
        if age_days > lookback_days:
            continue

        text = f"{row.counterparty} {row.description}"
        is_lender = _matches_lender(text)

        if age_days <= 90:
            if row.direction == "in":
                inflow_90d += row.amount_kes
            else:
                outflow_90d += row.amount_kes

        if not is_lender:
            continue

        if row.direction == "in" and age_days <= 365:
            disbursements.append(row.amount_kes)
            active_lenders.add(row.counterparty.lower())
        elif row.direction == "out" and age_days <= 365:
            repayments.append(row.amount_kes)
            active_lenders.add(row.counterparty.lower())

        if _matches_rollover(text):
            rollover_signals += 1
        if "late" in text.lower() or "overdue" in text.lower():
            late_events += 1

    inferred_rate = 1.0
    if disbursements and repayments:
        # Proxy: more repayments relative to disbursements suggests better behaviour.
        inferred_rate = min(1.0, len(repayments) / max(len(disbursements), 1))

    return {
        "mpesa_statement_days_covered": float(days_covered),
        "mpesa_lender_disbursement_count_12m": float(len(disbursements)),
        "mpesa_lender_repayment_count_12m": float(len(repayments)),
        "mpesa_inferred_repayment_rate": inferred_rate,
        "mpesa_active_lender_count": float(len(active_lenders)),
        "mpesa_avg_inferred_loan_kes": float(sum(disbursements) / max(len(disbursements), 1)),
        "mpesa_late_repayment_events_12m": float(late_events),
        "mpesa_loan_rollover_signals_12m": float(rollover_signals),
        "mpesa_net_cashflow_kes_90d": inflow_90d - outflow_90d,
    }
