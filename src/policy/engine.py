from __future__ import annotations

from src.config import AppConfig
from src.domain import ApplicantProfile, Channel, PolicyOutcome


class PolicyEngine:
    """Deterministic business and regulatory rules applied before ML decisions."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def evaluate(self, applicant: ApplicantProfile) -> PolicyOutcome:
        reasons: list[str] = []
        policy = self.config.policy

        if applicant.age < policy.min_age:
            reasons.append(f"Applicant below minimum age ({policy.min_age}).")

        if applicant.monthly_income_kes < policy.min_monthly_income_kes:
            reasons.append(
                f"Income below minimum threshold (KES {policy.min_monthly_income_kes:,.0f})."
            )

        debt_to_income = applicant.existing_debt_kes / max(applicant.monthly_income_kes, 1)
        if debt_to_income > policy.max_debt_to_income:
            reasons.append(
                f"Debt-to-income {debt_to_income:.0%} exceeds limit "
                f"({policy.max_debt_to_income:.0%})."
            )

        features = applicant.features

        if applicant.crb_defaults > policy.max_crb_defaults and features.get(
            "has_crb_record", 0
        ) >= 0.5:
            reasons.append("Active CRB default listing detected.")

        reasons.extend(self._channel_rules(applicant))
        return PolicyOutcome(passed=len(reasons) == 0, reasons=tuple(reasons))

    def _channel_rules(self, applicant: ApplicantProfile) -> list[str]:
        features = applicant.features
        rules = self.config.channel_minimums.get(applicant.channel.value, {})
        reasons: list[str] = []

        if applicant.channel in {Channel.MPESA, Channel.UNBANKED}:
            if features.get("has_mpesa_wallet", 0) < 0.5 and applicant.channel == Channel.UNBANKED:
                reasons.append("M-Pesa wallet data required for unbanked scoring.")
            wallet_rules = self.config.channel_minimums.get(
                applicant.channel.value,
                self.config.channel_minimums.get("mpesa", {}),
            )
            if features.get("kyc_tier", 0) < wallet_rules.get("min_kyc_tier", 2):
                reasons.append("M-Pesa KYC tier too low for lending.")
            if features.get("fuliza_utilization", 0) > wallet_rules.get("max_fuliza_utilization", 1):
                reasons.append("Fuliza/overdraft utilization exceeds channel limit.")
            if features.get("wallet_activity_days_90d", 0) < wallet_rules.get(
                "min_wallet_activity_days_90d", 0
            ):
                reasons.append("Insufficient M-Pesa wallet activity in last 90 days.")

        if applicant.channel == Channel.SACCO or features.get("has_sacco_membership", 0) >= 0.5:
            sacco_rules = self.config.channel_minimums.get("sacco", {})
            if features.get("membership_months", 0) < sacco_rules.get("min_membership_months", 0):
                reasons.append("SACCO membership tenure below minimum.")
            if features.get("share_capital_kes", 0) < sacco_rules.get("min_share_capital_kes", 0):
                reasons.append("Share capital below SACCO minimum.")
            if features.get("prior_loan_repayment_rate", 1) < sacco_rules.get("min_repayment_rate", 0):
                reasons.append("Historical SACCO loan repayment rate too low.")

        if applicant.channel == Channel.BANK or features.get("has_bank_account", 0) >= 0.5:
            bank_rules = self.config.channel_minimums.get("bank", {})
            if features.get("account_age_months", 0) < bank_rules.get("min_account_age_months", 0):
                reasons.append("Bank account age below minimum.")
            if features.get("bounced_cheques_12m", 0) > bank_rules.get("max_bounced_cheques_12m", 0):
                reasons.append("Too many bounced cheques in the last 12 months.")
            if features.get("avg_monthly_balance_kes", 0) < bank_rules.get("min_avg_balance_kes", 0):
                reasons.append("Average monthly balance below bank threshold.")

        if applicant.channel == Channel.MOBILE_LENDER:
            lender_rules = self.config.channel_minimums.get("mobile_lender", {})
            if features.get("mpesa_statement_days_covered", 0) < lender_rules.get(
                "min_mpesa_statement_days_covered", 0
            ):
                reasons.append("Insufficient M-Pesa statement history provided.")
            if features.get("mpesa_inferred_repayment_rate", 1) < lender_rules.get(
                "min_mpesa_inferred_repayment_rate", 0
            ):
                reasons.append("M-Pesa statement shows weak cross-lender repayment behaviour.")
            if features.get("mpesa_active_lender_count", 0) > lender_rules.get(
                "max_mpesa_active_lender_count", 99
            ):
                reasons.append("Too many active digital lenders on M-Pesa statement (stacking).")
            if features.get("mpesa_late_repayment_events_12m", 0) > lender_rules.get(
                "max_mpesa_late_repayment_events_12m", 99
            ):
                reasons.append("Late repayment events detected on M-Pesa statement.")

        return reasons
