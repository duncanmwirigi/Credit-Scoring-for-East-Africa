from __future__ import annotations

from src.config import AppConfig
from src.domain import ApplicantProfile, Channel, CreditDecision, Decision, LimitAdjustment, LoanLimitAssignment


class LoanLimitEngine:
    """Assign and adjust loan limits from score, history, and phone-derived signals."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.settings = config.loan_limits

    def _round_limit(self, amount: float) -> float:
        step = self.settings.get("round_to_kes", 500)
        return max(step, round(amount / step) * step)

    def _channel_limits(self, channel: Channel) -> dict:
        return self.settings["channels"][channel.value]

    def _tier_for_limit(self, limit_kes: float) -> str:
        tiers = self.settings.get("tiers", {})
        ordered = sorted(tiers.items(), key=lambda item: item[1], reverse=True)
        for name, threshold in ordered:
            if limit_kes >= threshold:
                return name
        return "starter"

    def assign(self, applicant: ApplicantProfile, decision: CreditDecision) -> LoanLimitAssignment:
        features = applicant.features
        channel_limits = self._channel_limits(applicant.channel)
        min_limit = channel_limits["min_limit_kes"]
        max_limit = channel_limits["max_limit_kes"]
        prior_limit = float(features.get("highest_prior_limit_kes", 0))
        requested = applicant.requested_amount_kes

        if decision.decision == Decision.DECLINE or not decision.policy.passed:
            return LoanLimitAssignment(
                approved_limit_kes=0.0,
                min_limit_kes=min_limit,
                max_limit_kes=max_limit,
                prior_limit_kes=prior_limit,
                requested_limit_kes=requested,
                adjustment=LimitAdjustment.SUSPENDED,
                adjustment_pct=0.0,
                tier="starter",
                reasons=decision.policy.reasons or ("Risk score or policy decline.",),
                next_review_days=30,
            )

        score_multiplier = self.settings["score_multipliers"].get(decision.decision.value, 0.0)
        if prior_limit > 0:
            base_limit = prior_limit
        else:
            base_limit = channel_limits["first_time_base_kes"]

        limit = base_limit * score_multiplier
        reasons: list[str] = []

        streak = int(features.get("on_time_repayment_streak", 0))
        repayment_rate = float(features.get("lifetime_repayment_rate", 1.0))
        default_count = int(features.get("lifetime_default_count", 0))
        days_since_default = float(features.get("days_since_last_default", 9999))

        if streak >= 6:
            limit *= self.settings["repayment_adjustments"]["streak_6_multiplier"]
            reasons.append(f"{streak}-loan on-time streak increased limit.")
        elif streak >= 3:
            limit *= self.settings["repayment_adjustments"]["streak_3_multiplier"]
            reasons.append(f"{streak}-loan on-time streak increased limit.")

        if repayment_rate >= 0.95 and float(features.get("lifetime_loans_count", 0)) >= 2:
            limit *= self.settings["repayment_adjustments"]["rate_above_95_multiplier"]
            reasons.append("Strong lifetime repayment rate.")

        if default_count > 0 and days_since_default < 365:
            limit *= self.settings["repayment_adjustments"]["recent_default_multiplier"]
            reasons.append("Recent default reduced limit.")

        if repayment_rate < 0.80 and float(features.get("lifetime_loans_count", 0)) >= 1:
            limit *= self.settings["repayment_adjustments"]["poor_rate_multiplier"]
            reasons.append("Weak repayment history reduced limit.")

        if float(features.get("alternative_data_consent", 0)) >= 0.5:
            income_ratio = float(features.get("income_declared_vs_sms_ratio", 1.0))
            if 0.85 <= income_ratio <= 1.15 and float(features.get("sms_salary_detected", 0)) >= 0.5:
                limit *= self.settings["alternative_data_adjustments"]["income_verified_multiplier"]
                reasons.append("SMS salary verified against declared income.")

            if float(features.get("sms_gambling_ratio", 0)) > 0.25:
                limit *= self.settings["alternative_data_adjustments"]["high_gambling_multiplier"]
                reasons.append("Elevated gambling SMS activity reduced limit.")

            if float(features.get("apps_lending_app_count", 0)) > 3:
                limit *= self.settings["alternative_data_adjustments"]["loan_stacking_apps_multiplier"]
                reasons.append("Multiple lending apps detected (stacking risk).")

        income_cap = applicant.monthly_income_kes * self.settings.get("max_income_multiple", 0.4)
        if limit > income_cap:
            limit = income_cap
            reasons.append("Limit capped at income multiple.")

        limit = min(limit, max_limit)
        limit = max(limit, min_limit) if decision.decision == Decision.APPROVE else min(limit, min_limit * 0.5)
        limit = min(limit, requested) if decision.decision == Decision.REVIEW else limit
        limit = self._round_limit(limit)

        if prior_limit <= 0:
            adjustment = LimitAdjustment.FIRST_TIME
            adjustment_pct = 1.0
        elif limit > prior_limit * 1.05:
            adjustment = LimitAdjustment.INCREASE
            adjustment_pct = (limit - prior_limit) / prior_limit
            reasons.append(f"Limit increased from KES {prior_limit:,.0f}.")
        elif limit < prior_limit * 0.95:
            adjustment = LimitAdjustment.DECREASE
            adjustment_pct = (prior_limit - limit) / prior_limit
            reasons.append(f"Limit reduced from KES {prior_limit:,.0f}.")
        else:
            adjustment = LimitAdjustment.MAINTAIN
            adjustment_pct = 0.0
            reasons.append("Limit maintained based on current behaviour.")

        next_review = 90 if adjustment in {LimitAdjustment.INCREASE, LimitAdjustment.MAINTAIN} else 30
        if decision.decision == Decision.REVIEW:
            next_review = 14

        return LoanLimitAssignment(
            approved_limit_kes=limit,
            min_limit_kes=min_limit,
            max_limit_kes=max_limit,
            prior_limit_kes=prior_limit,
            requested_limit_kes=requested,
            adjustment=adjustment,
            adjustment_pct=adjustment_pct,
            tier=self._tier_for_limit(limit),
            reasons=tuple(reasons[:6]),
            next_review_days=next_review,
        )
