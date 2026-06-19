from __future__ import annotations

"""Extract credit features from phone permissions: SMS inbox, call log, contacts, apps.

Collected on-device with explicit user consent — no third-party lender APIs required.
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Iterable

_COLLECTION_KEYWORDS = (
    "debt collector",
    "overdue",
    "default notice",
    "final demand",
    "recovery",
    "loan reminder",
    "repayment due",
)

_LENDER_PROMO_KEYWORDS = (
    "instant loan",
    "get cash now",
    "pre-approved",
    "borrow",
    "loan offer",
)

_GAMBLING_KEYWORDS = ("bet", "jackpot", "casino", "sportpesa", "betika", "odds")

_SALARY_KEYWORDS = ("salary", "paid to", "welfare", "allowance", "payroll")

_BILL_KEYWORDS = ("kplc", "nairobi water", "zuku", "safaricom bill", "airtime")


@dataclass(frozen=True)
class PhoneSmsMessage:
    received_at: datetime
    sender: str
    body: str


@dataclass(frozen=True)
class PhoneCallRecord:
    started_at: datetime
    direction: str  # "incoming" | "outgoing" | "missed"
    duration_seconds: float
    contact_name: str = ""
    phone_number: str = ""


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def derive_sms_features(
    messages: Iterable[PhoneSmsMessage],
    *,
    declared_income_kes: float = 0,
    lookback_days: int = 30,
) -> dict[str, float]:
    rows = list(messages)
    if not rows:
        return {
            "sms_salary_detected": 0.0,
            "sms_inferred_monthly_income_kes": 0.0,
            "sms_mpesa_txn_count_30d": 0.0,
            "sms_total_count_30d": 0.0,
            "sms_bill_pay_regularity": 0.0,
            "sms_other_lender_repayment_count": 0.0,
            "sms_collection_message_count_30d": 0.0,
            "sms_lender_promo_count_30d": 0.0,
            "sms_gambling_ratio": 0.0,
            "income_declared_vs_sms_ratio": 1.0,
        }

    latest = max(msg.received_at for msg in rows)
    recent = [msg for msg in rows if (latest - msg.received_at).days <= lookback_days]
    total = len(recent)

    mpesa_txn = 0
    bill_hits = 0
    lender_repayments = 0
    collection_msgs = 0
    lender_promos = 0
    gambling_msgs = 0
    salary_hits = 0
    inferred_income = 0.0

    for msg in recent:
        text = f"{msg.sender} {msg.body}"
        if "mpesa" in text.lower() or "confirmed" in text.lower():
            mpesa_txn += 1
        if _contains_any(text, _BILL_KEYWORDS):
            bill_hits += 1
        if _contains_any(text, _COLLECTION_KEYWORDS):
            collection_msgs += 1
            if "repay" in text.lower() or "payment" in text.lower():
                lender_repayments += 1
        if _contains_any(text, _LENDER_PROMO_KEYWORDS):
            lender_promos += 1
        if _contains_any(text, _GAMBLING_KEYWORDS):
            gambling_msgs += 1
        if _contains_any(text, _SALARY_KEYWORDS):
            salary_hits += 1

    salary_detected = 1.0 if salary_hits > 0 else 0.0
    if salary_detected and declared_income_kes > 0:
        inferred_income = declared_income_kes * 0.95
    elif mpesa_txn > 0 and declared_income_kes > 0:
        inferred_income = declared_income_kes * 0.7

    bill_regularity = min(1.0, bill_hits / 4.0)
    gambling_ratio = gambling_msgs / max(total, 1)
    income_ratio = inferred_income / max(declared_income_kes, 1) if declared_income_kes else 1.0

    return {
        "sms_salary_detected": salary_detected,
        "sms_inferred_monthly_income_kes": inferred_income,
        "sms_mpesa_txn_count_30d": float(mpesa_txn),
        "sms_total_count_30d": float(total),
        "sms_bill_pay_regularity": bill_regularity,
        "sms_other_lender_repayment_count": float(lender_repayments),
        "sms_collection_message_count_30d": float(collection_msgs),
        "sms_lender_promo_count_30d": float(lender_promos),
        "sms_gambling_ratio": gambling_ratio,
        "income_declared_vs_sms_ratio": income_ratio if declared_income_kes else 1.0,
    }


def derive_call_features(
    calls: Iterable[PhoneCallRecord],
    *,
    lookback_days: int = 30,
) -> dict[str, float]:
    rows = list(calls)
    if not rows:
        return {
            "call_total_count_30d": 0.0,
            "call_unique_contacts_30d": 0.0,
            "call_avg_duration_seconds": 0.0,
            "call_incoming_ratio": 0.5,
            "call_missed_ratio": 0.0,
            "call_collection_agency_count_30d": 0.0,
            "call_night_activity_ratio": 0.0,
        }

    latest = max(call.started_at for call in rows)
    recent = [call for call in rows if (latest - call.started_at).days <= lookback_days]
    if not recent:
        return derive_call_features([])

    contacts: set[str] = set()
    incoming = 0
    missed = 0
    night_calls = 0
    collection_calls = 0
    durations: list[float] = []

    for call in recent:
        key = call.contact_name or call.phone_number
        if key:
            contacts.add(key.lower())
        if call.direction == "incoming":
            incoming += 1
        if call.direction == "missed":
            missed += 1
        if call.duration_seconds > 0:
            durations.append(call.duration_seconds)
        label = f"{call.contact_name} {call.phone_number}".lower()
        if _contains_any(label, _COLLECTION_KEYWORDS):
            collection_calls += 1
        if call.started_at.time() >= time(22, 0) or call.started_at.time() <= time(6, 0):
            night_calls += 1

    total = len(recent)
    return {
        "call_total_count_30d": float(total),
        "call_unique_contacts_30d": float(len(contacts)),
        "call_avg_duration_seconds": float(sum(durations) / max(len(durations), 1)),
        "call_incoming_ratio": incoming / max(total, 1),
        "call_missed_ratio": missed / max(total, 1),
        "call_collection_agency_count_30d": float(collection_calls),
        "call_night_activity_ratio": night_calls / max(total, 1),
    }


def derive_device_features(
    *,
    device_tenure_days: float = 0,
    contacts_count: float = 0,
    saved_contacts_count: float = 0,
    apps_lending_app_count: float = 0,
    apps_gambling_app_count: float = 0,
    os_name: str = "android",
    os_version: str = "13",
    device_tier: int = 2,
    ram_gb: float = 3,
    storage_free_ratio: float = 0.5,
    dual_sim: bool = True,
    network_4g_plus: bool = True,
    model_age_months: float = 24,
) -> dict[str, float]:
    saved_ratio = saved_contacts_count / max(contacts_count, 1) if contacts_count else 0.0
    os_version_score = 0.5
    try:
        major = int(os_version.split(".")[0])
        os_version_score = min(1.0, major / 15.0)
    except ValueError:
        pass
    return {
        "device_tenure_days": device_tenure_days,
        "contacts_count": contacts_count,
        "contacts_saved_ratio": min(1.0, saved_ratio),
        "apps_lending_app_count": apps_lending_app_count,
        "apps_gambling_app_count": apps_gambling_app_count,
        "device_os_android": 1.0 if os_name.lower() == "android" else 0.0,
        "device_os_version_score": os_version_score,
        "device_tier": float(device_tier),
        "device_ram_gb": ram_gb,
        "device_storage_free_ratio": min(1.0, max(0.0, storage_free_ratio)),
        "device_dual_sim": 1.0 if dual_sim else 0.0,
        "device_network_4g_plus": 1.0 if network_4g_plus else 0.0,
        "device_model_age_months": model_age_months,
    }


def derive_phone_data_features(
    *,
    consent: float,
    sms_messages: Iterable[PhoneSmsMessage] | None = None,
    call_records: Iterable[PhoneCallRecord] | None = None,
    declared_income_kes: float = 0,
    device_tenure_days: float = 0,
    contacts_count: float = 0,
    saved_contacts_count: float = 0,
    apps_lending_app_count: float = 0,
    apps_gambling_app_count: float = 0,
    os_name: str = "android",
    os_version: str = "13",
    device_tier: int = 2,
    ram_gb: float = 3,
    storage_free_ratio: float = 0.5,
    dual_sim: bool = True,
    network_4g_plus: bool = True,
    model_age_months: float = 24,
) -> dict[str, float]:
    """Merge SMS, call-log, and device signals into one feature dict."""
    features: dict[str, float] = {"alternative_data_consent": consent}
    if consent < 0.5:
        return features

    features.update(
        derive_sms_features(sms_messages or [], declared_income_kes=declared_income_kes)
    )
    features.update(derive_call_features(call_records or []))
    features.update(
        derive_device_features(
            device_tenure_days=device_tenure_days,
            contacts_count=contacts_count,
            saved_contacts_count=saved_contacts_count,
            apps_lending_app_count=apps_lending_app_count,
            apps_gambling_app_count=apps_gambling_app_count,
            os_name=os_name,
            os_version=os_version,
            device_tier=device_tier,
            ram_gb=ram_gb,
            storage_free_ratio=storage_free_ratio,
            dual_sim=dual_sim,
            network_4g_plus=network_4g_plus,
            model_age_months=model_age_months,
        )
    )
    return features
