# East Africa Credit Scoring Engine

A production-style **credit scoring platform** for **M-Pesa**, **SACCO**, **bank**, and **mobile digital lender** channels in East Africa — covering telco-led lending, co-operatives, traditional banks, and app-based lenders such as **Tala**, **Branch**, **Zenka**, and **Okash**.

It combines channel-specific feature engineering, a calibrated ML model, scorecard conversion (300–850), deterministic policy rules, **SHAP explainability**, and a **FastAPI REST service** with regulatory audit trails.

## Platform at a glance

| Capability | Status | Entry point |
|------------|--------|-------------|
| Multi-channel scoring (M-Pesa / SACCO / bank / mobile lender) | ✅ | `score.py`, `/score` |
| ML training + evaluation (AUC, Gini, KS) | ✅ | `train.py` |
| Calibrated probability of default (PD) | ✅ | `src/ml/trainer.py` |
| Scorecard mapping (PDO methodology) | ✅ | `src/ml/scorer.py` |
| Policy engine (CRB, DTI, channel rules) | ✅ | `src/policy/engine.py` |
| SHAP feature explainability | ✅ | `src/ml/explainability.py` |
| Regulatory audit trail (JSON) | ✅ | `assets/audit_trails/` |
| FastAPI REST service | ✅ | `serve.py` → port 8000 |
| OpenAPI / Swagger docs | ✅ | `/docs` |
| Synthetic training data (no PII) | ✅ | `src/data/synthetic.py` |

## Supported channels

| Channel | `channel` value | API feature block | Examples |
|---------|-----------------|-------------------|----------|
| M-Pesa mobile money | `mpesa` | `mpesa_features` | Safaricom lending, Fuliza |
| Mobile digital lender | `mobile_lender` | `mobile_lender_features` | Tala, Branch, Zenka, Okash |
| SACCO | `sacco` | `sacco_features` | Stima, Harambee, Sheria SACCO |
| Bank | `bank` | `bank_features` | KCB, Equity, Co-op Bank |

The model uses **43 features** total (8 common + 35 channel-specific). Only the feature block matching the applicant's channel is populated; all others are zeroed before scoring.

## Architecture

```mermaid
flowchart TB
    subgraph inputs [Input Channels]
        M[M-Pesa]
        MLend[Mobile Lender]
        S[SACCO]
        B[Bank]
    end

    subgraph pipeline [Scoring Pipeline]
        FE[Feature Engineering]
        ML[Calibrated ML Model]
        SC[Scorecard 300-850]
        PO[Policy Engine]
        SH[SHAP Explainer]
    end

    subgraph outputs [Outputs]
        DEC[Approve / Review / Decline]
        AT[Audit Trail JSON]
        API[REST API Response]
    end

    M --> FE
    MLend --> FE
    S --> FE
    B --> FE
    FE --> ML
    ML --> SC
    ML --> SH
    SC --> PO
    PO --> DEC
    SH --> AT
    DEC --> API
    AT --> API
```

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **Feature engineering** | `src/features/engineering.py` | Channel-aware variables; masks inactive channel columns |
| **ML model** | `src/ml/trainer.py` | Gradient boosting with sigmoid calibration |
| **Scorecard** | `src/ml/scorer.py` | PD → credit score using PDO (Points to Double Odds) |
| **Policy engine** | `src/policy/engine.py` | Hard rules: CRB defaults, DTI caps, channel minimums |
| **SHAP explainability** | `src/ml/explainability.py` | Per-feature risk contributions for audit/disclosure |
| **Audit trails** | `src/ml/explainability.py` | Timestamped JSON records for compliance review |
| **REST API** | `src/api/app.py` | Typed HTTP interface with Pydantic validation |

## End-to-end workflow

```
1. train.py          →  synthetic portfolio  →  train model  →  models/ + assets/
2. score.py          →  sample applicants    →  score + SHAP  →  assets/audit_trails/
3. serve.py          →  load latest model    →  REST API      →  live scoring + audits
```

## Channel coverage

One unified model serves all four channels. Channel-specific features are zeroed out for rows that do not belong to that channel, so each applicant is scored on the signals that matter for their lending path.

### M-Pesa (mobile money)

| Feature | Description |
|---------|-------------|
| `kyc_tier` | KYC verification level (1–3) |
| `wallet_activity_days_90d` | Active wallet days in last 90 days |
| `avg_monthly_txn_count` | Average monthly transaction count |
| `avg_txn_amount_kes` | Average transaction amount (KES) |
| `cash_in_out_ratio` | Cash-in vs cash-out ratio |
| `merchant_spend_ratio` | Share of spend at merchants |
| `fuliza_utilization` | Fuliza/overdraft utilization (0–1) |
| `wallet_balance_volatility` | Balance fluctuation measure |
| `days_since_last_txn` | Recency of last transaction |

**Policy checks:** minimum KYC tier (≥ 2), max Fuliza utilization (≤ 85%), minimum wallet activity (≥ 30 days in 90d).

| Config key | Threshold |
|------------|-----------|
| `min_kyc_tier` | 2 |
| `max_fuliza_utilization` | 0.85 |
| `min_wallet_activity_days_90d` | 30 |

### SACCO (co-operative lending)

| Feature | Description |
|---------|-------------|
| `membership_months` | Tenure as SACCO member |
| `share_capital_kes` | Share capital contributed (KES) |
| `monthly_savings_kes` | Regular monthly savings (KES) |
| `savings_consistency_score` | Consistency of savings (0–1) |
| `prior_loan_repayment_rate` | Historical repayment rate within SACCO |
| `guarantor_count` | Number of guarantors |
| `guarantor_avg_score` | Average guarantor credit score |
| `dividend_years` | Years dividends received |

**Policy checks:** minimum membership (≥ 6 months), share capital (≥ KES 5,000), repayment rate (≥ 85%).

| Config key | Threshold |
|------------|-----------|
| `min_membership_months` | 6 |
| `min_share_capital_kes` | 5,000 |
| `min_repayment_rate` | 0.85 |

### Bank (traditional lending)

| Feature | Description |
|---------|-------------|
| `account_age_months` | Account tenure |
| `avg_monthly_balance_kes` | Average monthly balance (KES) |
| `salary_deposit_regularity` | Regularity of salary deposits (0–1) |
| `bounced_cheques_12m` | Bounced cheques in last 12 months |
| `overdraft_usage_ratio` | Overdraft utilization (0–1) |
| `credit_card_utilization` | Credit card utilization (0–1) |
| `existing_loan_count` | Number of existing loans |
| `branch_relationship_score` | Branch relationship strength (0–1) |

**Policy checks:** minimum account age (≥ 6 months), max bounced cheques (≤ 1 in 12m), minimum balance (≥ KES 10,000).

| Config key | Threshold |
|------------|-----------|
| `min_account_age_months` | 6 |
| `max_bounced_cheques_12m` | 1 |
| `min_avg_balance_kes` | 10,000 |

### Mobile digital lender (Tala, Branch, Zenka, Okash, etc.)

App-based **mobile loan** providers that disburse and collect via M-Pesa, use alternative data (SMS, app usage, device signals), and often serve underbanked customers with short-term, high-frequency loans.

| Feature | Description |
|---------|-------------|
| `platform_tenure_months` | Months as a customer on the lending app |
| `prior_loans_on_platform` | Number of previous loans with this lender |
| `platform_repayment_rate` | Historical on-time repayment rate (0–1) |
| `days_since_last_repayment` | Recency of last successful repayment |
| `active_digital_loans_count` | Active loans across digital lenders (loan stacking) |
| `avg_historical_loan_kes` | Average past loan size (KES) |
| `rollover_count_12m` | Loan extensions/rollovers in last 12 months |
| `app_engagement_score` | App login and usage engagement (0–1) |
| `mpesa_disbursement_linked` | M-Pesa linked for disbursement/repayment (0/1) |
| `alternative_data_score` | Alternative data signal strength — SMS, device, behaviour (0–1) |

**Policy checks:** minimum platform tenure (≥ 1 month), repayment rate (≥ 80%), max active digital loans (≤ 2), max rollovers (≤ 3 in 12m).

| Config key | Threshold |
|------------|-----------|
| `min_platform_tenure_months` | 1 |
| `min_platform_repayment_rate` | 0.80 |
| `max_active_digital_loans` | 2 |
| `max_rollover_count_12m` | 3 |

**Typical use cases:**
- First-time vs repeat borrower limits on Tala or Branch
- Detecting **loan stacking** across multiple digital lenders
- Pricing repeat borrowers with strong platform repayment history
- Declining applicants with excessive rollovers or CRB defaults

### Common features (all channels)

| Feature | Description |
|---------|-------------|
| `age` | Applicant age |
| `monthly_income_kes` | Declared monthly income (KES) |
| `requested_amount_kes` | Loan amount requested (KES) |
| `existing_debt_kes` | Existing debt obligations (KES) |
| `debt_to_income` | Derived: existing debt / income |
| `loan_to_income` | Derived: requested amount / income |
| `crb_defaults` | Active CRB default listings |
| `crb_inquiries_6m` | CRB inquiries in last 6 months |

## Project structure

```
credit_scoring/
├── config/
│   └── scoring.yaml              # Thresholds, scorecard, channel rules
├── src/
│   ├── __init__.py
│   ├── config.py                 # Typed YAML config loader
│   ├── domain.py                 # Applicant, Decision, Policy models
│   ├── data/
│   │   └── synthetic.py          # Synthetic portfolio generator
│   ├── features/
│   │   └── engineering.py        # Feature matrix builder + channel masking
│   ├── ml/
│   │   ├── trainer.py            # Train, evaluate, persist model
│   │   ├── scorer.py             # Score applicants + audit integration
│   │   ├── explainability.py     # SHAP explainer + audit trail writer
│   │   └── metrics.py            # AUC, Gini, KS, Brier
│   ├── api/
│   │   ├── app.py                # FastAPI application
│   │   └── schemas.py            # Pydantic request/response models
│   └── policy/
│       └── engine.py             # Business & regulatory rules
├── train.py                      # Training entry point
├── score.py                      # CLI scoring + SHAP + audit trails
├── serve.py                      # Start FastAPI on port 8000
├── models/                       # Saved models (generated)
│   ├── credit_model_*.joblib
│   ├── credit_model_*.json       # Model metadata
│   └── latest_model.txt          # Pointer to active model
├── assets/                       # Reports (generated)
│   ├── training_metrics.json
│   ├── feature_importance.csv
│   ├── roc_curve.png
│   ├── sample_decisions.json
│   └── audit_trails/             # Regulatory audit JSON per score
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10+
- scikit-learn, pandas, numpy, matplotlib, pyyaml, joblib
- fastapi, uvicorn, shap, pydantic

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes: `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `joblib`, `fastapi`, `uvicorn`, `shap`, `pydantic`.

> **Note:** Install `scikit-learn`, not the deprecated PyPI package `sklearn`.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (required before scoring or API)
python3 train.py

# 3. Score sample applicants with SHAP + audit trails
python3 score.py

# 4. Start the REST API
python3 serve.py
# → http://127.0.0.1:8000/docs
```

---

## 1. Training (`train.py`)

Generates a synthetic portfolio and trains a calibrated gradient boosting classifier.

```bash
python3 train.py
```

**What it does:**

1. Generates 8,000 synthetic applicants (30% M-Pesa, 25% mobile lender, 25% SACCO, 20% bank).
2. Splits 80/20 train/test with stratification on default label.
3. Trains `GradientBoostingClassifier` wrapped in `CalibratedClassifierCV`.
4. Evaluates on hold-out test set.
5. Saves versioned artifacts to `models/` and `assets/`.

**Latest training metrics** (synthetic portfolio):

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.909 |
| Gini | 0.818 |
| KS | 0.815 |
| Brier score | 0.058 |
| Test accuracy | 0.903 |
| Precision | 0.573 |
| Recall | 0.734 |
| Default rate | 12% |
| Feature count | 43 |
| Train / test split | 6,400 / 1,600 |

**Generated artifacts:**

| File | Contents |
|------|----------|
| `models/credit_model_*.joblib` | Serialized sklearn pipeline |
| `models/credit_model_*.json` | Feature list, scorecard, metrics |
| `models/latest_model.txt` | Pointer to active model |
| `assets/training_metrics.json` | Full evaluation report |
| `assets/feature_importance.csv` | Model feature importances |
| `assets/roc_curve.png` | ROC curve plot |

---

## 2. CLI scoring (`score.py`)

Scores six sample applicants across all channels — including **Tala-style mobile lender** cases — plus high-risk M-Pesa and mobile lender scenarios. Each score includes SHAP explainability and a persisted audit trail.

```bash
python3 score.py
```

**Sample output:**

```
MPESA-001      | channel=mpesa         | score=688 | pd=1.49%  | decision=APPROVE
SACCO-014      | channel=sacco         | score=684 | pd=1.72%  | decision=APPROVE
BANK-203       | channel=bank          | score=664 | pd=3.33%  | decision=REVIEW
TALA-001       | channel=mobile_lender | score=686 | pd=1.62%  | decision=APPROVE
TALA-RISK-99   | channel=mobile_lender | score=543 | pd=69.47% | decision=DECLINE
MPESA-RISK-77  | channel=mpesa         | score=569 | pd=48.78% | decision=DECLINE
```

Re-run `python3 train.py` then `python3 score.py` after config or feature changes to refresh scores.

**Sample applicants scored by `score.py`:**

| ID | Channel | Scenario |
|----|---------|----------|
| `MPESA-001` | `mpesa` | Strong M-Pesa wallet profile → APPROVE |
| `SACCO-014` | `sacco` | Long-tenure SACCO member → APPROVE |
| `BANK-203` | `bank` | Solid bank relationship → REVIEW |
| `TALA-001` | `mobile_lender` | Repeat Tala/Branch-style borrower → APPROVE |
| `TALA-RISK-99` | `mobile_lender` | Loan stacking + CRB default → DECLINE |
| `MPESA-RISK-77` | `mpesa` | High Fuliza use + CRB default → DECLINE |

**Generated artifacts:**

| File | Contents |
|------|----------|
| `assets/sample_decisions.json` | All sample scores with SHAP payloads |
| `assets/audit_trails/{audit_id}.json` | One regulatory audit file per applicant |

---

## 3. REST API (`serve.py`)

Production-ready FastAPI service that loads the latest trained model at startup.

```bash
python3 serve.py
```

- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Base URL:** `http://127.0.0.1:8000`

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health and model load status |
| `/model/info` | GET | Active model version, scorecard, decision bands |
| `/score` | POST | Score one applicant (SHAP + audit optional) |
| `/score/batch` | POST | Score up to 100 applicants in one request |

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

### `GET /model/info`

```bash
curl http://127.0.0.1:8000/model/info
```

Returns project name, model file, feature count, scorecard settings, and decision bands.

### `POST /score`

Score a single applicant. Set `include_shap: true` and `persist_audit_trail: true` for full regulatory output.

**M-Pesa example:**

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_id": "MPESA-001",
    "channel": "mpesa",
    "age": 34,
    "monthly_income_kes": 85000,
    "requested_amount_kes": 50000,
    "existing_debt_kes": 5000,
    "crb_defaults": 0,
    "crb_inquiries_6m": 0,
    "include_shap": true,
    "persist_audit_trail": true,
    "mpesa_features": {
      "kyc_tier": 3,
      "wallet_activity_days_90d": 88,
      "avg_monthly_txn_count": 95,
      "avg_txn_amount_kes": 4200,
      "cash_in_out_ratio": 0.95,
      "merchant_spend_ratio": 0.35,
      "fuliza_utilization": 0.05,
      "wallet_balance_volatility": 0.08,
      "days_since_last_txn": 0
    }
  }'
```

**SACCO example** — use `"channel": "sacco"` and provide `sacco_features`:

```json
"sacco_features": {
  "membership_months": 72,
  "share_capital_kes": 95000,
  "monthly_savings_kes": 18000,
  "savings_consistency_score": 0.97,
  "prior_loan_repayment_rate": 1.0,
  "guarantor_count": 4,
  "guarantor_avg_score": 760,
  "dividend_years": 6
}
```

**Bank example** — use `"channel": "bank"` and provide `bank_features`:

```json
"bank_features": {
  "account_age_months": 60,
  "avg_monthly_balance_kes": 280000,
  "salary_deposit_regularity": 0.98,
  "bounced_cheques_12m": 0,
  "overdraft_usage_ratio": 0.03,
  "credit_card_utilization": 0.12,
  "existing_loan_count": 0,
  "branch_relationship_score": 0.91
}
```

**Mobile digital lender example** (Tala / Branch) — use `"channel": "mobile_lender"` and provide `mobile_lender_features`:

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_id": "TALA-001",
    "channel": "mobile_lender",
    "age": 31,
    "monthly_income_kes": 48000,
    "requested_amount_kes": 15000,
    "existing_debt_kes": 8000,
    "crb_defaults": 0,
    "crb_inquiries_6m": 1,
    "include_shap": true,
    "persist_audit_trail": true,
    "mobile_lender_features": {
      "platform_tenure_months": 24,
      "prior_loans_on_platform": 8,
      "platform_repayment_rate": 0.96,
      "days_since_last_repayment": 12,
      "active_digital_loans_count": 1,
      "avg_historical_loan_kes": 12000,
      "rollover_count_12m": 1,
      "app_engagement_score": 0.88,
      "mpesa_disbursement_linked": 1,
      "alternative_data_score": 0.79
    }
  }'
```

**Response fields:**

| Field | Description |
|-------|-------------|
| `probability_of_default` | Model-estimated PD (0–1) |
| `credit_score` | Scorecard score (300–850) |
| `decision` | `approve`, `review`, or `decline` |
| `policy.passed` | Whether hard policy rules passed |
| `policy.reasons` | Policy decline reasons (if any) |
| `top_risk_factors` | Rule-based risk factor summary |
| `shap` | SHAP explanation (if requested) |
| `audit_id` | Audit trail ID (if persisted) |
| `model_version` | Model version used for the decision |

**Example response (truncated):**

```json
{
  "applicant_id": "TALA-001",
  "channel": "mobile_lender",
  "probability_of_default": 0.0162,
  "credit_score": 686,
  "decision": "approve",
  "policy": { "passed": true, "reasons": [] },
  "shap": {
    "base_value": -3.529231,
    "predicted_log_odds": -4.412,
    "summary": "Primary drivers increasing default risk: none. Primary drivers reducing risk: platform_repayment_rate, debt_to_income, monthly_income_kes.",
    "contributions": [
      {
        "feature": "platform_repayment_rate",
        "raw_value": 0.96,
        "shap_value": -0.42,
        "impact": "decreases_default_risk"
      }
    ]
  },
  "audit_id": "TALA-001-4fddb337",
  "model_version": "1.0.0"
}
```

### `POST /score/batch`

Score multiple applicants in one call. Batch defaults: `include_shap: false`, `persist_audit_trail: false` (override per applicant if needed).

```bash
curl -X POST http://127.0.0.1:8000/score/batch \
  -H "Content-Type: application/json" \
  -d '{
    "include_shap": false,
    "persist_audit_trail": false,
    "applicants": [
      {
        "applicant_id": "MPESA-001",
        "channel": "mpesa",
        "age": 34,
        "monthly_income_kes": 85000,
        "requested_amount_kes": 50000,
        "existing_debt_kes": 5000,
        "crb_defaults": 0,
        "crb_inquiries_6m": 0,
        "mpesa_features": { "...": "..." }
      }
    ]
  }'
```

---

## Decision logic

Final decisions combine **three independent layers**:

```mermaid
flowchart TD
    A[Applicant] --> B[ML Model → PD]
    B --> C[Scorecard → Credit Score]
    C --> D{Policy rules pass?}
    D -->|No| E[DECLINE]
    D -->|Yes| F{Score ≥ 680?}
    F -->|Yes| G[APPROVE]
    F -->|No| H{Score ≥ 550?}
    H -->|Yes| I[REVIEW]
    H -->|No| E
```

| Decision | Condition |
|----------|-----------|
| **APPROVE** | Policy passed **and** credit score ≥ 680 |
| **REVIEW** | Policy passed **and** 550 ≤ score < 680 |
| **DECLINE** | Policy failed **or** score < 550 |

Policy failures always result in **DECLINE**, regardless of credit score. Examples:

- Active **CRB default** (all channels)
- **Debt-to-income** above 45% (all channels)
- M-Pesa **KYC tier** too low or **Fuliza** over-utilised
- SACCO **membership** or **share capital** below minimum
- Bank **bounced cheques** or low average balance
- Mobile lender **loan stacking** (too many active digital loans) or **excessive rollovers**

### Scorecard formula

Credit scores use the industry-standard **PDO (Points to Double Odds)** method:

\[
\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{odds})
\]

where `Factor = PDO / ln(2)`, `Offset = base_score − Factor × ln(base_odds)`, and `odds = PD / (1 − PD)`.

Current settings (`config/scoring.yaml`):

| Setting | Value |
|---------|-------|
| Base score | 680 |
| Base odds | 50:1 |
| PDO | 20 |
| Score range | 300–850 |
| Approve band | ≥ 680 |
| Review band | ≥ 550 |

---

## SHAP explainability & audit trails

Every score can include a **SHAP (SHapley Additive exPlanations)** breakdown showing which features pushed default probability up or down. This supports:

- **CBK model risk management** guidelines
- **Consumer disclosure** (why was my application declined?)
- **Internal audit** and fair-lending reviews

### How SHAP is used here

1. Applicant features are preprocessed through the same pipeline used at training time.
2. `TreeExplainer` computes Shapley values on the underlying gradient boosting model.
3. Top contributions are ranked by absolute impact.
4. A plain-language `summary` is generated for non-technical reviewers.

| `shap_value` sign | Meaning |
|-------------------|---------|
| **Positive** | Feature **increases** default risk |
| **Negative** | Feature **decreases** default risk |

### Audit trail format

When `persist_audit_trail: true`, a JSON file is written to `assets/audit_trails/{audit_id}.json`:

| Field | Purpose |
|-------|---------|
| `audit_id` | Unique ID for compliance lookups (e.g. `MPESA-001-86f743a6`) |
| `scored_at` | UTC timestamp of the decision |
| `model_version` | Model version that produced the score |
| `applicant_id` | Applicant identifier |
| `channel` | Lending channel (`mpesa`, `sacco`, `bank`, `mobile_lender`) |
| `probability_of_default` | Final PD used for scorecard |
| `credit_score` | Final scorecard score |
| `decision` | Final decision (`approve` / `review` / `decline`) |
| `policy_passed` | Whether policy rules passed |
| `policy_reasons` | Hard-rule decline reasons (independent of ML) |
| `shap.contributions` | Top feature impacts with raw values |
| `shap.summary` | Plain-language explanation |
| `request_snapshot` | Input payload used for the decision |

---

## Configuration

All thresholds live in `config/scoring.yaml` — no magic numbers in code.

```yaml
data:
  n_samples: 8000
  default_rate: 0.12
  channel_distribution:
    mpesa: 0.30
    mobile_lender: 0.25
    sacco: 0.25
    bank: 0.20

channel_minimums:
  mobile_lender:
    min_platform_tenure_months: 1
    min_platform_repayment_rate: 0.80
    max_active_digital_loans: 2
    max_rollover_count_12m: 3
```

```yaml
# Other key sections
scorecard:          # PDO scorecard settings
decision_bands:     # approve / review thresholds
policy:             # Global rules (age, DTI, CRB, income)
channel_minimums:   # Per-channel rules (all four channels)
training:           # Model type, calibration, test split
```

| Section | What to tune |
|---------|--------------|
| `data.channel_distribution` | Mix: 30% M-Pesa, 25% mobile lender, 25% SACCO, 20% bank |
| `scorecard` | Base score, PDO, min/max score bounds |
| `decision_bands` | Approve (≥ 680) and review (≥ 550) thresholds |
| `policy` | Minimum age (18), max DTI (45%), CRB defaults (0), income floor (KES 15,000) |
| `channel_minimums.mpesa` | KYC tier, Fuliza cap, wallet activity |
| `channel_minimums.sacco` | Membership months, share capital, repayment rate |
| `channel_minimums.bank` | Account age, bounced cheques, min balance |
| `channel_minimums.mobile_lender` | Platform tenure, repayment rate, loan stacking, rollovers |
| `training.model_type` | `gradient_boosting` or `logistic_regression` |

After changing config or adding channels, re-run `python3 train.py` to retrain (feature count must match saved model).

---

## Key metrics

| Metric | Meaning | Latest value |
|--------|---------|--------------|
| **ROC-AUC** | Ranking quality of default vs non-default | 0.909 |
| **Gini** | `2 × AUC − 1`; higher = better separation | 0.818 |
| **KS** | Maximum separation between score distributions | 0.815 |
| **Brier** | Calibration error of predicted probabilities | 0.058 |
| **Precision** | Of predicted defaults, how many are actual defaults | 0.573 |
| **Recall** | Of actual defaults, how many the model catches | 0.734 |
| **Features** | Total model input columns | 43 |

---

## Design decisions

1. **Separation of ML and policy** — the model estimates risk; hard rules enforce compliance independently. A high score cannot override an active CRB default.
2. **Channel masking** — one unified model serves M-Pesa, SACCO, bank, and mobile digital lenders by zeroing irrelevant channel features per row.
3. **Calibrated probabilities** — `CalibratedClassifierCV` (sigmoid) improves PD reliability before scorecard mapping.
4. **Versioned artifacts** — each training run saves a timestamped model, metadata JSON, and feature importance CSV.
5. **SHAP audit trails** — every API or CLI score can persist a JSON audit file with feature-level explanations.
6. **FastAPI + Pydantic** — typed request validation per channel; invalid payloads are rejected before scoring.
7. **Synthetic data only** — no real customer PII. Replace `src/data/synthetic.py` with your ETL/CRB pipeline in production.

---

## Roadmap

### Built ✅

- Multi-channel feature engineering (M-Pesa, SACCO, bank, **mobile digital lenders**)
- Training pipeline with Gini, KS, Brier evaluation
- Scorecard conversion (300–850, PDO)
- Policy engine with channel-specific rules
- SHAP explainability with audit trail persistence
- FastAPI REST service with Swagger docs
- CLI scoring with **six sample applicants** (all four channels + two high-risk cases)

### Next steps

- Connect to CRB APIs (Metropol, TransUnion Kenya) for live bureau data
- Replace synthetic generator with feature store / data warehouse pipelines
- MLflow model registry and A/B testing between model versions
- API authentication (OAuth2 / API keys) and rate limiting
- Monitor PSI (Population Stability Index) and recalibrate quarterly
- Deploy to cloud (Docker + Kubernetes or managed container service)

---

## Author

**Duncan Mwirigi**  
GitHub: [github.com/duncanmwirigi](https://github.com/duncanmwirigi)  
X: https://x.com/AIStiqDan  
Website: https://bytecityinc.com

## License

MIT — use and modify freely for learning and projects.
