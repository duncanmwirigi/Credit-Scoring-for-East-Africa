# East Africa Credit Scoring Engine

Credit scoring for **M-Pesa**, **SACCO**, and **bank** lending channels. The system combines channel-specific feature engineering, a calibrated ML model, scorecard conversion, and deterministic policy rules — the same layered pattern used in mature fintech and banking stacks.

## Architecture

```mermaid
flowchart LR
    A[Applicant Data] --> B[Feature Engineering]
    B --> C[ML Model]
    C --> D[Scorecard 300-850]
    D --> E[Policy Engine]
    E --> F[Approve / Review / Decline]
```

| Layer | Responsibility |
|-------|----------------|
| **Feature engineering** | Channel-aware variables (M-Pesa wallet, SACCO membership, bank statements) |
| **ML model** | Probability of default (PD) with calibration |
| **Scorecard** | PD → credit score (300–850) using PDO methodology |
| **Policy engine** | Hard rules: CRB defaults, DTI caps, channel minimums |

## Channel coverage

### M-Pesa (mobile money)
- KYC tier, wallet activity, transaction velocity
- Cash-in/cash-out ratio, merchant spend mix
- Fuliza/overdraft utilization, balance volatility

### SACCO (co-operative lending)
- Membership tenure, share capital, savings consistency
- Prior loan repayment within the SACCO
- Guarantor network strength, dividend history

### Bank (traditional lending)
- Account age, average balance, salary deposit regularity
- Bounced cheques, overdraft and card utilization
- Existing loan exposure and branch relationship

## Project structure

```
credit_scoring/
├── config/
│   └── scoring.yaml           # Thresholds, scorecard, channel rules
├── src/
│   ├── config.py              # Typed configuration loader
│   ├── domain.py              # Applicant, decision, policy models
│   ├── data/
│   │   └── synthetic.py       # Realistic synthetic portfolio generator
│   ├── features/
│   │   └── engineering.py     # Feature matrix builder
│   ├── ml/
│   │   ├── trainer.py         # Train, evaluate, persist model
│   │   ├── scorer.py          # Score applicants, scorecard mapping
│   │   └── metrics.py         # AUC, Gini, KS, Brier
│   └── policy/
│       └── engine.py          # Business & regulatory rules
├── train.py                   # Train pipeline entry point
├── score.py                   # Score sample applicants
├── models/                    # Saved models (generated)
├── assets/                    # Reports and plots (generated)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- scikit-learn, pandas, numpy, matplotlib, pyyaml, joblib

```bash
pip install -r requirements.txt
```

## How to run

### 1. Train the model

```bash
python3 train.py
```

This will:
1. Generate an 8,000-applicant synthetic portfolio (45% M-Pesa, 30% SACCO, 25% bank).
2. Train a calibrated gradient boosting classifier.
3. Report **ROC-AUC**, **Gini**, and **KS** statistics.
4. Save the model to `models/` and reports to `assets/`.

### 2. Score applicants

```bash
python3 score.py
```

Scores four sample applicants (M-Pesa, SACCO, bank, and a high-risk M-Pesa case) and writes results to `assets/sample_decisions.json`.

Example output:

```
MPESA-001     | channel=mpesa | score=687 | pd=1.54% | decision=APPROVE
SACCO-014     | channel=sacco | score=686 | pd=1.62% | decision=APPROVE
BANK-203      | channel=bank  | score=686 | pd=1.57% | decision=APPROVE
MPESA-RISK-77 | channel=mpesa | score=546 | pd=67.76% | decision=DECLINE
```

Training typically achieves **ROC-AUC ~0.89**, **Gini ~0.79**, and **KS ~0.80** on the synthetic portfolio.

## Configuration

Edit `config/scoring.yaml` to tune:

- **Scorecard**: base score (680), PDO (20), min/max bounds
- **Decision bands**: approve ≥ 680, review ≥ 550
- **Policy**: minimum age, max DTI, CRB rules, income floor
- **Channel minimums**: KYC tier, SACCO share capital, bank balance thresholds

## Key metrics

| Metric | Meaning |
|--------|---------|
| **ROC-AUC** | Ranking quality of default vs non-default |
| **Gini** | `2 × AUC − 1`; higher = better separation |
| **KS** | Maximum separation between good/bad distributions |
| **Brier** | Calibration error of predicted probabilities |

## Design decisions (production patterns)

1. **Separation of ML and policy** — the model estimates risk; hard rules enforce compliance independently.
2. **Channel masking** — SACCO features are zeroed for M-Pesa rows so one unified model serves all channels.
3. **Calibrated probabilities** — `CalibratedClassifierCV` improves PD reliability for scorecard mapping.
4. **Versioned artifacts** — each training run saves a timestamped model, metadata JSON, and feature importance.
5. **Synthetic data only** — no real customer PII; replace `synthetic.py` with your ETL in production.

## Production next steps

- Connect to CRB APIs (Metropol, TransUnion) for live bureau data.
- Replace synthetic generator with feature store / data warehouse pipelines.
- Add SHAP or LIME for regulatory explainability.
- Deploy scorer as a REST API (FastAPI) with model registry (MLflow).
- Monitor PSI (Population Stability Index) and recalibrate quarterly.

## Author

**Duncan Mwirigi**  
GitHub: [github.com/duncanmwirigi](https://github.com/duncanmwirigi)  
X: https://x.com/AIStiqDan  
Website: https://bytecityinc.com

## License

MIT — use and modify freely for learning and projects.
