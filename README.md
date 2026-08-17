# Enterprise Fraud Detection & Risk Triage Platform

An end-to-end, production-ready financial fraud detection system that converts raw transaction telemetry into calibrated fraud probabilities, operational risk triage decisions (**PASS**, **STEP-UP 2FA**, **MANUAL REVIEW**, **BLOCK**), and dollar-weighted cost-minimization analysis under extreme class imbalance (~0.17% fraud rate).

---

## Executive Summary

This repository implements an enterprise transaction fraud triage pipeline using a **Tuned XGBoost Classifier** paired with **Isotonic Probability Calibration**. Operating under extreme class imbalance (1 fraud case per 571 legitimate transactions), the system achieves an untouched holdout **PR-AUC of 0.8529**, **ROC-AUC of 0.9844**, and a calibrated **Brier Score of 0.00037** across 56,898 holdout test transactions.

Rather than relying on default count-based decision thresholds (e.g., 0.50 or 0.90), operational decision boundaries are optimized using a **Dollar-Weighted Cost Matrix** trading off missed fraud amounts against a $10.00 manual investigation fee. Capping total holdout portfolio loss at **$3,652.98** at an optimal threshold of **0.64**, the system delivers a **$288.86 net financial saving** compared to standard operational baselines.

---

## Key Performance Indicators & Benchmark Summary

### Model Family Benchmarks (Untouched Holdout N = 56,898)

| Model Architecture | Holdout PR-AUC | Holdout ROC-AUC | Precision | Recall | Key Operational Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.6995 | 0.9749 | 0.0572 | 0.8983 | High false alarm rate (874 False Positives) |
| **Random Forest Ensemble** | 0.7887 | 0.9247 | 0.8750 | 0.3120 | Strong precision, missed low-value fraud |
| **LightGBM Classifier** | 0.4225 | 0.8912 | 0.7119 | 0.6510 | Leaf-growth collapse under raw class weighting |
| **CatBoost Classifier** | 0.8306 | 0.9810 | 0.8421 | 0.7820 | Stable native imbalance handling |
| **Tuned XGBoost Champion** | **0.8529** | **0.9844** | **0.8690** | **0.8202** | **Optimal balance scaling (√571 positive weight)** |

### Baseline vs. Champion System Impact

| Metric / Parameter | Baseline XGBoost (Phase 13) | Tuned Champion (Phase 19) | Technical & Business Impact |
| :--- | :---: | :---: | :--- |
| **Model Architecture** | Default XGBoost | Tuned XGBoost (`scale_pos_weight = 23.9`) | Eliminates probability score collapse |
| **Feature Pipeline** | 30 raw features | 34 processed features | +4 engineered behavioral flags |
| **Untouched Holdout PR-AUC** | 0.8354 | **0.8529** | +0.0175 out-of-sample lift |
| **Untouched Holdout ROC-AUC**| 0.9838 | **0.9844** | High class separation stability |
| **Calibrated Brier Score** | 0.00039 | **0.00037** | Enhanced probability reliability |
| **Optimal Decision Boundary**| 0.90 (Count-based) | **0.64 (Dollar-Weighted)** | Aligned with portfolio loss & P&L |
| **Holdout Portfolio Loss** | $3,941.84 | **$3,652.98** | **Net saving of $288.86 on holdout evaluation** |
| **Governance Audit** | Partial Audit | **100% EXCELLENT** | 8/8 production stress tests passed |

---

## System Architecture & Pipeline Flow

```text
RAW INBOUND TRANSACTION TELEMETRY (V1–V28, Time, Amount)
   │
   ├── 1. Feature Engineering Engine (30 ──► 34 Model Features)
   │      ├── Hour_Of_Day  : (Time / 3600) mod 24 (Daily transaction rhythms)
   │      ├── Is_Night_Tx  : Binary flag for 00:00 - 06:00 window (High-risk time slot)
   │      ├── Log_Amount   : log(1 + Amount) (Stabilizes heavy right-skewed amounts)
   │      └── Is_Micro_Tx  : Binary flag for Amount <= $1.00 (Automated card-testing)
   │
   ├── 2. Preprocessing & Scaling
   │      └── StandardScaler (Fitted strictly on Training set; zero data leakage)
   │
   ├── 3. Tuned XGBoost Inference Engine (scale_pos_weight = 23.9)
   │
   ├── 4. Isotonic Probability Calibration
   │      └── Raw score ──► Calibrated Probability of Default (PD)
   │
   ├── 5. Dollar-Weighted Cost Matrix & 4-Tier Operational Triage
   │      ├── P < 0.20          ──► LOW RISK      ──► Action: PASS (Auto Settlement)
   │      ├── 0.20 <= P < 0.60  ──► MEDIUM RISK   ──► Action: STEP-UP 2FA
   │      ├── 0.60 <= P < 0.64  ──► HIGH RISK     ──► Action: MANUAL REVIEW QUEUE
   │      └── P >= 0.64         ──► CRITICAL RISK ──► Action: BLOCK (Fraud Alert)
   │
   └── 6. Audit & Explainability Layer
          ├── Global Feature Risk Drivers (SHAP TreeExplainer)
          └── Local Log-Odds Waterfall Breakdown (Per-transaction compliance)
```

---

## Operational Triage & Financial Cost Optimization

The platform optimizes decision boundaries using a **Dollar-Weighted Cost Matrix**:

> **Total Portfolio Loss = Σ (Missed Fraud Amounts) + (False Positive Count × $10.00 Analyst Investigation Fee)**

### Holdout Risk Segmentation & Action Routing (N = 56,898)

| Risk Tier | Probability Range | Operational Action | Transaction Count | Volume Share |
| :--- | :---: | :---: | :---: | :---: |
| **Low Risk** | P < 0.20 | **PASS** | 56,798 | 99.824% |
| **Medium Risk** | 0.20 ≤ P < 0.60 | **STEP-UP 2FA** | 18 | 0.032% |
| **High Risk** | 0.60 ≤ P < 0.64 | **MANUAL REVIEW** | 12 | 0.021% |
| **Critical Risk** | P ≥ 0.64 | **BLOCK** | 70 | 0.123% |

---

## Explainability (SHAP Audit)

Global feature importance evaluated using `TreeExplainer` on validation sets identifies key fraud risk drivers:
* **V14** (Contribution: `+6.0290`): Strongest driver of high-risk fraud anomalies.
* **V10** (Contribution: `+1.2102`): Secondary structural anomaly flag.
* **V4** (Contribution: `+0.9497`): Positive correlation with elevated transaction intent risk.
* **V12** (Contribution: `+0.8432`): Key discriminator for card-testing patterns.
* **V17** (Contribution: `+0.7750`): Structural behavioral risk indicator.

Every high-risk score automatically generates a transaction-level **local SHAP waterfall decomposition** for compliance and fraud analyst verification.

---

## Repository Structure

```text
enterprise-fraud-detection-platform/
│
├── artifacts/
│   ├── logistic_model.pkl          # Trained Logistic Regression baseline
│   ├── random_forest_model.pkl     # Trained Random Forest ensemble model
│   ├── scaler.pkl                  # StandardScaler fitted strictly on training data
│   ├── threshold_config.json       # Operational threshold (0.64) & loss metadata
│   └── xgboost_model.pkl           # Final Tuned Champion XGBoost model
│
├── dashboard/
│   └── app.py                      # Interactive Streamlit fraud triage application
│
├── notebooks/
│   └── enterprise_fraud_detection_platform.ipynb  # End-to-end model development notebook
│
├── reports/
│   └── Enterprise_Fraud_Detection_Risk_Triage_Platform_Report.pdf # Comprehensive technical report
│
├── src/
│   ├── explainability.py           # Local & global SHAP waterfall feature breakdown
│   └── fraud_engine.py             # Feature engineering pipeline & 4-tier decision engine
│
├── .gitignore                      # Git ignore file
├── README.md                       # Master platform documentation
└── requirements.txt                # Python environment dependencies
```

---

## Production Stress & Edge-Case Validation

The system's inference engine passed 8/8 automated production verification checks:

1. **Single-Customer Real-Time Payload Scoring:** Validated low probability (`0.00005`), routed correctly to `PASS`.
2. **1,000-Transaction Batch Execution:** 999 routed to `PASS`, 1 routed to `BLOCK`.
3. **Missing Value (NaN) Handling:** Automated median imputation executed without failure.
4. **Extreme Amount Input ($1,000,000.00):** Scaler bounds processed without numerical overflow.
5. **Unseen Categorical Metadata:** Gracefully handled without runtime failure.
6. **Deterministic Parity:** Bit-for-bit probability reproduction verified across repeated runs.
7. **Threshold Boundary Enforcement (P = 0.6400):** Exactly mapped to `BLOCK`.
8. **Vector Feature Protection (30 vs 34):** Automated engineering layer engages seamlessly when receiving raw 30-feature vectors.

---

## Quickstart & Local Setup

### 1. Clone Repository & Install Dependencies

```bash
git clone [https://github.com/vetrivasanth123/enterprise-fraud-detection-platform.git](https://github.com/vetrivasanth123/enterprise-fraud-detection-platform.git)
cd enterprise-fraud-detection-platform
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch Streamlit Triage Dashboard

```bash
streamlit run dashboard/app.py
```

### 3. Run Inference via Python Engine

```python
from src.fraud_engine import FraudDecisionEngine

engine = FraudDecisionEngine(
    model_path='artifacts/xgboost_model.pkl',
    scaler_path='artifacts/scaler.pkl',
    config_path='artifacts/threshold_config.json'
)

# Example payload with 30 raw features
sample_payload = {
    'Time': 406.0,
    'Amount': 149.62,
    'V1': -1.3598, 'V2': -0.0727, 'V3': 2.5363, 'V4': 1.3782,
    # ... V5 through V28 ...
}

result = engine.predict_transaction(sample_payload)
print(result)
# Output: {'probability': 0.00034, 'risk_tier': 'LOW RISK', 'action': 'PASS'}
```

---

## Governance & Compliance

* **Performance Score:** Verified (Holdout PR-AUC 0.8529)
* **Generalization Score:** Verified (Zero train-test overfitting gap)
* **Decision Logic Audit:** Active (4-tier mapping enforced)
* **Business Impact Audit:** Minimized ($3,652.98 total portfolio risk loss)
* **Artifact Integrity:** Locked & Persisted (`artifacts/`)
* **Governance Completeness Score:** **100% EXCELLENT**
