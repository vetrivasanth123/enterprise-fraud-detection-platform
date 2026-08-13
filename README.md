# Enterprise Fraud Detection & Risk Triage Platform
> **End-to-End Machine Learning System for Real-Time Financial Fraud Risk Scoring**

---

## Executive Summary
This repository implements an enterprise-grade transaction fraud triage platform designed to handle extreme class imbalance (**0.17% fraud rate**, or 1 fraud per 571 legitimate transactions). The system combines a **Tuned XGBoost Classifier** with **Isotonic Probability Calibration**, achieving a holdout **PR-AUC of 0.8529** and a **Brier Score of 0.00037**.

Rather than using default count-based cutoffs, decision boundaries are optimized using a **Dollar-Weighted Cost Matrix** ($10 analyst review fee vs. fraud dollar loss). The system locks an optimal operational threshold of **0.64**, capping total portfolio risk loss at **$3,652.98** (saving $288.86 over default deployment).

---

## Key Performance Indicators & Metrics

| System Metric | Baseline Model (Phase 13) | Tuned Champion Model (Phase 19) | Business / Technical Impact |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | Default XGBoost | Tuned XGBoost ($\sqrt{\text{imbalance}}$ weight) | Prevents probability score collapse |
| **Feature Pipeline** | 30 raw features | 34 processed features | +4 behavioral timing & amount flags |
| **Untouched Holdout PR-AUC** | 0.8354 | **0.8529** | **+0.0175 Out-of-sample lift** |
| **Untouched Holdout ROC-AUC** | 0.9838 | **0.9844** | Excellent class separation |
| **Calibrated Brier Score** | 0.00039 | **0.00037** | Lower score = higher probability reliability |
| **Optimal Decision Boundary** | 0.90 (Count-based) | **0.64 (Dollar-Weighted)** | Aligned with business P&L |
| **Total Portfolio Financial Loss** | $3,941.84 | **$3,652.98** | **Saved $288.86 on holdout evaluation** |
| **Governance & Stress Testing** | Partial Audit | **100% EXCELLENT** | 8/8 Production stress tests passed |

---

## System Architecture & Pipeline Flow

```text
RAW INBOUND TRANSACTION (V1-V28, Time, Amount)
   │
   ├── 1. Feature Engineering Engine (30 -> 34 Features)
   │      ├── Hour_Of_Day  : Cyclical daily transaction rhythm
   │      ├── Is_Night_Tx  : High-risk time window flag (00:00 - 06:00)
   │      ├── Log_Amount   : Log-transformed transaction value
   │      └── Is_Micro_Tx  : Micro-charge card testing flag (<= $1.00)
   │
   ├── 2. Preprocessing & Scaling (StandardScaler fitted on train only)
   │
   ├── 3. Tuned XGBoost Inference Engine (scale_pos_weight = 23.9)
   │
   ├── 4. Isotonic Probability Calibration (Raw Score -> Calibrated PD)
   │
   ├── 5. Dollar-Weighted Cost Matrix Triage (Threshold = 0.64)
   │      ├── P < 0.20          --> LOW RISK        --> Action: PASS
   │      ├── 0.20 <= P < 0.60  --> MEDIUM RISK     --> Action: STEP-UP 2FA
   │      ├── 0.60 <= P < 0.64  --> HIGH RISK       --> Action: MANUAL REVIEW
   │      └── P >= 0.64          --> CRITICAL RISK   --> Action: FRAUD BLOCK
   │
   └── 6. Explainability & Governance Layer
          ├── Global Feature Importance (SHAP TreeExplainer)
          └── Local Log-Odds Waterfall Breakdown (Per Transaction)
