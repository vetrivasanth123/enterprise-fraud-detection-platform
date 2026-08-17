import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

# Project structure:
#
# enterprise-fraud-detection-platform/
# ├── dashboard/
# │   └── app.py
# ├── src/
# │   ├── fraud_engine.py
# │   └── explainability.py
# └── artifacts/
#     ├── xgboost_model.pkl
#     ├── scaler.pkl
#     └── threshold_config.json
#
# app.py is inside dashboard/, so project root is one level up.

ROOT_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = ROOT_DIR / "src"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.fraud_engine import FraudDecisionEngine
from src.explainability import FraudExplainabilityEngine


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Enterprise Fraud Risk Platform",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Enterprise Fraud Detection & Risk Triage Platform")

st.caption(
    "Real-Time Transaction Scoring, Multi-Tier Risk Allocation "
    "& SHAP Explainability"
)


# ============================================================
# LOAD MODEL / ENGINE / CONFIG
# ============================================================

@st.cache_resource
def load_resources():

    # Use absolute paths so execution location does not matter.
    model_path = ARTIFACTS_DIR / "xgboost_model.pkl"
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    config_path = ARTIFACTS_DIR / "threshold_config.json"

    engine = FraudDecisionEngine(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        config_path=str(config_path),
    )

    explainer = FraudExplainabilityEngine(
        model_path=str(model_path)
    )

    with open(config_path, "r") as f:
        config = json.load(f)

    return engine, explainer, config


try:

    engine, explainer, config = load_resources()

except Exception as e:

    st.error(
        f"Error loading model artifacts: {e}"
    )

    st.stop()


# ============================================================
# PROJECT DECISION BANDS
# ============================================================

# These are the same policy bands already implemented in
# src/fraud_engine.py and used in the project.

PASS_THRESHOLD = 0.20
TWO_FA_THRESHOLD = 0.60
REVIEW_THRESHOLD = 0.64


def decision_from_probability(probability):

    probability = float(probability)

    if probability < PASS_THRESHOLD:

        return "LOW RISK", "PASS"

    elif probability < TWO_FA_THRESHOLD:

        return "MEDIUM RISK", "STEP-UP 2FA"

    elif probability < REVIEW_THRESHOLD:

        return "HIGH RISK", "MANUAL REVIEW"

    else:

        return "CRITICAL RISK", "BLOCK"


# ============================================================
# REQUIRED RAW FEATURES
# ============================================================

REQUIRED_COLUMNS = (
    ["Time", "Amount"]
    + [f"V{i}" for i in range(1, 29)]
)


# ============================================================
# FIND EXISTING TRANSACTION DATA
# ============================================================

@st.cache_data
def find_transaction_dataset():

    search_roots = [
        ROOT_DIR / "data",
        ROOT_DIR / "datasets",
        ROOT_DIR / "artifacts",
        ROOT_DIR,
    ]

    files = []

    for root in search_roots:

        if not root.exists():
            continue

        try:

            files.extend(root.rglob("*.csv"))
            files.extend(root.rglob("*.parquet"))

        except Exception:
            pass

    # Remove duplicates
    unique_files = []
    seen = set()

    for file_path in files:

        try:

            resolved = file_path.resolve()

            if resolved not in seen:

                seen.add(resolved)
                unique_files.append(resolved)

        except Exception:

            pass

    # Prefer files that look like holdout/test data.
    unique_files.sort(
        key=lambda p: (
            0
            if any(
                word in p.name.lower()
                for word in [
                    "holdout",
                    "test",
                    "validation",
                    "val",
                ]
            )
            else 1,
            len(str(p)),
        )
    )

    for file_path in unique_files:

        try:

            if file_path.suffix.lower() == ".csv":

                df = pd.read_csv(file_path)

            elif file_path.suffix.lower() == ".parquet":

                df = pd.read_parquet(file_path)

            else:

                continue

            if all(
                column in df.columns
                for column in REQUIRED_COLUMNS
            ):

                return df, file_path

        except Exception:

            continue

    return None, None


# ============================================================
# FIND ONE REAL TRANSACTION FOR EACH DECISION TIER
# ============================================================

@st.cache_data
def find_representative_transactions():

    df, source_path = find_transaction_dataset()

    if df is None:

        return {}, None

    # Keep only columns actually required by the deployed
    # FraudDecisionEngine.
    raw_df = df[REQUIRED_COLUMNS].copy()

    raw_df = raw_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    raw_df = raw_df.dropna(
        subset=REQUIRED_COLUMNS
    )

    representatives = {}

    # --------------------------------------------------------
    # Every candidate is scored using the REAL deployed model.
    # Nothing is manually assigned.
    # --------------------------------------------------------

    for idx in raw_df.index:

        row = raw_df.loc[[idx]].copy()

        try:

            result = engine.predict_transaction(row)

            action = result["action"]

            if action not in representatives:

                representatives[action] = {
                    "payload": row.iloc[0].to_dict(),
                    "result": result,
                    "index": idx,
                }

            # Stop after finding all four actual decision tiers.
            if all(
                action_name in representatives
                for action_name in [
                    "PASS",
                    "STEP-UP 2FA",
                    "MANUAL REVIEW",
                    "BLOCK",
                ]
            ):

                break

        except Exception:

            continue

    return representatives, source_path


representatives, data_source = (
    find_representative_transactions()
)


# ============================================================
# 1. PORTFOLIO & SYSTEM OVERVIEW
# ============================================================

st.header("1. Portfolio & System Overview")

col1, col2, col3, col4 = st.columns(4)

precision_val = config.get(
    "holdout_precision",
    0.8690,
)

pr_auc_val = config.get(
    "holdout_pr_auc",
    0.8529,
)

threshold_val = config.get(
    "optimal_threshold",
    0.64,
)

loss_val = config.get(
    "holdout_loss",
    3652.98,
)


col1.metric(
    "Holdout Precision",
    f"{precision_val * 100:.1f}%",
    "+7.7% vs Default",
)

col2.metric(
    "Holdout PR-AUC",
    f"{pr_auc_val:.4f}",
    "+0.0175 Lift",
)

col3.metric(
    "Optimal Threshold",
    f"{threshold_val:.2f}",
    "Cost Minimum",
)

col4.metric(
    "Portfolio Loss",
    f"${loss_val:,.2f}",
    "-$288.86 Saved",
)

st.divider()


# ============================================================
# 2. REAL-TIME TRANSACTION SCORING
# ============================================================

st.header(
    "2. Real-Time Transaction Scoring & Risk Triage"
)

col_left, col_right = st.columns(
    [1, 1]
)


# ============================================================
# LEFT COLUMN
# ============================================================

with col_left:

    st.subheader("Transaction Inputs")

    demo_scenario = st.selectbox(
        "Load Scenario Preset",
        [
            "Custom Input",
            "🟢 Standard Pass (Low Risk)",
            "🟡 Step-Up 2FA (Medium Risk)",
            "🟠 Manual Review (Borderline Queue)",
            "🔴 Blocked Fraud (Critical Risk)",
        ],
    )


    # ========================================================
    # CUSTOM INPUT
    # ========================================================

    if demo_scenario == "Custom Input":

        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            max_value=100000.0,
            value=15.0,
            step=5.0,
        )

        time_val = st.number_input(
            "Transaction Time (Seconds)",
            min_value=0.0,
            max_value=172800.0,
            value=400.0,
            step=100.0,
        )

        st.markdown(
            "**PCA Anomaly Signals (V1 - V28)**"
        )

        v14 = st.slider(
            "V14 (Primary Risk Driver)",
            -20.0,
            10.0,
            0.0,
            0.5,
        )

        v10 = st.slider(
            "V10 (Secondary Anomaly Flag)",
            -20.0,
            10.0,
            0.0,
            0.5,
        )

        v4 = st.slider(
            "V4 (Transaction Intent Correlation)",
            -10.0,
            10.0,
            0.0,
            0.5,
        )

        payload = {
            "Time": float(time_val),
            "Amount": float(amount),
        }

        for i in range(1, 29):

            payload[f"V{i}"] = 0.0

        payload["V14"] = float(v14)
        payload["V10"] = float(v10)
        payload["V4"] = float(v4)

        selected_preset_action = None


    # ========================================================
    # REAL REPRESENTATIVE TRANSACTION PRESET
    # ========================================================

    else:

        preset_action_map = {

            "🟢 Standard Pass (Low Risk)":
                "PASS",

            "🟡 Step-Up 2FA (Medium Risk)":
                "STEP-UP 2FA",

            "🟠 Manual Review (Borderline Queue)":
                "MANUAL REVIEW",

            "🔴 Blocked Fraud (Critical Risk)":
                "BLOCK",
        }

        selected_preset_action = (
            preset_action_map[demo_scenario]
        )

        preset = representatives.get(
            selected_preset_action
        )

        if preset is None:

            st.error(
                f"No real transaction from the available "
                f"dataset was found for: "
                f"{selected_preset_action}"
            )

            st.info(
                "Please check that the original transaction "
                "CSV/Parquet containing Time, Amount and V1-V28 "
                "is present in the project."
            )

            payload = None

        else:

            payload = preset["payload"]

            preset_result = preset["result"]

            st.caption(
                "Preset uses an actual transaction evaluated "
                "through the deployed fraud model."
            )

            st.metric(
                "Actual Model Probability",
                f"{float(preset_result['probability']) * 100:.3f}%",
            )

            st.number_input(
                "Transaction Amount ($)",
                value=float(payload["Amount"]),
                disabled=True,
            )

            st.number_input(
                "Transaction Time (Seconds)",
                value=float(payload["Time"]),
                disabled=True,
            )

            st.markdown(
                "**PCA Anomaly Signals (V1 - V28)**"
            )

            st.slider(
                "V14 (Primary Risk Driver)",
                -20.0,
                10.0,
                float(payload["V14"]),
                0.5,
                disabled=True,
            )

            st.slider(
                "V10 (Secondary Anomaly Flag)",
                -20.0,
                10.0,
                float(payload["V10"]),
                0.5,
                disabled=True,
            )

            st.slider(
                "V4 (Transaction Intent Correlation)",
                -10.0,
                10.0,
                float(payload["V4"]),
                0.5,
                disabled=True,
            )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col_right:

    st.subheader("Decision Engine Output")

    evaluate = st.button(
        "Evaluate Transaction Payload",
        type="primary",
    )


    if evaluate:

        if payload is None:

            st.error(
                "No transaction payload is available."
            )

        else:

            # ------------------------------------------------
            # REAL MODEL EVALUATION
            # ------------------------------------------------

            result = engine.predict_transaction(
                pd.DataFrame([payload])
            )

            probability = float(
                result["probability"]
            )

            engine_tier = result["risk_tier"]
            engine_action = result["action"]


            # ------------------------------------------------
            # RE-CALCULATE THE POLICY BAND FROM THE ACTUAL
            # MODEL PROBABILITY.
            #
            # This is only a consistency check. We do NOT
            # overwrite the model probability.
            # ------------------------------------------------

            expected_tier, expected_action = (
                decision_from_probability(
                    probability
                )
            )


            # ------------------------------------------------
            # ENGINE / POLICY CONSISTENCY
            # ------------------------------------------------

            if (
                engine_tier != expected_tier
                or engine_action != expected_action
            ):

                st.error(
                    "Decision engine and probability-band "
                    "logic are inconsistent."
                )

                st.write(
                    {
                        "Probability": probability,
                        "Engine Tier": engine_tier,
                        "Engine Action": engine_action,
                        "Expected Tier": expected_tier,
                        "Expected Action": expected_action,
                    }
                )

            else:

                # ------------------------------------------------
                # PRESET CONSISTENCY
                # ------------------------------------------------

                if (
                    selected_preset_action is not None
                    and engine_action
                    != selected_preset_action
                ):

                    st.error(
                        f"The selected preset was intended for "
                        f"{selected_preset_action}, but the actual "
                        f"model returned {engine_action}."
                    )

                    st.info(
                        "The dashboard will not fabricate or "
                        "override the model result."
                    )

                else:

                    # ============================================
                    # RISK ACTION DISPLAY
                    # ============================================

                    if engine_action == "PASS":

                        st.success(
                            f"**Action: PASS** | "
                            f"Tier: {engine_tier}"
                        )

                    elif engine_action == "STEP-UP 2FA":

                        st.warning(
                            f"**Action: STEP-UP 2FA** | "
                            f"Tier: {engine_tier}"
                        )

                    elif engine_action == "MANUAL REVIEW":

                        st.warning(
                            f"**Action: MANUAL REVIEW** | "
                            f"Tier: {engine_tier}"
                        )

                    elif engine_action == "BLOCK":

                        st.error(
                            f"**Action: BLOCK** | "
                            f"Tier: {engine_tier}"
                        )


                    # ============================================
                    # PROBABILITY
                    # ============================================

                    st.metric(
                        "Calibrated Fraud Probability (PD)",
                        f"{probability * 100:.3f}%",
                    )

                    st.progress(
                        min(
                            max(probability, 0.0),
                            1.0,
                        )
                    )


                    # ============================================
                    # RISK BAND DESCRIPTION
                    # ============================================

                    if probability < 0.20:

                        st.caption(
                            "Risk band: P < 0.20 → PASS"
                        )

                    elif probability < 0.60:

                        st.caption(
                            "Risk band: 0.20 ≤ P < 0.60 "
                            "→ STEP-UP 2FA"
                        )

                    elif probability < 0.64:

                        st.caption(
                            "Risk band: 0.60 ≤ P < 0.64 "
                            "→ MANUAL REVIEW"
                        )

                    else:

                        st.caption(
                            "Risk band: P ≥ 0.64 → BLOCK"
                        )


                    # ============================================
                    # SHAP WATERFALL
                    # ============================================

                    st.subheader(
                        "Transaction SHAP Explainability"
                    )

                    try:

                        df_processed = (
                            engine.preprocess(
                                pd.DataFrame(
                                    [payload]
                                )
                            )
                        )

                        fig = (
                            explainer
                            .plot_local_waterfall(
                                df_processed
                            )
                        )

                        st.pyplot(
                            fig,
                            clear_figure=True,
                        )

                    except Exception as e:

                        st.error(
                            f"SHAP waterfall could not be "
                            f"generated: {e}"
                        )


# ============================================================
# FOOTER / DATA SOURCE
# ============================================================

if data_source is not None:

    st.caption(
        "Scenario presets are based on genuine transactions "
        "from the available project dataset and are scored "
        "using the deployed inference engine."
    )
