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
# ├── artifacts/
# │   ├── xgboost_model.pkl
# │   ├── scaler.pkl
# │   └── threshold_config.json
# └── data / datasets / ...
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
# LOAD DEPLOYMENT ARTIFACTS
# ============================================================

@st.cache_resource
def load_resources():

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
# DECISION THRESHOLDS
# ============================================================

# These are the SAME four bands already implemented in
# src/fraud_engine.py.

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
# REQUIRED RAW MODEL INPUTS
# ============================================================

REQUIRED_COLUMNS = (
    ["Time", "Amount"]
    + [f"V{i}" for i in range(1, 29)]
)


# ============================================================
# FIND TRANSACTION DATASET
# ============================================================

@st.cache_data
def find_transaction_dataset():

    search_roots = [
        ROOT_DIR / "data",
        ROOT_DIR / "datasets",
        ROOT_DIR / "artifacts",
        ROOT_DIR,
    ]

    candidate_files = []

    for root in search_roots:

        if not root.exists():
            continue

        try:

            candidate_files.extend(
                root.rglob("*.csv")
            )

            candidate_files.extend(
                root.rglob("*.parquet")
            )

        except Exception:

            continue


    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique_files = []
    seen = set()

    for file_path in candidate_files:

        try:

            resolved = file_path.resolve()

            if resolved not in seen:

                seen.add(resolved)
                unique_files.append(resolved)

        except Exception:

            continue


    # --------------------------------------------------------
    # Prefer likely holdout/test files
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Find first compatible dataset
    # --------------------------------------------------------

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

                return df, str(file_path)


        except Exception:

            continue


    return None, None


# ============================================================
# FIND REAL REPRESENTATIVE TRANSACTIONS
# ============================================================

@st.cache_data
def find_representative_transactions():

    df, source_path = find_transaction_dataset()

    if df is None:

        return {}, None


    # --------------------------------------------------------
    # Keep only model input columns
    # --------------------------------------------------------

    raw_df = df[REQUIRED_COLUMNS].copy()


    raw_df = raw_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )


    raw_df = raw_df.dropna(
        subset=REQUIRED_COLUMNS
    ).reset_index(drop=True)


    if len(raw_df) == 0:

        return {}, source_path


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Score the COMPLETE DATASET in ONE XGBoost CALL.
    #
    # We do NOT call predict_transaction() 228K times.
    # --------------------------------------------------------

    try:

        processed = engine.preprocess(
            raw_df
        )

        probabilities = (
            engine.model.predict_proba(
                processed
            )[:, 1]
        )

    except Exception as e:

        st.error(
            f"Unable to score transaction dataset: {e}"
        )

        return {}, source_path


    # --------------------------------------------------------
    # Identify the four existing decision bands
    # --------------------------------------------------------

    band_masks = {

        "PASS":
            probabilities < 0.20,

        "STEP-UP 2FA":
            (
                (probabilities >= 0.20)
                & (probabilities < 0.60)
            ),

        "MANUAL REVIEW":
            (
                (probabilities >= 0.60)
                & (probabilities < 0.64)
            ),

        "BLOCK":
            probabilities >= 0.64,
    }


    representatives = {}


    # --------------------------------------------------------
    # Pick one REAL transaction from each band
    # --------------------------------------------------------

    for action, mask in band_masks.items():

        positions = np.flatnonzero(mask)

        if len(positions) == 0:
            continue


        # First genuine transaction in the band.
        selected_position = int(
            positions[0]
        )


        selected_row = raw_df.iloc[
            selected_position
        ].copy()


        probability = float(
            probabilities[selected_position]
        )


        tier, expected_action = (
            decision_from_probability(
                probability
            )
        )


        # Sanity check
        if expected_action != action:
            continue


        representatives[action] = {

            "payload":
                selected_row.to_dict(),

            "result": {

                "probability":
                    round(
                        probability,
                        5,
                    ),

                "risk_tier":
                    tier,

                "action":
                    action,
            },

            "index":
                selected_position,
        }


    return representatives, source_path


# ============================================================
# LOAD REPRESENTATIVE TRANSACTIONS
# ============================================================

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
# LEFT COLUMN — INPUTS
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
    # PRESET
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
            preset_action_map[
                demo_scenario
            ]
        )


        preset = representatives.get(
            selected_preset_action
        )


        if preset is None:

            st.error(
                f"No real transaction was found for "
                f"{selected_preset_action}."
            )

            st.info(
                "The available dataset may not contain "
                "a transaction in this probability band."
            )

            payload = None


        else:

            payload = preset["payload"]


            preset_result = preset["result"]


            st.caption(
                "Preset uses a genuine transaction evaluated "
                "through the deployed model."
            )


            st.metric(
                "Actual Model Probability",
                f"{float(preset_result['probability']) * 100:.3f}%",
            )


            st.number_input(
                "Transaction Amount ($)",
                value=float(
                    payload["Amount"]
                ),
                disabled=True,
            )


            st.number_input(
                "Transaction Time (Seconds)",
                value=float(
                    payload["Time"]
                ),
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
# RIGHT COLUMN — DECISION OUTPUT
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
            # REAL DEPLOYED DECISION ENGINE
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
            # POLICY CONSISTENCY CHECK
            # ------------------------------------------------

            expected_tier, expected_action = (
                decision_from_probability(
                    probability
                )
            )


            if (
                engine_tier != expected_tier
                or engine_action != expected_action
            ):

                st.error(
                    "Decision engine and probability bands "
                    "are inconsistent."
                )


                st.write(
                    {
                        "Probability":
                            probability,

                        "Engine Tier":
                            engine_tier,

                        "Engine Action":
                            engine_action,

                        "Expected Tier":
                            expected_tier,

                        "Expected Action":
                            expected_action,
                    }
                )


            elif (
                selected_preset_action is not None
                and engine_action
                != selected_preset_action
            ):

                st.error(
                    f"Preset expected "
                    f"{selected_preset_action}, but the "
                    f"actual model returned "
                    f"{engine_action}."
                )


                st.info(
                    "No artificial probability or action "
                    "override is applied."
                )


            else:

                # ============================================
                # ACTION BADGE
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
                        max(
                            probability,
                            0.0,
                        ),
                        1.0,
                    )
                )


                # ============================================
                # BAND DESCRIPTION
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
                # SHAP
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
# DATA SOURCE INFORMATION
# ============================================================

if data_source is not None:

    st.caption(
        "Scenario presets are based on genuine transactions "
        "from the available project dataset and are scored "
        "using the deployed inference engine."
    )
