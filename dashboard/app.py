import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

# app.py:
# enterprise-fraud-detection-platform/
# └── dashboard/
#     └── app.py
#
# Therefore project root is one directory above app.py.

ROOT_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = ROOT_DIR / "src"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Add both project root and src to Python's import path.
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))
# ============================================================
# IMPORT CHECK
# ============================================================

if not hasattr(sys.modules["src.fraud_engine"], "FraudDecisionEngine"):
    st.error(
        "FraudDecisionEngine was not found inside src/fraud_engine.py"
    )
    st.stop()

# ============================================================
# IMPORT PROJECT MODULES
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

    engine = FraudDecisionEngine()

    explainer = FraudExplainabilityEngine()

    config_path = ROOT_DIR / "artifacts" / "threshold_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    return engine, explainer, config


try:

    engine, explainer, config = load_resources()

except Exception as e:

    st.error(f"Error loading model artifacts: {e}")
    st.stop()


# ============================================================
# PROJECT THRESHOLDS
# ============================================================

# These are the same four operational bands used by
# FraudDecisionEngine and documented in the project report.

PASS_THRESHOLD = 0.20
TWO_FA_THRESHOLD = 0.60
REVIEW_THRESHOLD = 0.64


# ============================================================
# PORTFOLIO / SYSTEM OVERVIEW
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
# DECISION BAND FUNCTION
# ============================================================

def decision_from_probability(probability):
    """
    Convert calibrated fraud probability into the same
    four-tier policy used by FraudDecisionEngine.

    P < 0.20
        LOW RISK / PASS

    0.20 <= P < 0.60
        MEDIUM RISK / STEP-UP 2FA

    0.60 <= P < 0.64
        HIGH RISK / MANUAL REVIEW

    P >= 0.64
        CRITICAL RISK / BLOCK
    """

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
# CHECK ENGINE CONSISTENCY
# ============================================================

def validate_engine_result(result):
    """
    Ensure the probability, tier and action are internally
    consistent.

    No dashboard override is performed.
    """

    probability = float(result["probability"])

    expected_tier, expected_action = decision_from_probability(
        probability
    )

    engine_tier = result["risk_tier"]
    engine_action = result["action"]

    return (
        probability,
        engine_tier,
        engine_action,
        expected_tier,
        expected_action,
    )


# ============================================================
# RAW TRANSACTION COLUMN CHECK
# ============================================================

REQUIRED_RAW_COLUMNS = (
    ["Time", "Amount"]
    + [f"V{i}" for i in range(1, 29)]
)


def is_transaction_dataset(df):
    """
    Check whether a dataframe contains the raw transaction
    structure required by FraudDecisionEngine.
    """

    return all(
        column in df.columns
        for column in REQUIRED_RAW_COLUMNS
    )


# ============================================================
# FIND EXISTING HOLDOUT / TRANSACTION DATA
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

    # Remove duplicates
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

    # Prefer files whose names indicate test/holdout data.
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

            if is_transaction_dataset(df):

                return df, str(file_path)

        except Exception:

            continue

    return None, None


# ============================================================
# BUILD REPRESENTATIVE FOUR-TIER PRESETS
# ============================================================

@st.cache_data
def build_representative_presets():

    df, source_path = find_transaction_dataset()

    if df is None:

        return {}, None

    # Only raw model input columns are passed into the
    # existing FraudDecisionEngine.
    #
    # Any target/label/index columns remain untouched in
    # the source dataframe but are NOT sent to the model.

    raw_df = df[REQUIRED_RAW_COLUMNS].copy()

    # Remove rows containing unusable values.
    raw_df = raw_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    raw_df = raw_df.dropna(
        subset=REQUIRED_RAW_COLUMNS
    )

    representatives = {}

    # --------------------------------------------------------
    # Evaluate transactions through the SAME decision engine.
    #
    # No probability is fabricated.
    # No tier is manually assigned.
    # --------------------------------------------------------

    for idx in raw_df.index:

        row = raw_df.loc[[idx]].copy()

        try:

            result = engine.predict_transaction(row)

            probability = float(
                result["probability"]
            )

            tier = result["risk_tier"]
            action = result["action"]

            if action not in representatives:

                representatives[action] = {
                    "payload": row.iloc[0].to_dict(),
                    "result": result,
                    "source_index": idx,
                }

            # Once all four actions have been found,
            # stop scanning.
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


# ============================================================
# LOAD REPRESENTATIVE EXAMPLES
# ============================================================

representative_presets, representative_source = (
    build_representative_presets()
)


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
# LEFT: TRANSACTION INPUTS
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
        key="scenario_select",
    )

    # --------------------------------------------------------
    # CUSTOM INPUT
    # --------------------------------------------------------

    if demo_scenario == "Custom Input":

        default_amount = 15.0
        default_time = 400.0
        default_v14 = 0.0
        default_v10 = 0.0
        default_v4 = 0.0

        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            max_value=100000.0,
            value=default_amount,
            step=5.0,
            key="custom_amount",
        )

        time_val = st.number_input(
            "Transaction Time (Seconds)",
            min_value=0.0,
            max_value=172800.0,
            value=default_time,
            step=100.0,
            key="custom_time",
        )

        st.markdown(
            "**PCA Anomaly Signals (V1 - V28)**"
        )

        v14 = st.slider(
            "V14 (Primary Risk Driver)",
            -20.0,
            10.0,
            default_v14,
            0.5,
            key="custom_v14",
        )

        v10 = st.slider(
            "V10 (Secondary Anomaly Flag)",
            -20.0,
            10.0,
            default_v10,
            0.5,
            key="custom_v10",
        )

        v4 = st.slider(
            "V4 (Transaction Intent Correlation)",
            -10.0,
            10.0,
            default_v4,
            0.5,
            key="custom_v4",
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

        preset_payload = None
        preset_target = None

    # --------------------------------------------------------
    # REPRESENTATIVE MODEL EXAMPLES
    # --------------------------------------------------------

    else:

        action_map = {
            "🟢 Standard Pass (Low Risk)": "PASS",
            "🟡 Step-Up 2FA (Medium Risk)": "STEP-UP 2FA",
            "🟠 Manual Review (Borderline Queue)": "MANUAL REVIEW",
            "🔴 Blocked Fraud (Critical Risk)": "BLOCK",
        }

        target_action = action_map[
            demo_scenario
        ]

        preset_target = target_action

        preset_data = representative_presets.get(
            target_action
        )

        if preset_data is None:

            st.warning(
                f"No genuine {target_action} example was "
                "found in the available transaction data."
            )

            st.info(
                "Custom Input remains available for "
                "real-time scoring."
            )

            payload = None
            preset_payload = None

        else:

            preset_payload = preset_data["payload"]

            payload = preset_payload

            result = preset_data["result"]

            st.caption(
                "Preset uses a genuine transaction evaluated "
                "through the deployed model and decision engine."
            )

            st.markdown(
                f"**Model Probability:** "
                f"{float(result['probability']) * 100:.3f}%"
            )

            # ------------------------------------------------
            # Display actual input values
            # ------------------------------------------------

            amount = float(
                preset_payload["Amount"]
            )

            time_val = float(
                preset_payload["Time"]
            )

            v14 = float(
                preset_payload["V14"]
            )

            v10 = float(
                preset_payload["V10"]
            )

            v4 = float(
                preset_payload["V4"]
            )

            st.number_input(
                "Transaction Amount ($)",
                value=amount,
                disabled=True,
            )

            st.number_input(
                "Transaction Time (Seconds)",
                value=time_val,
                disabled=True,
            )

            st.markdown(
                "**PCA Anomaly Signals (V1 - V28)**"
            )

            st.slider(
                "V14 (Primary Risk Driver)",
                -20.0,
                10.0,
                v14,
                0.5,
                disabled=True,
            )

            st.slider(
                "V10 (Secondary Anomaly Flag)",
                -20.0,
                10.0,
                v10,
                0.5,
                disabled=True,
            )

            st.slider(
                "V4 (Transaction Intent Correlation)",
                -10.0,
                10.0,
                v4,
                0.5,
                disabled=True,
            )


# ============================================================
# RIGHT: DECISION ENGINE OUTPUT
# ============================================================

with col_right:

    st.subheader("Decision Engine Output")

    evaluate_clicked = st.button(
        "Evaluate Transaction Payload",
        type="primary",
    )

    if evaluate_clicked:

        if payload is None:

            st.error(
                "No transaction payload is available."
            )

            st.stop()

        # ----------------------------------------------------
        # REAL DECISION ENGINE
        # ----------------------------------------------------

        if (
            demo_scenario != "Custom Input"
            and preset_payload is not None
        ):

            # Re-evaluate the selected representative
            # transaction rather than trusting the cached
            # displayed result.

            evaluation_df = pd.DataFrame(
                [preset_payload]
            )

            res = engine.predict_transaction(
                evaluation_df
            )

        else:

            res = engine.predict_transaction(
                payload
            )

        # ----------------------------------------------------
        # REAL MODEL OUTPUT
        # ----------------------------------------------------

        (
            prob,
            tier,
            action,
            expected_tier,
            expected_action,
        ) = validate_engine_result(res)

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if (
            tier != expected_tier
            or action != expected_action
        ):

            st.error(
                "Decision engine output is inconsistent "
                "with the configured probability bands."
            )

            st.write(
                {
                    "probability": prob,
                    "engine_tier": tier,
                    "engine_action": action,
                    "expected_tier": expected_tier,
                    "expected_action": expected_action,
                }
            )

            st.stop()

        # ----------------------------------------------------
        # PRESET VALIDATION
        # ----------------------------------------------------

        if (
            preset_target is not None
            and action != preset_target
        ):

            st.error(
                f"Preset mismatch: expected "
                f"{preset_target}, but the actual model "
                f"returned {action}."
            )

            st.stop()

        # ----------------------------------------------------
        # RISK ACTION BADGE
        # ----------------------------------------------------

        if action == "PASS":

            st.success(
                f"**Action: {action}** | Tier: {tier}"
            )

        elif action == "STEP-UP 2FA":

            st.warning(
                f"**Action: {action}** | Tier: {tier}"
            )

        elif action == "MANUAL REVIEW":

            st.warning(
                f"**Action: {action}** | Tier: {tier}"
            )

        elif action == "BLOCK":

            st.error(
                f"**Action: {action}** | Tier: {tier}"
            )

        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        st.metric(
            "Calibrated Fraud Probability (PD)",
            f"{prob * 100:.3f}%",
        )

        st.progress(
            min(
                max(prob, 0.0),
                1.0,
            )
        )

        # ----------------------------------------------------
        # EXPLANATION OF BAND
        # ----------------------------------------------------

        if action == "PASS":

            st.caption(
                "Risk band: P < 0.20 → PASS"
            )

        elif action == "STEP-UP 2FA":

            st.caption(
                "Risk band: 0.20 ≤ P < 0.60 → STEP-UP 2FA"
            )

        elif action == "MANUAL REVIEW":

            st.caption(
                "Risk band: 0.60 ≤ P < 0.64 → MANUAL REVIEW"
            )

        elif action == "BLOCK":

            st.caption(
                "Risk band: P ≥ 0.64 → BLOCK"
            )

        # ----------------------------------------------------
        # SHAP EXPLAINABILITY
        # ----------------------------------------------------

        st.subheader(
            "Transaction SHAP Explainability"
        )

        try:

            df_processed = engine.preprocess(
                payload
            )

            fig = explainer.plot_local_waterfall(
                df_processed
            )

            st.pyplot(
                fig,
                clear_figure=True,
            )

        except Exception as e:

            st.error(
                f"Unable to generate local SHAP "
                f"waterfall: {e}"
            )


# ============================================================
# OPTIONAL DEVELOPMENT INFORMATION
# ============================================================

if representative_source is not None:

    st.caption(
        "Scenario presets are drawn from the available "
        "project transaction data and re-evaluated through "
        "the deployed inference engine."
    )
