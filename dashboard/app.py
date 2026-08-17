import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

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

ROOT_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = ROOT_DIR / "src"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# IMPORT PROJECT ENGINES
# ============================================================

from src.fraud_engine import FraudDecisionEngine
from src.explainability import FraudExplainabilityEngine


# ============================================================
# STREAMLIT CONFIG
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
# LOCKED PROJECT DECISION BANDS
# ============================================================

# These match src/fraud_engine.py exactly.
#
# DO NOT change these for the dashboard.
#
# CV/report operating threshold:
# 0.64
#
# Four dashboard policy bands:
# < 0.20       PASS
# 0.20-<0.60   STEP-UP 2FA
# 0.60-<0.64   MANUAL REVIEW
# >= 0.64      BLOCK

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
# FIND A COMPATIBLE TRANSACTION DATASET
# ============================================================

@st.cache_data
def find_transaction_dataset():

    search_roots = [
        ROOT_DIR / "data",
        ROOT_DIR / "datasets",
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

    for path in candidate_files:

        try:

            resolved = path.resolve()

            if resolved not in seen:

                seen.add(resolved)
                unique_files.append(resolved)

        except Exception:

            continue


    # Prefer holdout/test/validation-looking files
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

                return df, str(file_path)

        except Exception:

            continue


    return None, None


# ============================================================
# FIND REAL TRANSACTIONS FOR THE NATURAL TIERS
# ============================================================

@st.cache_data
def find_real_representatives():

    df, source_path = find_transaction_dataset()

    if df is None:

        return {}, None


    raw_df = df[
        REQUIRED_COLUMNS
    ].copy()


    raw_df = raw_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )


    raw_df = raw_df.dropna(
        subset=REQUIRED_COLUMNS
    ).reset_index(drop=True)


    if len(raw_df) == 0:

        return {}, source_path


    try:

        processed = engine.preprocess(
            raw_df
        )

        probabilities = (
            engine.model.predict_proba(
                processed
            )[:, 1]
        )

    except Exception:

        return {}, source_path


    masks = {

        "PASS":
            probabilities < 0.20,

        "STEP-UP 2FA":
            (
                (probabilities >= 0.20)
                & (probabilities < 0.60)
            ),

        "BLOCK":
            probabilities >= 0.64,
    }


    representatives = {}


    for action, mask in masks.items():

        positions = np.flatnonzero(mask)

        if len(positions) == 0:
            continue


        position = int(
            positions[0]
        )


        row = raw_df.iloc[
            position
        ].copy()


        probability = float(
            probabilities[position]
        )


        tier, calculated_action = (
            decision_from_probability(
                probability
            )
        )


        if calculated_action != action:
            continue


        representatives[action] = {

            "payload":
                row.to_dict(),

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
        }


    return representatives, source_path


# ============================================================
# CREATE MANUAL-REVIEW DEMONSTRATION
# ============================================================

@st.cache_data
def find_manual_review_demo():

    df, source_path = find_transaction_dataset()

    if df is None:

        return None


    raw_df = df[
        REQUIRED_COLUMNS
    ].copy()


    raw_df = raw_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )


    raw_df = raw_df.dropna(
        subset=REQUIRED_COLUMNS
    ).reset_index(drop=True)


    if len(raw_df) == 0:

        return None


    # --------------------------------------------------------
    # We use real transactions as starting points.
    #
    # The dashboard then creates candidate feature vectors
    # between genuine low-risk and high-risk observations.
    #
    # These are DEMONSTRATION inputs only.
    #
    # The final probability is ALWAYS calculated by the
    # deployed XGBoost model.
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

    except Exception:

        return None


    low_positions = np.flatnonzero(
        probabilities < 0.20
    )

    high_positions = np.flatnonzero(
        probabilities >= 0.64
    )


    if (
        len(low_positions) == 0
        or len(high_positions) == 0
    ):

        return None


    # --------------------------------------------------------
    # Select a manageable number of low/high examples.
    # --------------------------------------------------------

    low_sample = low_positions[
        : min(25, len(low_positions))
    ]

    high_sample = high_positions[
        : min(25, len(high_positions))
    ]


    candidates = []


    # --------------------------------------------------------
    # Interpolate between genuine low-risk and high-risk
    # transactions.
    #
    # The model itself decides whether a candidate falls into
    # the 0.60-0.64 review band.
    # --------------------------------------------------------

    interpolation_values = np.linspace(
        0.0,
        1.0,
        101,
    )


    for low_idx in low_sample:

        low_row = raw_df.iloc[
            low_idx
        ].to_numpy(dtype=float)


        for high_idx in high_sample:

            high_row = raw_df.iloc[
                high_idx
            ].to_numpy(dtype=float)


            for alpha in interpolation_values:

                candidate = (
                    (1.0 - alpha) * low_row
                    + alpha * high_row
                )

                candidates.append(candidate)


    if len(candidates) == 0:

        return None


    candidates_array = np.asarray(
        candidates,
        dtype=float,
    )


    candidate_df = pd.DataFrame(
        candidates_array,
        columns=REQUIRED_COLUMNS,
    )


    candidate_df = candidate_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )


    candidate_df = candidate_df.dropna(
        subset=REQUIRED_COLUMNS
    ).reset_index(drop=True)


    if len(candidate_df) == 0:

        return None


    # --------------------------------------------------------
    # ONE BATCH MODEL EVALUATION
    # --------------------------------------------------------

    try:

        candidate_processed = (
            engine.preprocess(
                candidate_df
            )
        )

        candidate_probabilities = (
            engine.model.predict_proba(
                candidate_processed
            )[:, 1]
        )

    except Exception:

        return None


    # --------------------------------------------------------
    # Find actual model output inside:
    #
    # 0.60 <= P < 0.64
    # --------------------------------------------------------

    review_positions = np.flatnonzero(
        (
            candidate_probabilities >= 0.60
        )
        & (
            candidate_probabilities < 0.64
        )
    )


    if len(review_positions) == 0:

        return None


    # Choose the probability closest to the middle of
    # the documented review band.
    target = 0.62

    best_position = review_positions[
        np.argmin(
            np.abs(
                candidate_probabilities[
                    review_positions
                ]
                - target
            )
        )
    ]


    best_position = int(
        best_position
    )


    payload = candidate_df.iloc[
        best_position
    ].to_dict()


    probability = float(
        candidate_probabilities[
            best_position
        ]
    )


    tier, action = (
        decision_from_probability(
            probability
        )
    )


    if action != "MANUAL REVIEW":

        return None


    return {

        "payload":
            payload,

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

        "synthetic":
            True,
    }


# ============================================================
# LOAD REPRESENTATIVES
# ============================================================

real_representatives, data_source = (
    find_real_representatives()
)


manual_review_demo = (
    find_manual_review_demo()
)


# ============================================================
# 1. PORTFOLIO & SYSTEM OVERVIEW
# ============================================================

st.header(
    "1. Portfolio & System Overview"
)


col1, col2, col3, col4 = (
    st.columns(4)
)


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


col_left, col_right = (
    st.columns([1, 1])
)


# ============================================================
# LEFT SIDE
# ============================================================

with col_left:

    st.subheader(
        "Transaction Inputs"
    )


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
            "Time":
                float(time_val),

            "Amount":
                float(amount),
        }


        for i in range(1, 29):

            payload[
                f"V{i}"
            ] = 0.0


        payload["V14"] = float(v14)
        payload["V10"] = float(v10)
        payload["V4"] = float(v4)


        selected_preset_action = None
        selected_preset_result = None


    # ========================================================
    # PRESET INPUT
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


        # ----------------------------------------------------
        # Manual Review is the special dashboard demo.
        # ----------------------------------------------------

        if (
            selected_preset_action
            == "MANUAL REVIEW"
        ):

            preset = (
                manual_review_demo
            )

            if preset is None:

                st.error(
                    "The deployed model did not produce "
                    "a probability inside the 0.60-0.64 "
                    "Manual Review band from the dashboard "
                    "demonstration search."
                )

                st.info(
                    "No model, threshold, calibration, or "
                    "CV/report result has been changed."
                )

                payload = None
                selected_preset_result = None

            else:

                payload = preset[
                    "payload"
                ]

                selected_preset_result = (
                    preset["result"]
                )

                st.caption(
                    "Dashboard demonstration input scored "
                    "by the deployed fraud model."
                )


        # ----------------------------------------------------
        # Other three presets use genuine transactions.
        # ----------------------------------------------------

        else:

            preset = (
                real_representatives.get(
                    selected_preset_action
                )
            )


            if preset is None:

                st.error(
                    f"No genuine transaction was found "
                    f"for {selected_preset_action} "
                    f"in the available dataset."
                )

                payload = None
                selected_preset_result = None

            else:

                payload = preset[
                    "payload"
                ]

                selected_preset_result = (
                    preset["result"]
                )

                st.caption(
                    "Preset uses a genuine transaction "
                    "evaluated through the deployed model."
                )


        # ----------------------------------------------------
        # DISPLAY SELECTED PRESET INPUT
        # ----------------------------------------------------

        if payload is not None:

            st.metric(
                "Actual Model Probability",
                f"{float(selected_preset_result['probability']) * 100:.3f}%",
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
# RIGHT SIDE
# ============================================================

with col_right:

    st.subheader(
        "Decision Engine Output"
    )


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
            # FINAL EVALUATION THROUGH ACTUAL ENGINE
            # ------------------------------------------------

            result = (
                engine.predict_transaction(
                    pd.DataFrame(
                        [payload]
                    )
                )
            )


            probability = float(
                result["probability"]
            )


            tier = result[
                "risk_tier"
            ]


            action = result[
                "action"
            ]


            # ------------------------------------------------
            # INDEPENDENT POLICY CHECK
            # ------------------------------------------------

            expected_tier, expected_action = (
                decision_from_probability(
                    probability
                )
            )


            if (
                tier != expected_tier
                or action != expected_action
            ):

                st.error(
                    "Decision engine and dashboard "
                    "policy bands are inconsistent."
                )


                st.write(
                    {
                        "Model Probability":
                            probability,

                        "Engine Tier":
                            tier,

                        "Engine Action":
                            action,

                        "Expected Tier":
                            expected_tier,

                        "Expected Action":
                            expected_action,
                    }
                )


            elif (
                selected_preset_action
                is not None
                and action
                != selected_preset_action
            ):

                st.error(
                    f"The actual deployed model returned "
                    f"{action}, while this preset is intended "
                    f"for {selected_preset_action}."
                )


                st.info(
                    "The dashboard does not override the "
                    "model probability or decision."
                )


            else:

                # ============================================
                # ACTION BADGE
                # ============================================

                if action == "PASS":

                    st.success(
                        f"**Action: PASS** | "
                        f"Tier: {tier}"
                    )


                elif action == "STEP-UP 2FA":

                    st.warning(
                        f"**Action: STEP-UP 2FA** | "
                        f"Tier: {tier}"
                    )


                elif action == "MANUAL REVIEW":

                    st.warning(
                        f"**Action: MANUAL REVIEW** | "
                        f"Tier: {tier}"
                    )


                elif action == "BLOCK":

                    st.error(
                        f"**Action: BLOCK** | "
                        f"Tier: {tier}"
                    )


                # ============================================
                # PROBABILITY
                # ============================================

                st.metric(
                    "Fraud Probability",
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
                # RISK BAND
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
# FOOTER
# ============================================================

if data_source is not None:

    st.caption(
        "Model outputs and risk actions are generated by "
        "the deployed inference engine using the project's "
        "locked decision policy."
    )
