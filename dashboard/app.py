import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="Enterprise Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide"
)

# Load Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('artifacts/xgboost_model.pkl')
    scaler = joblib.load('artifacts/scaler.pkl')
    with open('artifacts/threshold_config.json', 'r') as f:
        config = json.load(f)
    return model, scaler, config

try:
    model, scaler, config = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False

# Title Banner
st.title("🛡️ Enterprise Fraud Detection & Risk Triage Platform")
st.markdown("Real-Time Transaction Scoring, Multi-Tier Risk Allocation & SHAP Explainability")
st.divider()

if not artifacts_loaded:
    st.error("Error loading model artifacts from 'artifacts/' directory. Please ensure Phase 8/9 artifacts exist.")
    st.stop()

# ---------------------------------------------------------
# SECTION 1: Portfolio Overview KPIs
# ---------------------------------------------------------
st.header("1. Portfolio & System Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Evaluation Set", "56,898", "Holdout Set")
with col2:
    st.metric("Historical Fraud Rate", "0.156%", "-0.02% vs Train")
with col3:
    st.metric("Operational Threshold", f"{config['optimal_threshold']:.2f}", "Max F1-Score")
with col4:
    st.metric("Holdout Precision", f"{config['holdout_precision']*100:.1f}%", "+7.7% vs Default")

st.divider()

# ---------------------------------------------------------
# SECTION 2: Model Performance Benchmark
# ---------------------------------------------------------
st.header("2. Champion Model Performance Metrics")
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

with m_col1:
    st.metric("PR-AUC (Primary)", "0.8354")
with m_col2:
    st.metric("ROC-AUC", "0.9844")
with m_col3:
    st.metric("Precision", f"{config['holdout_precision']:.4f}")
with m_col4:
    st.metric("Recall", f"{config['holdout_recall']:.4f}")
with m_col5:
    st.metric("F1-Score", f"{config['holdout_f1']:.4f}")

st.divider()

# ---------------------------------------------------------
# SECTION 3: Live Transaction Risk Profiler
# ---------------------------------------------------------
st.header("3. Interactive Transaction Risk Profiler")
st.write("Adjust transaction attributes below to simulate live scoring through the Decision Engine:")

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=25000.0, value=126.75, step=10.0)
with input_col2:
    time_val = st.number_input("Time Elapsed (Seconds)", min_value=0.0, max_value=180000.0, value=50000.0, step=1000.0)
with input_col3:
    risk_preset = st.selectbox("Simulated PCA Vector Preset", ["Normal Legitimate", "Critical Fraud Pattern"])

# Generate Feature Vector based on Preset
v_cols = [f'V{i}' for i in range(1, 29)]
v_values = {}

if risk_preset == "Normal Legitimate":
    for col in v_cols:
        v_values[col] = 0.0
else:
    # High risk preset derived from Phase 10 fraud sample
    for col in v_cols:
        v_values[col] = 0.0
    v_values['V14'] = -5.33
    v_values['V10'] = -7.94
    v_values['V4'] = 6.45
    v_values['V12'] = -5.84
    v_values['V17'] = -11.43

# Construct Input DataFrame
input_dict = {'Time': [time_val], 'Amount': [amount]}
input_dict.update({k: [v] for k, v in v_values.items()})

input_df = pd.DataFrame(input_dict)
# Reorder to match model training feature order
feature_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
input_df = input_df[feature_order]

# Preprocess & Score
input_scaled = input_df.copy()
input_scaled[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])

prob = float(model.predict_proba(input_scaled)[:, 1][0])
opt_thresh = config['optimal_threshold']

# Assign Action and Color
if prob < 0.20:
    band, action, color = "LOW RISK", "PASS", "green"
elif prob < 0.60:
    band, action, color = "MEDIUM RISK", "STEP-UP AUTH (2FA)", "orange"
elif prob < opt_thresh:
    band, action, color = "HIGH RISK", "MANUAL REVIEW QUEUE", "orange"
else:
    band, action, color = "CRITICAL RISK", "FRAUD ALERT (BLOCK)", "red"

res_col1, res_col2, res_col3 = st.columns(3)

with res_col1:
    st.metric("Fraud Probability", f"{prob*100:.2f}%")
with res_col2:
    st.metric("Assigned Risk Band", band)
with res_col3:
    st.metric("Operational Action", action)

if color == "red":
    st.error(f"🚨 ALERT: Transaction scored above operational threshold ({opt_thresh:.2f}). Action: {action}")
elif color == "orange":
    st.warning(f"⚠️ WARNING: Transaction requires further verification. Action: {action}")
else:
    st.success(f"✅ CLEAR: Transaction approved for settlement. Action: {action}")

st.divider()

# ---------------------------------------------------------
# SECTION 4: Local SHAP Explainability
# ---------------------------------------------------------
st.header("4. Root-Cause Explainability (SHAP Breakdown)")
st.write("Decomposition of latent features contributing to transaction risk score:")

explainer = shap.TreeExplainer(model)
shap_exp = explainer(input_scaled)
shap_vals = shap_exp.values[0]

shap_df = pd.DataFrame({
    'Feature': input_scaled.columns,
    'Feature Value': input_df.iloc[0].values,
    'SHAP Impact (Log-Odds)': shap_vals
}).sort_values(by='SHAP Impact (Log-Odds)', key=abs, ascending=False)

st.dataframe(shap_df.head(10).style.highlight_max(subset=['SHAP Impact (Log-Odds)'], color='lightcoral'), use_container_width=True)

st.caption("Enterprise Fraud Detection Platform — Completed & Verified Holdout Pipeline")
