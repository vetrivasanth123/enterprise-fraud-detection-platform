import joblib
import json
import numpy as np
import pandas as pd

class FraudDecisionEngine:
    def __init__(self, model_path='artifacts/xgboost_model.pkl',
                 scaler_path='artifacts/scaler.pkl',
                 config_path='artifacts/threshold_config.json'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.optimal_threshold = self.config['optimal_threshold']
        self.unscaled_cols = ['Time', 'Amount']

    def preprocess(self, df_input):
        df_processed = df_input.copy()
        df_processed[self.unscaled_cols] = self.scaler.transform(df_processed[self.unscaled_cols])
        return df_processed

    def predict_transaction(self, df_single_row):
        df_scaled = self.preprocess(df_single_row)
        prob = float(self.model.predict_proba(df_scaled)[:, 1][0])

        if prob < 0.20:
            risk_band, action = "LOW RISK", "PASS"
        elif prob < 0.60:
            risk_band, action = "MEDIUM RISK", "STEP-UP AUTH (2FA)"
        elif prob < self.optimal_threshold:
            risk_band, action = "HIGH RISK", "MANUAL REVIEW QUEUE"
        else:
            risk_band, action = "CRITICAL RISK", "FRAUD ALERT (BLOCK)"

        return {
            'fraud_probability': round(prob, 4),
            'fraud_probability_pct': f"{prob * 100:.2f}%",
            'risk_band': risk_band,
            'action': action,
            'is_fraud_flag': prob >= self.optimal_threshold
        }
