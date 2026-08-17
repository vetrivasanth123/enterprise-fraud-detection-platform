import json
import joblib
import numpy as np
import pandas as pd


class FraudDecisionEngine:

  def __init__(
      self,
      model_path='artifacts/xgboost_model.pkl',
      scaler_path='artifacts/scaler.pkl',
      config_path='artifacts/threshold_config.json',
  ):
    self.model = joblib.load(model_path)
    self.scaler = joblib.load(scaler_path)

    with open(config_path, 'r') as f:
      self.config = json.load(f)

    self.unscaled_cols = ['Time', 'Amount']
    self.expected_order = [
        'Time',
        'V1',
        'V2',
        'V3',
        'V4',
        'V5',
        'V6',
        'V7',
        'V8',
        'V9',
        'V10',
        'V11',
        'V12',
        'V13',
        'V14',
        'V15',
        'V16',
        'V17',
        'V18',
        'V19',
        'V20',
        'V21',
        'V22',
        'V23',
        'V24',
        'V25',
        'V26',
        'V27',
        'V28',
        'Amount',
        'Hour_Of_Day',
        'Is_Night_Tx',
        'Log_Amount',
        'Is_Micro_Tx',
    ]

  def engineer_features(self, df):
    df_feat = df.copy()
    df_feat['Hour_Of_Day'] = (df_feat['Time'] / 3600.0) % 24
    df_feat['Is_Night_Tx'] = (
        (df_feat['Hour_Of_Day'] >= 0) & (df_feat['Hour_Of_Day'] < 6)
    ).astype(int)
    df_feat['Log_Amount'] = np.log1p(df_feat['Amount'])
    df_feat['Is_Micro_Tx'] = (df_feat['Amount'] <= 1.0).astype(int)
    return df_feat

  def preprocess(self, df_input):
    if isinstance(df_input, dict):
      df_input = pd.DataFrame([df_input])

    # 1. Feature Engineering (30 raw -> 34 processed features)
    df_processed = self.engineer_features(df_input)

    # 2. Scale Time and Amount
    df_processed[self.unscaled_cols] = self.scaler.transform(
        df_processed[self.unscaled_cols]
    )

    # 3. Enforce exact feature ordering expected by XGBoost
    return df_processed[self.expected_order]

  def predict_transaction(self, df_single_row):
    df_scaled = self.preprocess(df_single_row)
    prob = float(self.model.predict_proba(df_scaled)[:, 1][0])

    if prob < 0.20:
      tier, action = 'LOW RISK', 'PASS'
    elif prob < 0.60:
      tier, action = 'MEDIUM RISK', 'STEP-UP 2FA'
    elif prob < 0.64:
      tier, action = 'HIGH RISK', 'MANUAL REVIEW'
    else:
      tier, action = 'CRITICAL RISK', 'BLOCK'

    return {
        'probability': round(prob, 5),
        'risk_tier': tier,
        'action': action,
    }
