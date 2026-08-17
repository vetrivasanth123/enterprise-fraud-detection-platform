import joblib
import matplotlib.pyplot as plt
import shap


class FraudExplainabilityEngine:

  def __init__(self, model_path='artifacts/xgboost_model.pkl'):
    self.model = joblib.load(model_path)
    self.explainer = shap.TreeExplainer(self.model)

  def get_local_explanation(self, df_processed):
    return self.explainer(df_processed)

  def plot_local_waterfall(self, df_processed):
    shap_values = self.explainer(df_processed)
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    return fig
