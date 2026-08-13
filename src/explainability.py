import shap
import pandas as pd

def get_shap_explainer(model):
    return shap.TreeExplainer(model)

def explain_single_transaction(explainer, df_single_scaled):
    shap_exp = explainer(df_single_scaled)
    vals = shap_exp.values[0]
    cols = df_single_scaled.columns

    df_shap = pd.DataFrame({
        'feature': cols,
        'value': df_single_scaled.iloc[0].values,
        'shap_value': vals
    }).sort_values(by='shap_value', key=abs, ascending=False)

    return df_shap
