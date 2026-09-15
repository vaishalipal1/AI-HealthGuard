"""
explainable_ai.py
Generates a human-readable explanation of a single prediction using SHAP
(SHapley Additive exPlanations) values from the trained model. Falls back
to the model's built-in feature_importances_ if SHAP is unavailable, so
the app never breaks. Works for any disease via disease_config labels.
"""

import numpy as np
from utils.disease_config import friendly_label


def explain_prediction(model, scaled_input, feature_columns, raw_row, disease_key):
    """
    Returns a list of dicts, sorted by impact, e.g.:
    [{"feature": "Glucose level", "value": 165.0, "impact": "increases risk", "weight": 0.34}, ...]
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(scaled_input)
        values = np.array(raw_values)

        # Normalize whatever shape/version SHAP returns to a flat 1-D
        # array of length n_features (values for the "positive class").
        if values.ndim == 4:
            values = values[0, :, 1]
        elif values.ndim == 3:
            values = values[0, :, 1]
        elif values.ndim == 2:
            if values.shape[-1] == 2 and values.shape[0] == len(feature_columns):
                values = values[:, 1]
            else:
                values = values[0]
    except Exception:
        values = np.array(model.feature_importances_)

    values = np.asarray(values).flatten()

    explanations = []
    for i, col in enumerate(feature_columns):
        weight = float(values[i])
        raw_val = raw_row[col].iloc[0] if hasattr(raw_row[col], "iloc") else raw_row[col]
        explanations.append({
            "feature": friendly_label(disease_key, col),
            "value": float(raw_val),
            "impact": "increases risk" if weight > 0 else "lowers risk",
            "weight": round(abs(weight), 4),
        })

    explanations.sort(key=lambda x: x["weight"], reverse=True)
    return explanations


def top_reasons_text(explanations, top_n=3):
    """Build a short plain-language summary from the top contributing features."""
    top = explanations[:top_n]
    parts = [f"{e['feature']} ({e['value']}) {e['impact']}" for e in top]
    return "; ".join(parts)
