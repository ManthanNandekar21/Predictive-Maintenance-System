"""
prediction.py
Loads the trained XGBoost binary failure model and returns predictions.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "xgboost_failure_model.pkl"


def get_risk_level(prob: float) -> tuple[str, str]:
    """
    Map failure probability to a human-readable risk level and colour.
    Thresholds are configurable and clearly distinct from model classification.
    """
    if prob < 0.25:
        return "Low",      "#4CAF50"   # green
    elif prob < 0.50:
        return "Medium",   "#FFC107"   # amber
    elif prob < 0.75:
        return "High",     "#FF9800"   # orange
    else:
        return "Critical", "#F44336"   # red


def load_model():
    """Load the saved XGBoost model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run train_models.py first to train and save the model."
        )
    return joblib.load(MODEL_PATH)


def predict(model, feature_df: pd.DataFrame) -> dict:
    """
    Run the binary failure classifier on a single-row feature DataFrame.

    Returns
    -------
    dict with keys:
        prediction      : 0 or 1
        label           : 'No Failure' | 'Failure'
        prob_failure    : float  (0-1)
        prob_no_failure : float  (0-1)
        risk_level      : str
        risk_color      : str
    """
    prob_arr = model.predict_proba(feature_df)[0]
    prob_no_fail = float(prob_arr[0])
    prob_fail    = float(prob_arr[1])
    pred         = int(prob_fail >= 0.5)
    risk_level, risk_color = get_risk_level(prob_fail)

    return {
        "prediction"      : pred,
        "label"           : "Failure" if pred == 1 else "No Failure",
        "prob_failure"    : prob_fail,
        "prob_no_failure" : prob_no_fail,
        "risk_level"      : risk_level,
        "risk_color"      : risk_color,
    }
