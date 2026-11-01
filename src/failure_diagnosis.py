"""
failure_diagnosis.py
Loads a separately trained multi-label classifier for failure sub-type diagnosis.
TWF / HDF / PWF / OSF / RNF are predicted as independent binary outputs.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "failure_type_model.pkl"

FAILURE_TYPES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

FAILURE_DESCRIPTIONS = {
    "TWF": "Occurs when tool wear exceeds the product-type tolerance limit.",
    "HDF": "Occurs when the temperature difference between process and air is too low at high rotational speed.",
    "PWF": "Occurs when mechanical power (torque × speed) is outside the safe operating envelope.",
    "OSF": "Occurs when tool wear × torque product exceeds the product-type threshold.",
    "RNF": "Random failure with 0.1% probability regardless of sensor conditions.",
}

CONFIDENCE_THRESHOLD = 0.30   # minimum probability to assert a failure type


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Subtype model not found at {MODEL_PATH}. Run train_models.py first."
        )
    return joblib.load(MODEL_PATH)


def diagnose(model, feature_df: pd.DataFrame,
             binary_prediction: int) -> dict:
    """
    Predict failure sub-types for a single sample.

    Returns
    -------
    dict with keys:
        available       : bool   — False if subtype model unavailable
        subtypes        : list   — active failure type codes above threshold
        probabilities   : dict   — {code: probability}
        descriptions    : list   — human-readable subtype descriptions
        unclassified    : bool   — True if Failure predicted but no subtype found
    """
    result = {
        "available"   : True,
        "subtypes"    : [],
        "probabilities": {},
        "descriptions" : [],
        "unclassified" : False,
    }

    try:
        if model is None:
            result["available"] = False
            return result

        # model is a dict of {flag: classifier}
        probs = {}
        for flag, clf in model.items():
            p = float(clf.predict_proba(feature_df)[0][1])
            probs[flag] = round(p, 4)

        result["probabilities"] = probs

        if binary_prediction == 0:
            # No failure predicted — subtype diagnosis not relevant
            return result

        active = [f for f, p in probs.items() if p >= CONFIDENCE_THRESHOLD]
        result["subtypes"] = active
        result["descriptions"] = [
            f"**{FAILURE_TYPES[f]}** ({f}): {FAILURE_DESCRIPTIONS[f]}"
            for f in active
        ]
        if not active:
            result["unclassified"] = True

    except Exception as e:
        result["available"] = False
        result["error"] = str(e)

    return result
