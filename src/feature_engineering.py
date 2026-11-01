"""
feature_engineering.py
Reproduces the exact feature engineering from the notebook.
"""
import numpy as np
import pandas as pd

# Feature names after sanitization (XGBoost-safe names, no [ ] < >)
SANITIZED_FEATURE_ORDER = [
    'Air temperature K',
    'Process temperature K',
    'Rotational speed rpm',
    'Torque Nm',
    'Tool wear min',
    'Temp_Delta',
    'Power_W',
    'Wear_per_Torque',
    'Speed_Torque_Ratio',
    'Type_H',
    'Type_L',
    'Type_M',
]

# Original feature names (before sanitization)
ORIGINAL_FEATURE_ORDER = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Temp_Delta',
    'Power_W',
    'Wear_per_Torque',
    'Speed_Torque_Ratio',
    'Type_H',
    'Type_L',
    'Type_M',
]


def sanitize_col(name: str) -> str:
    """Remove characters forbidden by XGBoost from column names."""
    return name.replace('[', '').replace(']', '').replace('<', '').replace('>', '')


def engineer_features(air_temp: float, proc_temp: float, rpm: float,
                       torque: float, tool_wear: float,
                       product_type: str) -> pd.DataFrame:
    """
    Given raw user inputs, compute all engineered features and return
    a single-row DataFrame with sanitized column names ready for XGBoost.

    Parameters
    ----------
    air_temp     : Air temperature [K]
    proc_temp    : Process temperature [K]
    rpm          : Rotational speed [rpm]
    torque       : Torque [Nm]
    tool_wear    : Tool wear [min]
    product_type : 'L', 'M', or 'H'

    Returns
    -------
    pd.DataFrame  with exactly the columns in SANITIZED_FEATURE_ORDER
    """
    # Engineered features
    temp_delta          = proc_temp - air_temp
    power_w             = torque * rpm * 2 * np.pi / 60
    wear_per_torque     = tool_wear / (torque + 1e-5)
    speed_torque_ratio  = rpm / (torque + 1e-5)

    # One-hot encode Type
    type_h = 1 if product_type == 'H' else 0
    type_l = 1 if product_type == 'L' else 0
    type_m = 1 if product_type == 'M' else 0

    row = {
        'Air temperature K'    : air_temp,
        'Process temperature K': proc_temp,
        'Rotational speed rpm' : rpm,
        'Torque Nm'            : torque,
        'Tool wear min'        : tool_wear,
        'Temp_Delta'           : temp_delta,
        'Power_W'              : power_w,
        'Wear_per_Torque'      : wear_per_torque,
        'Speed_Torque_Ratio'   : speed_torque_ratio,
        'Type_H'               : type_h,
        'Type_L'               : type_l,
        'Type_M'               : type_m,
    }

    df = pd.DataFrame([row], columns=SANITIZED_FEATURE_ORDER)
    return df.astype(float)


def engineer_from_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Batch feature engineering from a raw DataFrame (with original column names).
    Used during model training.
    """
    df = df_raw.copy()
    df['Temp_Delta']         = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Power_W']            = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 2 * np.pi / 60
    df['Wear_per_Torque']    = df['Tool wear [min]'] / (df['Torque [Nm]'] + 1e-5)
    df['Speed_Torque_Ratio'] = df['Rotational speed [rpm]'] / (df['Torque [Nm]'] + 1e-5)
    df = pd.get_dummies(df, columns=['Type'], prefix='Type')
    for c in ['Type_H', 'Type_L', 'Type_M']:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].astype(int)
    return df
