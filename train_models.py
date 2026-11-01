"""
train_models.py
===============
Run this script ONCE before starting the Streamlit dashboard.
It loads ai4i2020_balanced.csv, trains:
  1. XGBoost binary failure classifier  → models/xgboost_failure_model.pkl
  2. Per-flag RandomForest classifiers  → models/failure_type_model.pkl

Usage
-----
    python train_models.py --csv ai4i2020_balanced.csv
"""
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")
    sys.exit(1)

from src.feature_engineering import engineer_from_dataframe, SANITIZED_FEATURE_ORDER

RANDOM_STATE = 42
MODEL_DIR    = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

SUB_FLAGS    = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
TARGET_COL   = 'Machine failure'
DROP_COLS    = ['UDI', 'Product ID']


def sanitize_col(name: str) -> str:
    return name.replace('[','').replace(']','').replace('<','').replace('>','')


def load_and_prepare(csv_path: str):
    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")
    print(f"  Failure rate: {df[TARGET_COL].mean()*100:.2f}%")

    # Drop identifiers
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Feature engineering
    df = engineer_from_dataframe(df)

    # Build feature matrix (sanitized column names for XGBoost)
    base_features = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
        'Temp_Delta', 'Power_W', 'Wear_per_Torque', 'Speed_Torque_Ratio',
        'Type_H', 'Type_L', 'Type_M',
    ]
    # Keep only columns that exist
    base_features = [f for f in base_features if f in df.columns]

    X_raw = df[base_features].astype(float)
    # Sanitize for XGBoost
    X = X_raw.rename(columns=sanitize_col)

    y     = df[TARGET_COL].astype(int)
    y_sub = df[SUB_FLAGS].astype(int)   # for sub-type model

    return X, y, y_sub


def train_binary_model(X, y):
    print("\n── Training XGBoost binary failure classifier ──")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=pos_w,
        eval_metric='logloss',
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    f1     = f1_score(y_te, y_pred)
    auc    = roc_auc_score(y_te, y_prob)
    print(f"  Test F1  : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_te, y_pred,
          target_names=['No Failure', 'Failure'], digits=4))
    return model


def train_subtype_models(X, y_binary, y_sub):
    """
    Train one RandomForest classifier per failure flag.
    Models are trained only on failure records to capture failure-mode patterns.
    """
    print("\n── Training failure sub-type classifiers ──")
    # Use all records (failure + no-failure) so each classifier learns
    # to distinguish its flag from the rest
    subtype_models = {}

    for flag in SUB_FLAGS:
        y_flag   = y_sub[flag].values
        n_pos    = y_flag.sum()
        n_total  = len(y_flag)

        if n_pos < 10:
            print(f"  {flag}: only {n_pos} positive samples — skipping (too few).")
            continue

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight='balanced',
            n_jobs=-1, random_state=RANDOM_STATE,
        )
        clf.fit(X, y_flag)

        y_pred = clf.predict(X)
        f1     = f1_score(y_flag, y_pred, zero_division=0)
        print(f"  {flag}: n_pos={n_pos:4d} / {n_total} | Train F1={f1:.4f}  "
              f"[{flag} = {n_pos/n_total*100:.2f}%]")
        subtype_models[flag] = clf

    if not subtype_models:
        print("  No sub-type models trained (insufficient data).")
    return subtype_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='ai4i2020_balanced.csv',
                        help='Path to balanced CSV dataset')
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"ERROR: CSV file not found: {args.csv}")
        print("Please ensure ai4i2020_balanced.csv is in the project root.")
        sys.exit(1)

    X, y, y_sub = load_and_prepare(args.csv)

    # 1. Binary model
    xgb_model = train_binary_model(X, y)
    xgb_path  = MODEL_DIR / "xgboost_failure_model.pkl"
    joblib.dump(xgb_model, xgb_path)
    print(f"\n  ✓ XGBoost model saved → {xgb_path}")

    # 2. Sub-type models
    sub_models = train_subtype_models(X, y, y_sub)
    sub_path   = MODEL_DIR / "failure_type_model.pkl"
    joblib.dump(sub_models, sub_path)
    print(f"  ✓ Sub-type models saved → {sub_path}")

    print("\n" + "="*50)
    print("Training complete. You can now run the dashboard:")
    print("  streamlit run app.py")
    print("="*50)


if __name__ == "__main__":
    main()
