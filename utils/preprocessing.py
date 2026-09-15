"""
preprocessing.py
Generic data-cleaning and scaling logic, driven by disease_config.py,
shared by training/train_model.py and app.py so the exact same
transformation is applied at train time and at prediction time.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from utils.disease_config import DISEASES, feature_names

# Columns where a value of 0 is biologically implausible / means "missing".
# Only applied when the column exists for that disease.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                    "trestbps", "chol", "thalach"]


def load_dataset(disease_key: str) -> pd.DataFrame:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "datasets", DISEASES[disease_key]["dataset"])
    return pd.read_csv(path)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Replace biologically impossible 0 values with NaN, then impute with median."""
    df = df.copy()
    for col in ZERO_AS_MISSING:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
    return df


def split_features_target(df: pd.DataFrame, disease_key: str):
    cols = feature_names(disease_key)
    X = df[cols]
    y = df["target"] if "target" in df.columns else df["Outcome"]
    return X, y


def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def transform_input(raw_input: dict, scaler: StandardScaler, disease_key: str):
    """
    Take a dict of form-submitted values, build a single-row DataFrame in the
    correct column order for this disease, clean it, then scale it.
    Returns (scaled_array, cleaned_dataframe).
    """
    cols = feature_names(disease_key)
    row = {col: float(raw_input.get(col, 0)) for col in cols}
    df = pd.DataFrame([row], columns=cols)
    df = clean_dataframe(df)
    scaled = scaler.transform(df)
    return scaled, df
