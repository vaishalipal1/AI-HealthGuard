"""
train_model.py
Trains a RandomForestClassifier for EVERY disease listed in
utils/disease_config.py and saves each as its own pickle bundle
(model + scaler + feature columns + metrics) under models/.

Run from the project root:
    python training/train_model.py
"""

import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.disease_config import DISEASES, feature_names
from utils.preprocessing import load_dataset, clean_dataframe, split_features_target, fit_scaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def train_one(disease_key):
    print(f"\n=== Training: {DISEASES[disease_key]['label']} ===")
    df = load_dataset(disease_key)
    df = clean_dataframe(df)
    X, y = split_features_target(df, disease_key)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
    }
    print("Metrics:", metrics)

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_names(disease_key),
        "metrics": metrics,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, DISEASES[disease_key]["model_file"])
    joblib.dump(bundle, model_path)
    print(f"Saved -> {model_path}")


def main():
    for disease_key in DISEASES:
        train_one(disease_key)
    print("\nAll models trained successfully.")


if __name__ == "__main__":
    main()
