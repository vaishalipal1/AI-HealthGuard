# AI-HealthGuard 🩺

Full-stack multi-disease AI health prediction web app — Flask + scikit-learn,
covering **Diabetes, Heart Disease, Liver Disease, and Kidney Disease**, with
login/signup, explainable AI (SHAP), a built-in health chatbot, prediction
history, and downloadable PDF reports.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train all models (creates models/*.pkl for each disease)

```bash
python training/train_model.py
```

## 2. Run the app

```bash
python app.py
```

Open http://127.0.0.1:5000 — sign up, log in, pick a condition under
**New Prediction**, and try the **Assistant** chatbot.

> If you're upgrading from the single-disease version, delete the old
> `database/database.db` first (the table schema changed to support
> multiple diseases) — it will be recreated automatically on first run.

## Project Flow

1. **Signup/Login** — accounts in `database/database.db` (SQLite), hashed passwords.
2. **Choose a condition** — Diabetes 🩸, Heart Disease ❤️, Liver Disease 🫀, or Kidney Disease 🫘.
3. **Prediction** — dynamic form per disease (defined once in `utils/disease_config.py`).
4. **Model** — one RandomForestClassifier per disease, trained on its own dataset in `datasets/`.
5. **Explainable AI** — SHAP shows which factors pushed the prediction up or down.
6. **History** — every prediction saved per-user, across all diseases.
7. **Report** — downloadable PDF (parameters, result, top factors, disclaimer).
8. **Assistant** — a lightweight rule-based chatbot (`utils/chatbot.py`) that explains
   medical terms (glucose, BMI, cholesterol, creatinine, bilirubin, etc.) and can answer
   "what was my last result?" using your saved history. No external API needed.

## Adding another disease

1. Add a dataset CSV to `datasets/` with a `target` column (1 = positive).
2. Add one entry to `DISEASES` in `utils/disease_config.py` (features, labels, icon, color).
3. Re-run `python training/train_model.py` — it trains every disease in the config automatically.

## Notes

- All four datasets here are **synthetically generated** (Pima/UCI-Heart/ILPD/CKD-style
  columns) since no internet access was available while building this. Swap in the real
  Kaggle/UCI CSVs (same column names) and re-run training for realistic accuracy.
- This app is for educational/screening purposes only — not a medical diagnosis tool.
