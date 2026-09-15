"""
disease_config.py
Single source of truth for every disease the app can predict:
its display name, icon, feature columns (with friendly labels,
input type and normal-range hints), and the dataset/model file names.
Add a new disease by adding one entry here + a dataset in datasets/.
"""

DISEASES = {
    "diabetes": {
        "label": "Diabetes",
        "icon": "🩸",
        "color": "#5B8DEF",
        "dataset": "health_dataset.csv",
        "model_file": "diabetes_model.pkl",
        "features": [
            {"name": "Pregnancies", "label": "Pregnancies", "type": "number", "step": "1", "min": "0", "hint": "Number of times pregnant"},
            {"name": "Glucose", "label": "Glucose (mg/dL)", "type": "number", "step": "0.1", "hint": "Normal fasting: 70-100"},
            {"name": "BloodPressure", "label": "Blood Pressure (mm Hg)", "type": "number", "step": "0.1", "hint": "Normal: ~80"},
            {"name": "SkinThickness", "label": "Skin Thickness (mm)", "type": "number", "step": "0.1"},
            {"name": "Insulin", "label": "Insulin (mu U/ml)", "type": "number", "step": "0.1"},
            {"name": "BMI", "label": "BMI", "type": "number", "step": "0.1", "hint": "Normal: 18.5-24.9"},
            {"name": "DiabetesPedigreeFunction", "label": "Diabetes Pedigree Function", "type": "number", "step": "0.001", "hint": "Family history score"},
            {"name": "Age", "label": "Age", "type": "number", "step": "1", "min": "1"},
        ],
    },
    "heart": {
        "label": "Heart Disease",
        "icon": "❤️",
        "color": "#F06292",
        "dataset": "heart_disease.csv",
        "model_file": "heart_model.pkl",
        "features": [
            {"name": "age", "label": "Age", "type": "number", "step": "1"},
            {"name": "sex", "label": "Sex", "type": "select", "options": [("1", "Male"), ("0", "Female")]},
            {"name": "cp", "label": "Chest Pain Type", "type": "select", "options": [("0", "Typical Angina"), ("1", "Atypical Angina"), ("2", "Non-anginal Pain"), ("3", "Asymptomatic")]},
            {"name": "trestbps", "label": "Resting Blood Pressure", "type": "number", "step": "0.1", "hint": "Normal: ~120"},
            {"name": "chol", "label": "Cholesterol (mg/dL)", "type": "number", "step": "0.1", "hint": "Normal: <200"},
            {"name": "fbs", "label": "Fasting Blood Sugar > 120 mg/dL", "type": "select", "options": [("1", "Yes"), ("0", "No")]},
            {"name": "restecg", "label": "Resting ECG", "type": "select", "options": [("0", "Normal"), ("1", "ST-T Abnormality"), ("2", "LV Hypertrophy")]},
            {"name": "thalach", "label": "Max Heart Rate Achieved", "type": "number", "step": "0.1"},
            {"name": "exang", "label": "Exercise-Induced Angina", "type": "select", "options": [("1", "Yes"), ("0", "No")]},
            {"name": "oldpeak", "label": "ST Depression (Oldpeak)", "type": "number", "step": "0.1"},
            {"name": "slope", "label": "Slope of ST Segment", "type": "select", "options": [("0", "Upsloping"), ("1", "Flat"), ("2", "Downsloping")]},
            {"name": "ca", "label": "Major Vessels Colored (0-4)", "type": "select", "options": [("0", "0"), ("1", "1"), ("2", "2"), ("3", "3"), ("4", "4")]},
            {"name": "thal", "label": "Thalassemia", "type": "select", "options": [("1", "Normal"), ("2", "Fixed Defect"), ("3", "Reversible Defect")]},
        ],
    },
    "liver": {
        "label": "Liver Disease",
        "icon": "🫀",
        "color": "#66BB6A",
        "dataset": "liver_disease.csv",
        "model_file": "liver_model.pkl",
        "features": [
            {"name": "Age", "label": "Age", "type": "number", "step": "1"},
            {"name": "Gender", "label": "Gender", "type": "select", "options": [("1", "Male"), ("0", "Female")]},
            {"name": "Total_Bilirubin", "label": "Total Bilirubin", "type": "number", "step": "0.01", "hint": "Normal: 0.3-1.2"},
            {"name": "Direct_Bilirubin", "label": "Direct Bilirubin", "type": "number", "step": "0.01", "hint": "Normal: 0.1-0.3"},
            {"name": "Alkaline_Phosphotase", "label": "Alkaline Phosphotase", "type": "number", "step": "1", "hint": "Normal: 44-147"},
            {"name": "Alamine_Aminotransferase", "label": "ALT (SGPT)", "type": "number", "step": "1", "hint": "Normal: 7-56"},
            {"name": "Aspartate_Aminotransferase", "label": "AST (SGOT)", "type": "number", "step": "1", "hint": "Normal: 8-48"},
            {"name": "Total_Protiens", "label": "Total Proteins", "type": "number", "step": "0.1", "hint": "Normal: 6.0-8.3"},
            {"name": "Albumin", "label": "Albumin", "type": "number", "step": "0.1", "hint": "Normal: 3.5-5.0"},
            {"name": "Albumin_and_Globulin_Ratio", "label": "Albumin/Globulin Ratio", "type": "number", "step": "0.01", "hint": "Normal: 1.0-2.5"},
        ],
    },
    "kidney": {
        "label": "Kidney Disease",
        "icon": "🫘",
        "color": "#FFB74D",
        "dataset": "kidney_disease.csv",
        "model_file": "kidney_model.pkl",
        "features": [
            {"name": "age", "label": "Age", "type": "number", "step": "1"},
            {"name": "blood_pressure", "label": "Blood Pressure", "type": "number", "step": "0.1"},
            {"name": "specific_gravity", "label": "Urine Specific Gravity", "type": "number", "step": "0.001", "hint": "Normal: 1.005-1.030"},
            {"name": "albumin", "label": "Albumin (0-5)", "type": "select", "options": [("0", "0"), ("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5")]},
            {"name": "sugar", "label": "Sugar (0-5)", "type": "select", "options": [("0", "0"), ("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5")]},
            {"name": "blood_glucose_random", "label": "Random Blood Glucose", "type": "number", "step": "0.1"},
            {"name": "blood_urea", "label": "Blood Urea", "type": "number", "step": "0.1", "hint": "Normal: 7-20"},
            {"name": "serum_creatinine", "label": "Serum Creatinine", "type": "number", "step": "0.01", "hint": "Normal: 0.6-1.3"},
            {"name": "sodium", "label": "Sodium", "type": "number", "step": "0.1", "hint": "Normal: 135-145"},
            {"name": "potassium", "label": "Potassium", "type": "number", "step": "0.1", "hint": "Normal: 3.5-5.0"},
            {"name": "haemoglobin", "label": "Haemoglobin", "type": "number", "step": "0.1", "hint": "Normal: 13.5-17.5"},
        ],
    },
}


def feature_names(disease_key):
    return [f["name"] for f in DISEASES[disease_key]["features"]]


def friendly_label(disease_key, field_name):
    for f in DISEASES[disease_key]["features"]:
        if f["name"] == field_name:
            return f["label"]
    return field_name
