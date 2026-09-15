"""
app.py
AI-HealthGuard - Flask web app for multi-disease risk prediction
(Diabetes, Heart, Liver, Kidney) with explainable AI, prediction
history, downloadable PDF reports, and a built-in health chatbot.

Run:
    python training/train_model.py   # once, to create the model .pkl files
    python app.py                    # starts the web server
"""

import os
import json
import sqlite3
import joblib
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, g, send_file
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

from utils.disease_config import DISEASES, feature_names
from utils.preprocessing import transform_input
from utils.explainable_ai import explain_prediction, top_reasons_text
from utils.report_generator import generate_pdf_report
from utils.chatbot import answer as chatbot_answer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "database.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "static", "reports")

app = Flask(__name__)
app.secret_key = os.environ.get("HEALTHGUARD_SECRET_KEY", "dev-secret-change-me")
app.jinja_env.globals.update(DISEASES=DISEASES)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# ---------------------------------------------------------------- database
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            disease_key TEXT NOT NULL DEFAULT 'diabetes',
            input_json TEXT NOT NULL,
            result TEXT NOT NULL,
            probability REAL NOT NULL,
            top_reasons TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.commit()
    conn.close()


# ------------------------------------------------------------ auth (Flask-Login)
class User(UserMixin):
    def __init__(self, row):
        self.id = str(row["id"])
        self.username = row["username"]


@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return User(row) if row else None


# ------------------------------------------------------------------- model
_model_cache = {}


def get_model_bundle(disease_key):
    if disease_key not in _model_cache:
        model_path = os.path.join(MODELS_DIR, DISEASES[disease_key]["model_file"])
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model for '{disease_key}' not found. Run `python training/train_model.py` first."
            )
        _model_cache[disease_key] = joblib.load(model_path)
    return _model_cache[disease_key]


def valid_disease_or_404(disease_key):
    if disease_key not in DISEASES:
        flash("Unknown disease type.", "error")
        return False
    return True


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return redirect(url_for("dashboard") if current_user.is_authenticated else url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        db = get_db()
        existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            flash("Username already taken.", "error")
            return redirect(url_for("signup"))
        db.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), datetime.now().isoformat()),
        )
        db.commit()
        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("login.html", mode="signup")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        db = get_db()
        row = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row and check_password_hash(row["password_hash"], password):
            login_user(User(row))
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html", mode="login")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    total = db.execute(
        "SELECT COUNT(*) c FROM predictions WHERE user_id = ?", (current_user.id,)
    ).fetchone()["c"]
    positive = db.execute(
        "SELECT COUNT(*) c FROM predictions WHERE user_id = ? AND result = 'Positive'",
        (current_user.id,),
    ).fetchone()["c"]
    recent = db.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 5",
        (current_user.id,),
    ).fetchall()
    # per-disease counts, for the dashboard cards
    per_disease = {}
    for key in DISEASES:
        c = db.execute(
            "SELECT COUNT(*) c FROM predictions WHERE user_id = ? AND disease_key = ?",
            (current_user.id, key),
        ).fetchone()["c"]
        per_disease[key] = c
    return render_template("dashboard.html", total=total, positive=positive,
                            recent=recent, per_disease=per_disease)


@app.route("/predict")
@login_required
def disease_select():
    """Step 1: choose which disease to screen for."""
    return render_template("disease_select.html")


@app.route("/predict/<disease_key>", methods=["GET", "POST"])
@login_required
def prediction(disease_key):
    if not valid_disease_or_404(disease_key):
        return redirect(url_for("disease_select"))

    if request.method == "POST":
        bundle = get_model_bundle(disease_key)
        model, scaler = bundle["model"], bundle["scaler"]
        cols = feature_names(disease_key)

        raw_form = {col: request.form.get(col, 0) for col in cols}
        scaled_input, cleaned_row = transform_input(raw_form, scaler, disease_key)

        pred = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0][1]
        result = "Positive" if pred == 1 else "Negative"

        explanations = explain_prediction(model, scaled_input, cols, cleaned_row, disease_key)
        reasons = top_reasons_text(explanations)

        db = get_db()
        db.execute(
            """INSERT INTO predictions
               (user_id, disease_key, input_json, result, probability, top_reasons, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (current_user.id, disease_key, json.dumps(raw_form), result, float(proba),
             reasons, datetime.now().isoformat()),
        )
        db.commit()
        prediction_id = db.execute("SELECT last_insert_rowid() id").fetchone()["id"]

        return redirect(url_for("result", prediction_id=prediction_id))

    disease = DISEASES[disease_key]
    return render_template("prediction.html", disease_key=disease_key, disease=disease)


@app.route("/result/<int:prediction_id>")
@login_required
def result(prediction_id):
    db = get_db()
    row = db.execute(
        "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
        (prediction_id, current_user.id),
    ).fetchone()
    if not row:
        flash("Prediction not found.", "error")
        return redirect(url_for("dashboard"))
    input_data = json.loads(row["input_json"])
    disease = DISEASES[row["disease_key"]]
    return render_template(
        "result.html",
        prediction_id=prediction_id,
        disease_key=row["disease_key"],
        disease=disease,
        result=row["result"],
        probability=round(row["probability"] * 100, 1),
        input_data=input_data,
        top_reasons=row["top_reasons"],
    )


@app.route("/history")
@login_required
def history():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC",
        (current_user.id,),
    ).fetchall()
    return render_template("history.html", rows=rows)


@app.route("/report/<int:prediction_id>")
@login_required
def report(prediction_id):
    db = get_db()
    row = db.execute(
        "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
        (prediction_id, current_user.id),
    ).fetchone()
    if not row:
        flash("Prediction not found.", "error")
        return redirect(url_for("dashboard"))

    disease_key = row["disease_key"]
    bundle = get_model_bundle(disease_key)
    model, scaler = bundle["model"], bundle["scaler"]
    input_data = json.loads(row["input_json"])
    scaled_input, cleaned_row = transform_input(input_data, scaler, disease_key)
    explanations = explain_prediction(model, scaled_input, feature_names(disease_key), cleaned_row, disease_key)

    pdf_path = os.path.join(REPORTS_DIR, f"report_{prediction_id}.pdf")
    generate_pdf_report(
        pdf_path, current_user.username, disease_key, input_data,
        row["result"], row["probability"], explanations,
    )
    return render_template("report.html", prediction_id=prediction_id,
                            result=row["result"], disease=DISEASES[disease_key])


@app.route("/report/<int:prediction_id>/download")
@login_required
def report_download(prediction_id):
    pdf_path = os.path.join(REPORTS_DIR, f"report_{prediction_id}.pdf")
    if not os.path.exists(pdf_path):
        return redirect(url_for("report", prediction_id=prediction_id))
    return send_file(pdf_path, as_attachment=True, download_name=f"HealthGuard_Report_{prediction_id}.pdf")


# ------------------------------------------------------------------ chatbot
@app.route("/chatbot")
@login_required
def chatbot_page():
    return render_template("chatbot.html")


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    message = (request.json or {}).get("message", "")

    db = get_db()
    last = db.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
        (current_user.id,),
    ).fetchone()
    history_summary = None
    if last:
        disease_label = DISEASES[last["disease_key"]]["label"]
        history_summary = (
            f"Your most recent screening was for {disease_label}, result: {last['result']} "
            f"({round(last['probability'] * 100, 1)}% probability), on {last['created_at'][:16]}."
        )

    reply = chatbot_answer(message, history_summary)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    init_db()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    app.run(debug=True)
