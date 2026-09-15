"""
chatbot.py
A lightweight, rule-based "HealthGuard Assistant" chatbot. No external API
needed — it matches keywords in the user's message to answer general
health FAQs, explain medical terms used in the app, and (when given
access to the user's prediction history) answer questions about their
own past results.

This is intentionally simple/offline so the app works with zero extra
setup. Swap `answer()` for a real LLM API call later if desired.
"""

import re

FAQ = [
    (["glucose", "sugar level", "blood sugar"],
     "Fasting blood glucose is normally 70-100 mg/dL. 100-125 is prediabetic range, "
     "126+ on two separate tests usually indicates diabetes. This app's Diabetes "
     "predictor uses your glucose value as one of its strongest signals."),

    (["bmi", "body mass index"],
     "BMI = weight(kg) / height(m)^2. Under 18.5 is underweight, 18.5-24.9 is normal, "
     "25-29.9 is overweight, and 30+ is obese. Higher BMI raises diabetes and heart "
     "disease risk in the models here."),

    (["blood pressure", "bp", "hypertension"],
     "A normal resting blood pressure is around 120/80 mm Hg. 130/80 and above is "
     "considered high blood pressure (hypertension), a major risk factor for both "
     "heart and kidney disease."),

    (["cholesterol", "chol"],
     "Total cholesterol under 200 mg/dL is desirable, 200-239 is borderline high, and "
     "240+ is high. High cholesterol is one of the inputs the Heart Disease predictor "
     "uses."),

    (["creatinine", "kidney function"],
     "Serum creatinine is normally 0.6-1.3 mg/dL. Higher values can indicate reduced "
     "kidney function, which is why it's a key input to the Kidney Disease predictor."),

    (["bilirubin", "liver function", "sgot", "sgpt", "alt", "ast"],
     "Bilirubin, ALT (SGPT) and AST (SGOT) are liver enzymes/markers. Elevated levels "
     "can indicate liver stress or damage, and are core inputs to the Liver Disease "
     "predictor here."),

    (["accurate", "accuracy", "reliable", "trust"],
     "These models are trained on structured datasets and give a probability, not a "
     "certain diagnosis. They're meant for early screening/awareness — always confirm "
     "any concerning result with a qualified doctor and proper lab tests."),

    (["how does this work", "how it works", "how does the app work", "explainable"],
     "You enter your health parameters, the trained model predicts a risk probability, "
     "and the Explainable AI section shows which specific values pushed that risk up "
     "or down — so you're not just getting a yes/no, but the 'why'."),

    (["symptom", "diabetes symptom"],
     "Common diabetes symptoms include frequent urination, excess thirst, unexplained "
     "weight loss, fatigue, and blurred vision. This isn't a diagnosis — get tested if "
     "you notice these."),

    (["heart disease symptom", "chest pain"],
     "Warning signs of heart disease include chest pain/pressure, shortness of breath, "
     "pain radiating to the arm/jaw, and unusual fatigue. Seek medical help immediately "
     "for sudden chest pain."),

    (["diet", "food", "eat"],
     "General healthy-eating guidance: more vegetables, whole grains and lean protein, "
     "less refined sugar and salt, and regular portion control. For a condition-specific "
     "diet plan, please consult a dietitian or doctor."),

    (["exercise", "workout"],
     "Around 150 minutes of moderate activity a week (brisk walking, cycling, etc.) is "
     "a commonly recommended general guideline for heart and metabolic health — check "
     "with a doctor first if you have an existing condition."),
]

GREETINGS = ["hi", "hello", "hey", "namaste", "helo"]
THANKS = ["thanks", "thank you", "thanku", "shukriya"]


def _match_history_question(message: str, history_summary: str | None) -> str | None:
    triggers = ["my result", "my prediction", "my history", "my last", "my report",
                "what was my", "my risk"]
    if any(t in message for t in triggers):
        if history_summary:
            return f"Here's what I have on record for you: {history_summary}"
        return ("You don't have any saved predictions yet. Go to **New Prediction**, "
                "pick a disease, and submit your parameters — I'll be able to reference "
                "it here afterwards.")
    return None


def answer(user_message: str, history_summary: str | None = None) -> str:
    if not user_message or not user_message.strip():
        return "Please type a question — I can help with health terms, risk factors, or your past results."

    msg = user_message.lower().strip()

    if any(re.fullmatch(rf"{g}[!. ]*", msg) for g in GREETINGS) or msg in GREETINGS:
        return ("Hi! I'm the HealthGuard Assistant 🤖. Ask me about glucose, BMI, blood "
                "pressure, cholesterol, kidney/liver markers, or your past predictions.")

    if any(t in msg for t in THANKS):
        return "You're welcome! Stay healthy 💙 — ask me anything else anytime."

    history_reply = _match_history_question(msg, history_summary)
    if history_reply:
        return history_reply

    for keywords, response in FAQ:
        if any(k in msg for k in keywords):
            return response

    return ("I'm not sure about that one yet — I can currently help with glucose, BMI, "
            "blood pressure, cholesterol, kidney/liver markers, general symptoms, diet/"
            "exercise basics, and questions about your own past predictions. For anything "
            "else, please consult a doctor.")
