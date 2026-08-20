"""
Flask app for the AI Student Dropout Predictor.
- GET  /            -> HTML form
- POST /predict      -> HTML form submission, renders result
- POST /api/predict  -> JSON API, returns risk score + label
"""
import json
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

with open("model/feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

CAT_OPTIONS = {
    "gender": ["Male", "Female"],
    "family_income_level": ["Low", "Medium", "High"],
    "parental_education": ["None", "Secondary", "Bachelor", "Postgraduate"],
}

def build_feature_row(form: dict) -> pd.DataFrame:
    raw = {
        "age": float(form["age"]),
        "gender": form["gender"],
        "family_income_level": form["family_income_level"],
        "parental_education": form["parental_education"],
        "distance_from_home_km": float(form["distance_from_home_km"]),
        "has_scholarship": int(form.get("has_scholarship", 0)),
        "part_time_job": int(form.get("part_time_job", 0)),
        "high_school_gpa": float(form["high_school_gpa"]),
        "entrance_exam_score": float(form["entrance_exam_score"]),
        "prior_failures": int(form["prior_failures"]),
        "semester": int(form["semester"]),
        "attendance_rate": float(form["attendance_rate"]),
        "study_hours_per_week": float(form["study_hours_per_week"]),
        "current_gpa": float(form["current_gpa"]),
        "lms_login_freq_per_week": float(form["lms_login_freq_per_week"]),
        "extracurricular_activities": int(form.get("extracurricular_activities", 0)),
        "counseling_visits": int(form["counseling_visits"]),
    }
    df = pd.DataFrame([raw])
    df_enc = pd.get_dummies(df, columns=["gender", "family_income_level", "parental_education"])
    # Align to training-time feature columns (missing dummy cols -> 0)
    df_enc = df_enc.reindex(columns=FEATURE_COLS, fill_value=0)
    return df_enc

def predict(form: dict):
    X = build_feature_row(form)
    X_s = scaler.transform(X)
    proba = float(model.predict_proba(X_s)[0, 1])
    label = "High Risk" if proba >= 0.5 else "Low Risk"
    return proba, label

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", cat_options=CAT_OPTIONS, result=None)


@app.route("/predict", methods=["POST"])
def predict_form():
    proba, label = predict(request.form)
    return render_template(
        "index.html",
        cat_options=CAT_OPTIONS,
        result={"probability": round(proba * 100, 1), "label": label},
        form=request.form,
    )


@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    proba, label = predict(data)
    return jsonify({"dropout_probability": round(proba, 4), "risk_label": label})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001, debug=True)