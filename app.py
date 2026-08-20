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
    "gender": ["Male", "Female", "Other"],
    "family_income_level": ["Low", "Medium", "High"],
    "parental_education": ["None", "Secondary", "Bachelor", "Postgraduate"],
}