from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import logging
import traceback

app = Flask(__name__) 

logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    gpa_model = joblib.load(
        os.path.join(BASE_DIR, "models", "regression", "histgb_gpa_pipeline.pkl")
    )
    dropout_model = joblib.load(
        os.path.join(BASE_DIR, "models", "classification", "logistic_dropout_pipeline.pkl")
    )
except Exception as e:
    logging.error("Model loading failed")
    raise e

REQUIRED_FEATURES = [
    "AttendanceRate",
    "StudyHours",
    "ParentSupport",
    "Avg_TestScore",
    "ScoreStd",
    "SocialDistraction",
    "Grade",
    "ParentalEducation",
    "FreeTime",
    "Extracurricular",
    "SES_Quartile",
    "Race",
    "SchoolType",
    "Age",
    "StudyEfficiency",
    "Locale",
    "InternetAccess",
    "Gender",
    "LowAttendance"
]

def validate_input_schema(input_json):
    missing = set(REQUIRED_FEATURES) - set(input_json.keys())
    extra = set(input_json.keys()) - set(REQUIRED_FEATURES)

    if missing:
        return {
            "error": "Missing required features",
            "missing_features": sorted(list(missing))
        }

    if extra:
        return {
            "error": "Unexpected extra features",
            "extra_features": sorted(list(extra))
        }

    return None

def get_risk_tier(risk):
    if risk < 0.25:
        return "Low Risk"
    elif risk < 0.5:
        return "Moderate Risk"
    elif risk < 0.75:
        return "High Risk"
    else:
        return "Critical Risk"

def get_intervention_level(risk_tier):
    return {
        "Low Risk": "NONE",
        "Moderate Risk": "MONITOR",
        "High Risk": "SUPPORT",
        "Critical Risk": "URGENT"
    }[risk_tier]

def get_key_risk_drivers(row):
    drivers = []
    if row["AttendanceRate"] < 0.75:
        drivers.append("Low Attendance")
    if row["ParentSupport"] == 0:
        drivers.append("Low Parent Support")
    if row["StudyHours"] < 2:
        drivers.append("Low Study Hours")
    if row["SocialDistraction"] == 1:
        drivers.append("High Social Distraction")
    if row["ScoreStd"] > 15:
        drivers.append("Inconsistent Performance")
    if row["Avg_TestScore"] < 50:
        drivers.append("Low Effort with Weak Scores")
    return drivers

def advisory_notes(drivers):
    notes_map = {
        "Low Attendance": "Improve attendance consistency",
        "Low Parent Support": "Engage parent or guardian",
        "Low Study Hours": "Increase focused study time",
        "High Social Distraction": "Reduce digital/social distractions",
        "Inconsistent Performance": "Stabilize academic routine",
        "Low Effort with Weak Scores": "Boost motivation and academic discipline"
    }
    return [notes_map[d] for d in drivers if d in notes_map]

def focus_areas(drivers):
    areas = set()
    if "Low Attendance" in drivers or "Low Study Hours" in drivers:
        areas.add("Academics")
    if "Low Parent Support" in drivers:
        areas.add("Family Support")
    if "Low Effort with Weak Scores" in drivers:
        areas.add("Motivation")
    if "High Social Distraction" in drivers:
        areas.add("Behavioral Discipline")
    return list(areas)

def rule_based_action(risk_tier):
    if risk_tier == "Critical Risk":
        return "Immediate counseling and intervention"
    if risk_tier == "High Risk":
        return "Targeted academic and behavioral support"
    if risk_tier == "Moderate Risk":
        return "Regular monitoring and mentoring"
    return "No intervention required"

def optimized_action(dropout_prob):
    if dropout_prob > 0.8:
        return "Intensive Intervention"
    if dropout_prob > 0.5:
        return "Targeted Support"
    return "No Action"

def student_intelligence_pipeline(student_json):
    df = pd.DataFrame([student_json], columns=REQUIRED_FEATURES)
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isnull().any().any():
        raise ValueError("Invalid or non-numeric input values detected")

    gpa_pred = float(np.clip(gpa_model.predict(df)[0], 0, 4))
    dropout_prob = float(dropout_model.predict_proba(df)[0][1])

    gpa_risk = max(0, (3.0 - gpa_pred) / 3.0)
    total_risk = 0.6 * dropout_prob + 0.4 * gpa_risk

    risk_tier = get_risk_tier(total_risk)
    intervention_level = get_intervention_level(risk_tier)

    drivers = get_key_risk_drivers(student_json)

    return {
        "GPA_Predicted": round(gpa_pred, 2),
        "Dropout_Probability": round(dropout_prob, 3),
        "Risk_Tier": risk_tier,
        "Intervention_Level": intervention_level,
        "Key_Risk_Drivers": drivers,
        "Advisory_Notes": advisory_notes(drivers),
        "Suggested_Focus_Areas": focus_areas(drivers),
        "Rule_Based_Action": rule_based_action(risk_tier),
        "Optimized_Action": optimized_action(dropout_prob),
        "Policy_Utility_Score": round(total_risk, 3),
        "Confidence": round(max(dropout_prob, 1 - dropout_prob), 2)
    }

@app.route("/")
def home():
    return jsonify({
        "status": "Student Risk & Performance Intelligence System running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        schema_error = validate_input_schema(data)
        if schema_error:
            return jsonify(schema_error), 400

        logging.info(f"Incoming request: {data}")

        result = student_intelligence_pipeline(data)
        return jsonify(result)

    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
