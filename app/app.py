from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
MODELS_PATH = ROOT / "models" / "metrics.json"
DB_PATH = ROOT / "data" / "predictions.db"
IMAGES_DIR = ROOT / "app" / "static" / "images"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
load_dotenv(ROOT / ".env")

from predict_service import FEATURE_ORDER, PredictionService  # noqa: E402
from data_loader import load_heart_dataset  # noqa: E402
from visualization import ensure_visual_assets  # noqa: E402


app = Flask(__name__)
service = PredictionService()
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression (oldpeak)",
    "slope": "Slope of Peak ST Segment",
    "ca": "Major Vessels (ca)",
    "thal": "Thal",
}


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                age REAL NOT NULL,
                sex REAL NOT NULL,
                cp REAL NOT NULL,
                trestbps REAL NOT NULL,
                chol REAL NOT NULL,
                fbs REAL NOT NULL,
                restecg REAL NOT NULL,
                thalach REAL NOT NULL,
                exang REAL NOT NULL,
                oldpeak REAL NOT NULL,
                slope REAL NOT NULL,
                ca REAL NOT NULL,
                thal REAL NOT NULL,
                prediction INTEGER NOT NULL,
                label TEXT NOT NULL,
                confidence REAL
            )
            """
        )
        conn.commit()


def validate_features(payload: dict) -> None:
    ranges = {
        "age": (1, 120),
        "sex": (0, 1),
        "cp": (1, 4),
        "trestbps": (1, 300),
        "chol": (1, 1000),
        "fbs": (0, 1),
        "restecg": (0, 2),
        "thalach": (1, 300),
        "exang": (0, 1),
        "oldpeak": (0, 10),
        "slope": (1, 3),
        "ca": (0, 4),
        "thal": (3, 7),
    }
    for key, value in payload.items():
        min_v, max_v = ranges[key]
        if value < min_v or value > max_v:
            raise ValueError(f"Invalid value for {key}. Expected between {min_v} and {max_v}.")


def save_prediction(payload: dict, result: dict) -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                created_at, age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal, prediction, label, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                payload["age"],
                payload["sex"],
                payload["cp"],
                payload["trestbps"],
                payload["chol"],
                payload["fbs"],
                payload["restecg"],
                payload["thalach"],
                payload["exang"],
                payload["oldpeak"],
                payload["slope"],
                payload["ca"],
                payload["thal"],
                result["prediction"],
                result["label"],
                result["confidence"],
            ),
        )
        conn.commit()


def load_metrics() -> dict:
    if MODELS_PATH.exists():
        return json.loads(MODELS_PATH.read_text(encoding="utf-8"))
    return {}


def risk_level(result: dict) -> str:
    confidence = float(result.get("confidence") or 0.0)
    prediction = int(result.get("prediction", 0))
    if prediction == 0:
        return "Low"
    if confidence < 0.8:
        return "Moderate"
    return "High"


def prediction_insights(payload: dict[str, float]) -> list[str]:
    insights = []
    if payload["chol"] >= 240:
        insights.append("Higher cholesterol is associated with increased cardiovascular risk.")
    if payload["trestbps"] >= 140:
        insights.append("Elevated resting blood pressure can contribute to cardiac risk.")
    if payload["oldpeak"] >= 2:
        insights.append("Higher ST depression (oldpeak) can indicate exercise-related cardiac stress.")
    if payload["exang"] == 1:
        insights.append("Exercise-induced angina may be a relevant risk indicator.")
    if payload["ca"] >= 2:
        insights.append("A higher number of major vessels (ca) can increase risk probability.")
    if not insights:
        insights.append("No single indicator is extreme; prediction comes from combined feature patterns.")
    return insights[:3]


def recommendations_for_level(level: str) -> list[str]:
    base = [
        "Maintain a balanced diet and reduce processed foods.",
        "Engage in regular physical activity based on your fitness level.",
        "Monitor blood pressure and cholesterol levels periodically.",
        "Consult a healthcare professional for personalized guidance.",
    ]
    if level == "High":
        base[3] = "Consult a healthcare professional promptly for full clinical evaluation."
    return base


def readable_input_summary(payload: dict[str, float]) -> list[tuple[str, str]]:
    sex_map = {0: "Female", 1: "Male"}
    cp_map = {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-anginal Pain",
        4: "Asymptomatic",
    }
    fbs_map = {0: "False", 1: "True"}
    restecg_map = {
        0: "Normal",
        1: "ST-T Abnormality",
        2: "Left Ventricular Hypertrophy",
    }
    exang_map = {0: "No", 1: "Yes"}
    slope_map = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
    thal_map = {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}

    rows = [
        ("Age", f"{int(payload['age'])} years"),
        ("Sex", sex_map.get(int(payload["sex"]), str(payload["sex"]))),
        ("Chest Pain Type", cp_map.get(int(payload["cp"]), str(payload["cp"]))),
        ("Resting Blood Pressure", f"{int(payload['trestbps'])} mm Hg"),
        ("Serum Cholesterol", f"{int(payload['chol'])} mg/dl"),
        ("Fasting Blood Sugar > 120 mg/dl", fbs_map.get(int(payload["fbs"]), str(payload["fbs"]))),
        ("Resting ECG", restecg_map.get(int(payload["restecg"]), str(payload["restecg"]))),
        ("Maximum Heart Rate", f"{int(payload['thalach'])} bpm"),
        ("Exercise Induced Angina", exang_map.get(int(payload["exang"]), str(payload["exang"]))),
        ("ST Depression (oldpeak)", f"{payload['oldpeak']:.1f}"),
        ("Slope of Peak ST Segment", slope_map.get(int(payload["slope"]), str(payload["slope"]))),
        ("Major Vessels (ca)", str(int(payload["ca"]))),
        ("Thal", thal_map.get(int(payload["thal"]), str(payload["thal"]))),
    ]
    return rows


def local_feature_explanations(payload: dict[str, float], top_n: int = 3) -> list[dict[str, str]]:
    try:
        X, _ = load_heart_dataset()
        means = X.mean()
        stds = X.std().replace(0, 1.0)
        model = joblib.load(ROOT / "models" / "best_model.joblib")
        estimator = model.named_steps.get("model", model)
        if hasattr(estimator, "feature_importances_"):
            weights = pd.Series(estimator.feature_importances_, index=FEATURE_ORDER)
        elif hasattr(estimator, "coef_"):
            weights = pd.Series(abs(estimator.coef_[0]), index=FEATURE_ORDER)
        else:
            weights = pd.Series(1.0, index=FEATURE_ORDER)

        z_scores = pd.Series(payload).sub(means).div(stds).abs()
        contribution = (z_scores * weights).sort_values(ascending=False).head(top_n)
        explanations = []
        for feature, score in contribution.items():
            direction = "higher" if payload[feature] >= means[feature] else "lower"
            explanations.append(
                {
                    "feature": FEATURE_LABELS.get(feature, feature),
                    "message": f"{FEATURE_LABELS.get(feature, feature)} is {direction} than dataset average.",
                    "score": f"{float(score):.3f}",
                }
            )
        return explanations
    except Exception:
        return []


def generate_pdf_report(context: dict[str, Any]) -> io.BytesIO:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    def line(text: str, gap: int = 18) -> None:
        nonlocal y
        pdf.drawString(50, y, text)
        y -= gap
        if y < 70:
            pdf.showPage()
            y = height - 50

    pdf.setTitle("Health Risk Assessment Report")
    pdf.setFont("Helvetica-Bold", 16)
    line("Health Risk Assessment Report", 24)
    pdf.setFont("Helvetica", 11)
    line(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    line("")
    line(f"Risk Assessment: {context['risk_level']} Cardiovascular Risk")
    line(f"Predicted Label: {context['result']['label']}")
    line(f"Confidence Score: {context['confidence_pct']}%")
    line("")
    line("Clinical Insights:")
    for item in context["insights"]:
        line(f"- {item}")
    line("")
    line("Recommended Actions:")
    for item in context["recommendations"]:
        line(f"- {item}")
    line("")
    line("Input Summary:")
    for label, value in context["summary_rows"]:
        line(f"- {label}: {value}")
    line("")
    line("Disclaimer:")
    line(
        "This system is intended for academic and educational purposes only and does not",
        14,
    )
    line("replace professional medical advice, diagnosis, or treatment.", 14)
    pdf.save()
    buffer.seek(0)
    return buffer


def dashboard_stats() -> dict[str, Any]:
    with get_db_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        high_risk = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction = 1"
        ).fetchone()[0]
        latest = conn.execute(
            "SELECT created_at FROM predictions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        recent = conn.execute(
            """
            SELECT created_at, label, confidence
            FROM predictions
            ORDER BY id DESC
            LIMIT 5
            """
        ).fetchall()

    metrics = load_metrics()
    best_model = metrics.get("selected_model", "N/A")
    best_accuracy = None
    if best_model in metrics and isinstance(metrics[best_model], dict):
        best_accuracy = metrics[best_model].get("accuracy")

    high_risk_pct = (high_risk / total * 100) if total else 0
    return {
        "total_predictions": total,
        "high_risk_cases": high_risk,
        "high_risk_pct": high_risk_pct,
        "latest_prediction_time": latest["created_at"] if latest else "N/A",
        "best_model": best_model,
        "best_accuracy": best_accuracy,
        "recent_predictions": recent,
    }


def fetch_history_rows(filters: dict[str, str], limit: int | None = 100) -> list[sqlite3.Row]:
    conditions = []
    params: list[Any] = []

    start_date = filters.get("start_date", "").strip()
    end_date = filters.get("end_date", "").strip()
    risk_level_filter = filters.get("risk_level", "").strip()
    search_text = filters.get("q", "").strip()

    if start_date:
        conditions.append("date(created_at) >= date(?)")
        params.append(start_date)
    if end_date:
        conditions.append("date(created_at) <= date(?)")
        params.append(end_date)
    if risk_level_filter == "high":
        conditions.append("prediction = 1 AND IFNULL(confidence, 0) >= 0.8")
    elif risk_level_filter == "moderate":
        conditions.append("prediction = 1 AND IFNULL(confidence, 0) < 0.8")
    elif risk_level_filter == "low":
        conditions.append("prediction = 0")
    if search_text:
        conditions.append("(created_at LIKE ? OR label LIKE ?)")
        params.extend([f"%{search_text}%", f"%{search_text}%"])

    query = """
        SELECT id, created_at, age, trestbps, chol, thalach, prediction, label, confidence
        FROM predictions
    """
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id DESC"
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    with get_db_connection() as conn:
        return conn.execute(query, params).fetchall()


def parse_form_values(form_data: dict) -> dict:
    parsed = {}
    for feature in FEATURE_ORDER:
        value = form_data.get(feature, "").strip()
        if value == "":
            raise ValueError(f"Missing required field: {feature}")
        parsed[feature] = float(value)
    validate_features(parsed)
    return parsed


@app.route("/", methods=["GET"])
def index():
    return render_template("home.html")


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = parse_form_values(request.form)
        result = service.predict(payload)
        level = risk_level(result)
        save_prediction(payload, result)
        summary_rows = readable_input_summary(payload)
        insights = prediction_insights(payload)
        recommendations = recommendations_for_level(level)
        top_factors = local_feature_explanations(payload, top_n=3)
        return render_template(
            "result.html",
            result=result,
            form_data=payload,
            confidence_pct=round(float(result.get("confidence") or 0.0) * 100, 2),
            risk_level=level,
            insights=insights,
            recommendations=recommendations,
            summary_rows=summary_rows,
            top_factors=top_factors,
            payload_json=json.dumps(payload),
        )
    except Exception as exc:
        return render_template("predict.html", error=str(exc), form_data=request.form), 400


@app.route("/about", methods=["GET"])
def about_page():
    return render_template("about.html")


@app.route("/model-info", methods=["GET"])
def model_info_page():
    ensure_visual_assets()
    return render_template(
        "model_info.html",
        metrics=load_metrics(),
        feature_count=len(FEATURE_ORDER),
        images={
            "metrics_chart": "images/model_metrics.png",
            "confusion_matrix": "images/confusion_matrix.png",
            "feature_importance": "images/feature_importance.png",
        },
    )


@app.route("/history", methods=["GET"])
def history_page():
    filters = {
        "start_date": request.args.get("start_date", ""),
        "end_date": request.args.get("end_date", ""),
        "risk_level": request.args.get("risk_level", ""),
        "q": request.args.get("q", ""),
    }
    rows = fetch_history_rows(filters, limit=300)
    return render_template("history.html", rows=rows, filters=filters)


@app.route("/history/export", methods=["GET"])
def export_history_csv():
    filters = {
        "start_date": request.args.get("start_date", ""),
        "end_date": request.args.get("end_date", ""),
        "risk_level": request.args.get("risk_level", ""),
        "q": request.args.get("q", ""),
    }
    rows = fetch_history_rows(filters, limit=None)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "created_at", "age", "trestbps", "chol", "thalach", "prediction", "label", "confidence"])
    for row in rows:
        writer.writerow(
            [
                row["id"],
                row["created_at"],
                row["age"],
                row["trestbps"],
                row["chol"],
                row["thalach"],
                row["prediction"],
                row["label"],
                row["confidence"],
            ]
        )
    mem = io.BytesIO(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="prediction_history.csv", mimetype="text/csv")


@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    return render_template("dashboard.html", stats=dashboard_stats())


@app.route("/api/predict", methods=["POST"])
def predict_api():
    try:
        json_payload = request.get_json(force=True) or {}
        payload = {k: float(v) for k, v in json_payload.items() if k in FEATURE_ORDER}
        missing = [name for name in FEATURE_ORDER if name not in payload]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400
        validate_features(payload)
        result = service.predict(payload)
        save_prediction(payload, result)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/download-report", methods=["POST"])
def download_report():
    payload_raw = request.form.get("payload_json", "")
    if not payload_raw:
        return redirect(url_for("predict_page"))
    payload = {k: float(v) for k, v in json.loads(payload_raw).items() if k in FEATURE_ORDER}
    result = service.predict(payload)
    level = risk_level(result)
    context = {
        "result": result,
        "confidence_pct": round(float(result.get("confidence") or 0.0) * 100, 2),
        "risk_level": level,
        "insights": prediction_insights(payload),
        "recommendations": recommendations_for_level(level),
        "summary_rows": readable_input_summary(payload),
    }
    pdf_bytes = generate_pdf_report(context)
    return send_file(
        pdf_bytes,
        as_attachment=True,
        download_name="health_risk_assessment_report.pdf",
        mimetype="application/pdf",
    )


@app.route("/clear-history", methods=["POST"])
def clear_history():
    with get_db_connection() as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()
    return redirect(url_for("history_page"))


init_db()

if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "true").lower() == "true",
    )
