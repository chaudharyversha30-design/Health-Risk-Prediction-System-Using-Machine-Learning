# Health Risk Prediction ML (Flask + Scikit-learn)

Professional end-to-end machine learning project for predicting cardiovascular health risk using the UCI Heart Disease dataset.

## Features

- Binary classification: `At Risk` vs `Not At Risk`
- Baseline model: Logistic Regression
- Performance model: Random Forest
- Evaluation: Accuracy, Precision, Recall, F1-score
- Best-model selection prioritizing Recall
- Flask multi-page web app (Home, Predict, About, Model Info, History)
- Dashboard page with live system statistics
- Result page with prediction label, probability score, top contributing factors, and disclaimer
- One-click PDF assessment report download
- Risk-level badge (Low/Moderate/High) with prediction insights
- User-friendly dropdown inputs for encoded medical fields
- Input validation for required fields and safe value ranges
- SQLite prediction history with timestamp and confidence logging
- History filters (date/risk level/search) and CSV export
- Model evaluation charts: metrics comparison, confusion matrix, feature importance
- Evaluation depth: ROC-AUC, confusion matrix values, 5-fold cross-validation
- JSON API endpoint for integration (`/api/predict`)

## Project Structure

```text
health-risk-prediction-ml/
├─ app/
│  ├─ app.py
│  ├─ static/css/styles.css
│  └─ templates/
│     ├─ base.html
│     ├─ home.html
│     ├─ dashboard.html
│     ├─ predict.html
│     ├─ result.html
│     ├─ about.html
│     ├─ model_info.html
│     └─ history.html
├─ data/
├─ models/
├─ src/
│  ├─ data_loader.py
│  ├─ predict_service.py
│  ├─ visualization.py
│  └─ train.py
├─ tests/
├─ requirements.txt
└─ README.md
```

## 1) Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Train Models

```bash
python src/train.py
```

What this does:
- Loads local dataset if present at `data/heart_disease.csv`
- If missing, downloads from UCI via `ucimlrepo`
- Trains Logistic Regression and Random Forest
- Saves:
  - `models/logistic_regression.joblib`
  - `models/best_model.joblib`
  - `models/metrics.json`

## 3) Run the Web App

```bash
python app/app.py
```

You can optionally configure host/port/debug via `.env`:

```bash
cp .env.example .env
```

Open:
- Web UI: `http://127.0.0.1:5000`
- Dashboard: `http://127.0.0.1:5000/dashboard`
- Predict page: `http://127.0.0.1:5000/predict`
- About page: `http://127.0.0.1:5000/about`
- Model info page: `http://127.0.0.1:5000/model-info`
- History page: `http://127.0.0.1:5000/history`

## 4) Use the API

POST request to `http://127.0.0.1:5000/api/predict`

Example body:

```json
{
  "age": 57,
  "sex": 1,
  "cp": 2,
  "trestbps": 130,
  "chol": 236,
  "fbs": 0,
  "restecg": 1,
  "thalach": 174,
  "exang": 0,
  "oldpeak": 0.0,
  "slope": 2,
  "ca": 1,
  "thal": 2
}
```

## 5) Run Tests

```bash
pytest
```

## 6) Production-style Run

Windows (Waitress):

```bash
waitress-serve --host=0.0.0.0 --port=8000 app.app:app
```

Linux/macOS (Gunicorn):

```bash
gunicorn -w 2 -b 0.0.0.0:8000 app.app:app
```

Deployment options:
- Render
- Railway
- Fly.io

## Notes

- This project is for educational and research use only.
- It is not a substitute for professional medical diagnosis.
