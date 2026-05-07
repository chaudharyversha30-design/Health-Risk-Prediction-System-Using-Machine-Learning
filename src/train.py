from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_heart_dataset


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
BASELINE_PATH = MODELS_DIR / "logistic_regression.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)

    cm = confusion_matrix(y_test, preds)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1]),
        },
        "classification_report": classification_report(y_test, preds, zero_division=0),
    }
    print(f"\n{name} metrics:")
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"  {key}: {metrics[key]:.4f}")
    return metrics


def cross_validate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        n_jobs=None,
    )
    return {
        "accuracy_mean": float(scores["test_accuracy"].mean()),
        "precision_mean": float(scores["test_precision"].mean()),
        "recall_mean": float(scores["test_recall"].mean()),
        "f1_mean": float(scores["test_f1"].mean()),
        "roc_auc_mean": float(scores["test_roc_auc"].mean()),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_heart_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logistic_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    logistic_pipeline.fit(X_train, y_train)
    logistic_metrics = evaluate_model("Logistic Regression", logistic_pipeline, X_test, y_test)
    logistic_cv = cross_validate_model(logistic_pipeline, X, y)
    joblib.dump(logistic_pipeline, BASELINE_PATH)

    rf_pipeline = Pipeline(
        steps=[
            ("model", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    rf_metrics = evaluate_model("Random Forest", rf_pipeline, X_test, y_test)
    rf_cv = cross_validate_model(rf_pipeline, X, y)

    # Pick model prioritizing recall, then F1.
    best_model = rf_pipeline
    best_name = "random_forest"
    if (
        logistic_metrics["recall"] > rf_metrics["recall"]
        or (
            logistic_metrics["recall"] == rf_metrics["recall"]
            and logistic_metrics["f1_score"] > rf_metrics["f1_score"]
        )
    ):
        best_model = logistic_pipeline
        best_name = "logistic_regression"

    joblib.dump(best_model, BEST_MODEL_PATH)

    payload = {
        "selected_model": best_name,
        "logistic_regression": {k: v for k, v in logistic_metrics.items() if k != "classification_report"},
        "random_forest": {k: v for k, v in rf_metrics.items() if k != "classification_report"},
        "cross_validation": {
            "logistic_regression": logistic_cv,
            "random_forest": rf_cv,
        },
        "interpretation": (
            "Recall is prioritized in this project because identifying at-risk patients is more important "
            "than minimizing false positives. Missing a high-risk case may delay intervention."
        ),
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved best model: {best_name} -> {BEST_MODEL_PATH}")
    print(f"Saved metrics file -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
