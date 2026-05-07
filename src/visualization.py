from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

from data_loader import load_heart_dataset
from predict_service import FEATURE_ORDER


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
IMAGES_DIR = ROOT / "app" / "static" / "images"


def _plot_metrics_chart(metrics: dict) -> None:
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    lr_vals = [
        metrics["logistic_regression"]["accuracy"],
        metrics["logistic_regression"]["precision"],
        metrics["logistic_regression"]["recall"],
        metrics["logistic_regression"]["f1_score"],
    ]
    rf_vals = [
        metrics["random_forest"]["accuracy"],
        metrics["random_forest"]["precision"],
        metrics["random_forest"]["recall"],
        metrics["random_forest"]["f1_score"],
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, lr_vals, width, label="Logistic Regression")
    ax.bar(x + width / 2, rf_vals, width, label="Random Forest")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Model Metrics Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "model_metrics.png", dpi=160)
    plt.close(fig)


def _plot_confusion_matrix() -> None:
    X, y = load_heart_dataset()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = joblib.load(BEST_MODEL_PATH)
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not At Risk", "At Risk"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (Best Model)")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def _plot_feature_importance() -> None:
    model = joblib.load(BEST_MODEL_PATH)
    estimator = model.named_steps.get("model", model)
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_[0])
    else:
        importance = np.zeros(len(FEATURE_ORDER))

    order = np.argsort(importance)[::-1]
    features = [FEATURE_ORDER[i] for i in order]
    values = [importance[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(features, values)
    ax.set_title("Feature Importance (Best Model)")
    ax.set_ylabel("Relative Importance")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)


def ensure_visual_assets() -> None:
    if not METRICS_PATH.exists() or not BEST_MODEL_PATH.exists():
        return
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    _plot_metrics_chart(metrics)
    _plot_confusion_matrix()
    _plot_feature_importance()
