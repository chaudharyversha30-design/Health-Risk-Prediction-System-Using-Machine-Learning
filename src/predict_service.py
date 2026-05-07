from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"

FEATURE_ORDER = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


class PredictionService:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                "Model not found. Run `python src/train.py` before starting the web app."
            )
        self.model = joblib.load(model_path)

    def predict(self, features: Dict[str, float]) -> Dict[str, float | int | str]:
        data = pd.DataFrame([[features[name] for name in FEATURE_ORDER]], columns=FEATURE_ORDER)
        pred = int(self.model.predict(data)[0])

        confidence = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(data)[0]
            confidence = float(max(probs))

        label = "At Risk" if pred == 1 else "Not At Risk"
        return {"prediction": pred, "label": label, "confidence": confidence}
