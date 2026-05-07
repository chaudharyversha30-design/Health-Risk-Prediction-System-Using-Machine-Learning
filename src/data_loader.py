from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DATA_PATH = DATA_DIR / "heart_disease.csv"
UCI_FOLDER_PATH = DATA_DIR / "heart+disease" / "processed.cleveland.data"

HEART_COLUMNS = [
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
    "target",
]


def _clean_heart_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean = clean.replace("?", pd.NA).dropna()
    clean.columns = HEART_COLUMNS

    for col in HEART_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna().reset_index(drop=True)
    clean["target"] = (clean["target"] > 0).astype(int)
    return clean


def download_uci_heart_dataset(output_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Download the UCI Heart Disease dataset via ucimlrepo and save as CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "ucimlrepo is required to download dataset. Install dependencies first."
        ) from exc

    dataset = fetch_ucirepo(id=45)
    features = dataset.data.features
    target = dataset.data.targets

    target_name = target.columns[0]
    combined = pd.concat([features, target.rename(columns={target_name: "target"})], axis=1)
    combined = combined.rename(
        columns={
            "thalach": "thalach",
            "trestbps": "trestbps",
        }
    )

    cleaned = combined.copy()
    cleaned = cleaned.replace("?", pd.NA).dropna()
    cleaned["target"] = (pd.to_numeric(cleaned["target"], errors="coerce") > 0).astype(int)
    cleaned = cleaned.dropna().reset_index(drop=True)

    # Ensure the expected schema order for the rest of the project.
    cleaned = cleaned[
        [
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
            "target",
        ]
    ]
    cleaned.to_csv(output_path, index=False)
    return cleaned


def load_heart_dataset(csv_path: Path = RAW_DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset with this priority:
    1) cleaned local CSV (heart_disease.csv)
    2) local raw UCI file (heart+disease/processed.cleveland.data)
    3) auto-download from UCI using ucimlrepo
    """
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif UCI_FOLDER_PATH.exists():
        df = pd.read_csv(UCI_FOLDER_PATH, header=None)
    else:
        df = download_uci_heart_dataset(csv_path)

    if "target" not in df.columns:
        if df.shape[1] == 14:
            df.columns = HEART_COLUMNS
        else:
            raise ValueError("Dataset must include a 'target' column.")

    if df.shape[1] == 14 and list(df.columns) != HEART_COLUMNS:
        df.columns = HEART_COLUMNS

    df = _clean_heart_dataframe(df)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y
