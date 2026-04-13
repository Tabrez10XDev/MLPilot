import numpy as np
import pandas as pd
import joblib
import os
import uuid
from typing import Any, Dict
from python_cli.utils import ensure_parent_dir, get_dataset_name

def save_model_bundle(model_bundle: Dict[str, Any], model_path: str) -> None:
    ensure_parent_dir(model_path)
    joblib.dump(model_bundle, model_path)


def load_model_bundle(model_path: str) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model_bundle = joblib.load(model_path)
    except Exception as exc:
        raise ValueError(
            f"Invalid model file: {model_path}. The file is corrupted or not a valid MLPilot model bundle."
        ) from exc

    if not isinstance(model_bundle, dict):
        raise ValueError(
            f"Invalid model file: {model_path}. Expected a model bundle dictionary."
        )

    required_keys = {"model_id", "model_name", "task_type", "pipeline", "target_column"}
    missing_keys = required_keys - set(model_bundle.keys())
    if missing_keys:
        raise ValueError(f"Saved model bundle is missing keys: {sorted(missing_keys)}")

    return model_bundle

def generate_model_id() -> str:
    return uuid.uuid4().hex[:8]

def resolve_model_path(data_path: str, save_path: str | None, model_id: str) -> str:
    if save_path:
        return save_path
    dataset_name = get_dataset_name(data_path)
    filename = f"{dataset_name}_model_{model_id}.pkl"
    return os.path.join("models", filename)

def save_predictions(
    input_df: pd.DataFrame,
    predictions: np.ndarray,
    output_path: str,
    model_id: str,
    probabilities: np.ndarray | None = None,
) -> None:
    ensure_parent_dir(output_path)

    output_df = input_df.copy()
    output_df["prediction"] = predictions
    output_df["model_id"] = model_id

    if probabilities is not None:
        output_df["confidence"] = probabilities

    output_df.to_csv(output_path, index=False)


