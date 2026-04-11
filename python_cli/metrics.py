import numpy as np
import pandas as pd
import math
from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    average = "binary" if pd.Series(y_true).nunique() == 2 else "weighted"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_default_metric(task_type: str) -> str:
    return "f1" if task_type == "classification" else "rmse"


def is_higher_better(metric_name: str) -> bool:
    return metric_name in {"accuracy", "precision", "recall", "f1", "r2"}

