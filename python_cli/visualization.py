import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from python_cli.utils import ensure_parent_dir


def save_classification_comparison_plot(results, output_path: str) -> None:
    model_names = [r.model_name for r in results]
    f1_scores = [r.metrics["f1"] for r in results]

    ensure_parent_dir(output_path)
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, f1_scores)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("F1 Score")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_regression_comparison_plot(results, output_path: str) -> None:
    model_names = [r.model_name for r in results]
    rmse_scores = [r.metrics["rmse"] for r in results]

    ensure_parent_dir(output_path)
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, rmse_scores)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("RMSE")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix_plot(y_true, y_pred, output_path: str) -> None:
    ensure_parent_dir(output_path)
    plt.figure(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_regression_scatter_plot(y_true, y_pred, output_path: str) -> None:
    ensure_parent_dir(output_path)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_roc_curve_plot(pipeline, x_test, y_test, output_path: str) -> None:
    if not hasattr(pipeline, "predict_proba"):
        return

    probabilities = pipeline.predict_proba(x_test)[:, 1]

    ensure_parent_dir(output_path)
    plt.figure(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_feature_importance_plot(pipeline, feature_names, output_path: str) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    top_k = min(10, len(importances))
    indices = np.argsort(importances)[-top_k:][::-1]

    selected_names = [feature_names[i] for i in indices]
    selected_values = importances[indices]

    ensure_parent_dir(output_path)
    plt.figure(figsize=(10, 6))
    plt.bar(selected_names, selected_values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()