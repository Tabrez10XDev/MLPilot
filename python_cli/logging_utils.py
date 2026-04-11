import json
import os
from typing import List, Dict, Any

from python_cli.utils import ensure_parent_dir
from python_cli.config import TrainConfig, ModelResult


def save_training_report(
    config: TrainConfig,
    task_type: str,
    primary_metric: str,
    results: List[ModelResult],
    model_id: str,
    model_path: str,
    interface: str,
) -> None:
    ensure_parent_dir(config.output_path)
    payload = {
        "model_id": model_id,
        "model_path": model_path,
        "interface": interface,
        "data_path": config.data_path,
        "target_column": config.target_column,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "best_model": results[0].model_name,
        "rankings": [
            {"model": result.model_name, "metrics": result.metrics}
            for result in results
        ],
    }
    with open(config.output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

def update_history_log(entry: dict, history_path: str = "outputs/history_log.json"):
    ensure_parent_dir(history_path)

    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except:
            history = []
    else:
        history = []

    history.append(entry)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def print_training_summary(
    task_type: str,
    primary_metric: str,
    results: List[ModelResult],
    model_id: str,
    model_path: str,
    output_path: str,
) -> None:
    print("\n=== MLPilot Training Results ===")
    print(f"Detected task type: {task_type}")
    print(f"Primary metric: {primary_metric}")
    print(f"Best model: {results[0].model_name}")
    print(f"Model ID: {model_id}")
    print(f"Saved model path: {model_path}")
    print(f"Report path: {output_path}\n")

    for idx, result in enumerate(results, start=1):
        print(f"{idx}. {result.model_name}")
        for metric_name, metric_value in result.metrics.items():
            print(f"   - {metric_name}: {metric_value:.4f}")
        print()


def print_evaluation_summary(model_bundle: Dict[str, Any], metrics: Dict[str, float]) -> None:
    print("\n=== MLPilot Evaluation Results ===")
    print(f"Model ID: {model_bundle['model_id']}")
    print(f"Model name: {model_bundle['model_name']}")
    print(f"Task type: {model_bundle['task_type']}\n")
    for metric_name, metric_value in metrics.items():
        print(f"- {metric_name}: {metric_value:.4f}")
