import argparse
import os
import shutil
import time
from datetime import datetime

from python_cli.config import TrainConfig, EvaluateConfig, PredictConfig
from python_cli.utils import ensure_parent_dir
from python_cli.data_loader import load_dataset, load_inference_data
from python_cli.preprocess import detect_task_type
from python_cli.models import train_and_select_best
from python_cli.metrics import evaluate_classification, evaluate_regression
from python_cli.persistance import load_model_bundle, save_model_bundle, resolve_model_path, generate_model_id
from python_cli.logging_utils import save_training_report, update_history_log, print_training_summary, print_evaluation_summary
from python_cli.visualization import (
    save_classification_comparison_plot,
    save_regression_comparison_plot,
    save_confusion_matrix_plot,
    save_regression_scatter_plot,
    save_feature_importance_plot,
    save_roc_curve_plot,
)


def _validate_train_args(config: TrainConfig) -> None:
    if not (0 <= config.random_seed <= 4294967295):
        raise ValueError("Seed must be between 0 and 4294967295")

def handle_train(args: argparse.Namespace, interface: str = "python") -> None:
    start_time = time.time()
    started_at = datetime.now().isoformat()
    config = TrainConfig(
        data_path=args.data,
        target_column=args.target,
        test_size=args.test_size,
        random_seed=args.seed,
        metric=args.metric,
        save_path=args.save,
        output_path=args.output,
    )

    _validate_train_args(config)

    x, y = load_dataset(config.data_path, config.target_column)
    task_type = detect_task_type(y)
    results, primary_metric, best_pipeline, x_test, y_test, best_predictions = train_and_select_best(
        x, y, task_type, config
    )
    finished_at = datetime.now().isoformat()
    duration_seconds = round(time.time() - start_time, 4)
    model_id = generate_model_id()
    model_path = resolve_model_path(config.data_path, config.save_path, model_id)


    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()

    save_feature_importance_plot(
        best_pipeline,
        feature_names,
        os.path.join("outputs/plots", f"{model_id}_feature_importance.png"),
    )
    save_roc_curve_plot(
        best_pipeline,
        x_test,
        y_test,
        os.path.join("outputs/plots", f"{model_id}_roc_curve.png"),
    )

    model_bundle = {
        "model_id": model_id,
        "model_name": results[0].model_name,
        "task_type": task_type,
        "pipeline": best_pipeline,
        "target_column": config.target_column,
        "primary_metric": primary_metric,
        "metrics": results[0].metrics,
        "feature_columns": list(x.columns),
        "interface": interface,
        "training_duration_seconds": duration_seconds,
    }

    save_model_bundle(model_bundle, model_path)
    
    plot_dir = "outputs/plots"
    if task_type == "classification":
        save_classification_comparison_plot(
            results,
            os.path.join(plot_dir, f"{model_id}_comparison.png"),
        )
        save_confusion_matrix_plot(
            y_test,
            best_predictions,
            os.path.join(plot_dir, f"{model_id}_confusion_matrix.png"),
        )
    else:
        save_regression_comparison_plot(
            results,
            os.path.join(plot_dir, f"{model_id}_comparison.png"),
        )
        save_regression_scatter_plot(
            y_test,
            best_predictions,
            os.path.join(plot_dir, f"{model_id}_predicted_vs_actual.png"),
        )

    try:
        feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        save_feature_importance_plot(
            best_pipeline,
            feature_names,
            os.path.join(plot_dir, f"{model_id}_feature_importance.png"),
        )
    except Exception:
        pass
    
    save_training_report(
        config,
        task_type,
        primary_metric,
        results,
        model_id,
        model_path,
        interface,
    )

    history_entry = {
        "model_id": model_id,
        "model_path": model_path,
        "interface": interface,
        "data_path": config.data_path,
        "target_column": config.target_column,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "best_model": results[0].model_name,
        "metrics": results[0].metrics,
        "interface": interface,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "num_rows": int(len(x)),
        "num_features": int(len(x.columns)),
    }

    update_history_log(history_entry)
    print_training_summary(task_type, primary_metric, results, model_id, model_path, config.output_path)

def handle_predict(args: argparse.Namespace) -> None:
    config = PredictConfig(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
    )

    model_bundle = load_model_bundle(config.model_path)
    raw_df = load_inference_data(config.input_path)
    prediction_df = raw_df.copy()

    target_column = model_bundle["target_column"]
    if target_column in raw_df.columns:
        raw_df = raw_df.drop(columns=[target_column])

    expected_columns = model_bundle.get("feature_columns")
    if expected_columns is not None:
        missing_columns = [col for col in expected_columns if col not in raw_df.columns]
        if missing_columns:
            raise ValueError(
                f"Input data is missing required feature columns: {missing_columns}"
            )
        raw_df = raw_df[expected_columns]

    pipeline = model_bundle["pipeline"]
    predictions = pipeline.predict(raw_df)

    prediction_df["prediction"] = predictions
    prediction_df["model_id"] = model_bundle["model_id"]

    if (
        model_bundle["task_type"] == "classification"
        and hasattr(pipeline, "predict_proba")
    ):
        probabilities = pipeline.predict_proba(raw_df)
        prediction_df["confidence"] = probabilities.max(axis=1)

    ensure_parent_dir(config.output_path)
    prediction_df.to_csv(config.output_path, index=False)

    print("\n=== MLPilot Prediction Results ===")
    print("\n--- Predicted Results are written to output/predictions.csv ---")
    print(f"Model ID: {model_bundle['model_id']}")
    print(f"Model name: {model_bundle['model_name']}")
    print(f"Predictions saved to: {config.output_path}")
    print(f"Rows predicted: {len(predictions)}")

def handle_evaluate(args: argparse.Namespace) -> None:
    config = EvaluateConfig(
        model_path=args.model,
        data_path=args.data,
        target_column=args.target,
    )

    model_bundle = load_model_bundle(config.model_path)
    x, y = load_dataset(config.data_path, config.target_column)

    expected_target = model_bundle["target_column"]
    if config.target_column != expected_target:
        raise ValueError(
            f"Provided target column '{config.target_column}' does not match saved model target '{expected_target}'"
        )

    expected_columns = model_bundle.get("feature_columns")
    if expected_columns is not None:
        missing_columns = [col for col in expected_columns if col not in x.columns]
        if missing_columns:
            raise ValueError(
                f"Evaluation data is missing required feature columns: {missing_columns}"
            )
        x = x[expected_columns]

    predictions = model_bundle["pipeline"].predict(x)
    task_type = model_bundle["task_type"]

    if task_type == "classification":
        metrics = evaluate_classification(y, predictions)
    else:
        metrics = evaluate_regression(y, predictions)

    print_evaluation_summary(model_bundle, metrics)

def handle_clean() -> None:
    dirs_to_remove = ["models", "outputs"]

    for d in dirs_to_remove:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Removed directory: {d}")
        else:
            print(f"Directory not found (skipped): {d}")

    print("\nCleanup complete.")
