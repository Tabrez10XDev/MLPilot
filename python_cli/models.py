import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from typing import Any, Dict, List, Tuple
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

from python_cli.config import ModelResult, TrainConfig
from python_cli.preprocess import  build_preprocessor
from python_cli.metrics import evaluate_classification, evaluate_regression, get_default_metric, is_higher_better


def get_candidate_models(task_type: str, random_seed: int) -> Dict[str, Any]:
    if task_type == "classification":
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_seed),
            "knn_classifier": KNeighborsClassifier(),
            "decision_tree_classifier": DecisionTreeClassifier(random_state=random_seed),
            "random_forest_classifier": RandomForestClassifier(
                n_estimators=100,
                random_state=random_seed,
            ),
            "svm_classifier": SVC(probability=True, random_state=random_seed),
            "gradient_boosting_classifier": GradientBoostingClassifier(
                random_state=random_seed
            ),
        }

        if XGBOOST_AVAILABLE:
            models["xgboost_classifier"] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_seed,
                eval_metric="logloss",
            )
        else:
            print("⚠️ XGBoost is unavailable on this system. Skipping XGBoost models.")

        return models

    models = {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(),
        "knn_regressor": KNeighborsRegressor(),
        "decision_tree_regressor": DecisionTreeRegressor(random_state=random_seed),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=random_seed,
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(
            random_state=random_seed
        ),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost_regressor"] = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_seed,
        )
    else:
        print("⚠️ XGBoost is unavailable on this system. Skipping XGBoost models.")


    return models

def train_and_select_best(
    x: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    config: TrainConfig,
) -> Tuple[List[ModelResult], str, Pipeline, pd.DataFrame, pd.Series, np.ndarray]:
    if not 0 < config.test_size < 1:
        raise ValueError("Test size must be between 0 and 1")

    preprocessor = build_preprocessor(x)
    models = get_candidate_models(task_type, config.random_seed)

    stratify = y if task_type == "classification" else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=stratify,
    )

    results: List[ModelResult] = []
    trained_pipelines: Dict[str, Pipeline] = {}
    predictions_by_model: Dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", clone(model)),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        if task_type == "classification":
            metrics = evaluate_classification(y_test, predictions)
        else:
            metrics = evaluate_regression(y_test, predictions)

        results.append(ModelResult(model_name=model_name, metrics=metrics))
        trained_pipelines[model_name] = pipeline
        predictions_by_model[model_name] = predictions

    primary_metric = config.metric or get_default_metric(task_type)
    if primary_metric not in results[0].metrics:
        raise ValueError(
            f"Metric '{primary_metric}' is not valid for task type '{task_type}'"
        )

    reverse = is_higher_better(primary_metric)
    results.sort(key=lambda item: item.metrics[primary_metric], reverse=reverse)

    best_model_name = results[0].model_name
    best_pipeline = trained_pipelines[best_model_name]
    best_predictions = predictions_by_model[best_model_name]

    return results, primary_metric, best_pipeline, x_test, y_test, best_predictions