
from dataclasses import dataclass
from typing import Dict


@dataclass
class TrainConfig:
    data_path: str
    target_column: str
    test_size: float = 0.2
    random_seed: int = 42
    metric: str | None = None
    save_path: str | None = None
    output_path: str = "outputs/output.json"
    max_models: int | None = None
    max_train_seconds: int = 180


@dataclass
class PredictConfig:
    model_path: str
    input_path: str
    output_path: str = "outputs/predictions.csv"


@dataclass
class EvaluateConfig:
    model_path: str
    data_path: str
    target_column: str


@dataclass
class ModelResult:
    model_name: str
    metrics: Dict[str, float]