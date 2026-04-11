import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def detect_task_type(y: pd.Series) -> str:
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"

    unique_count = y.nunique(dropna=True)
    total_count = len(y)

    if pd.api.types.is_integer_dtype(y) and unique_count <= max(20, int(0.05 * total_count)):
        return "classification"
    if unique_count <= 10:
        return "classification"

    return "regression"


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = x.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
