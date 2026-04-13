import pandas as pd
import os
from typing import Tuple

def load_dataset(path: str, target_column: str):
    try:
        df = pd.read_csv(path)
    except Exception:
        raise ValueError(f"Failed to read dataset at '{path}'. File may be corrupted.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    if df.isnull().sum().sum() > 0:
        print("⚠️ Warning: Dataset contains missing values. Applying automatic handling.")

        for col in df.columns:
            if df[col].isnull().any():
                numeric_col = pd.to_numeric(df[col], errors="coerce")

                if numeric_col.notna().sum() == df[col].notna().sum():
                    df[col] = numeric_col.fillna(numeric_col.median())
                else:
                    df[col] = df[col].fillna("missing")

    x = df.drop(columns=[target_column])
    y = df[target_column]

    if len(df) < 10:
        raise ValueError("Dataset too small for training.")

    if y.nunique() == 1:
        raise ValueError("Target column has only one class. Cannot train model.")

    return x, y

def load_inference_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input dataset is empty")

    return df
