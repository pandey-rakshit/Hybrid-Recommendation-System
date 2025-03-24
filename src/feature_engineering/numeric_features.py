# src/feature_engineering/numeric_features.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler

def combine_text_and_numeric(
    df: pd.DataFrame,
    text_matrix: csr_matrix,
    numeric_cols: list,
    scaler: StandardScaler = None
):
    """
    Horizontally stack a sparse text matrix with scaled numeric features.
    Returns the combined sparse matrix plus the fitted (or used) scaler.
    """
    if not numeric_cols:
        return text_matrix, scaler

    # If no scaler provided, we fit a new one
    if scaler is None:
        scaler = StandardScaler()
        numeric_data = df[numeric_cols].fillna(0).values
        numeric_data_scaled = scaler.fit_transform(numeric_data)
    else:
        numeric_data = df[numeric_cols].fillna(0).values
        numeric_data_scaled = scaler.transform(numeric_data)

    numeric_matrix = csr_matrix(numeric_data_scaled)
    combined = hstack([text_matrix, numeric_matrix])
    return combined, scaler
