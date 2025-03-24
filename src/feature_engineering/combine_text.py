# src/feature_engineering/combine_text.py
import pandas as pd
from typing import List, Optional
from src.data_preprocessing.parse_json import parse_json_column
from src.data_preprocessing.cleaning import clean_text

def create_combined_text(
    df: pd.DataFrame,
    json_cols: Optional[List[str]] = None,
    text_cols: Optional[List[str]] = None,
    combined_col: str = "combined_text",
    json_key: str = "name"
) -> None:
    """
    Merges data from JSON-like columns (extracting 'json_key') and raw text columns
    into a single combined column. Modifies df in place.

    :param df: DataFrame
    :param json_cols: Columns containing JSON or JSON-like data
    :param text_cols: Columns already in plain text
    :param combined_col: Name of the new combined text column
    :param json_key: Key to extract from JSON structures
    """
    if json_cols is None:
        json_cols = []
    if text_cols is None:
        text_cols = []

    # Parse JSON columns and clean them
    for col in json_cols:
        if col in df.columns:
            parsed_series = parse_json_column(df, col, key=json_key)
            col_list = f"{col}_list"
            col_str = f"{col}_str"
            df[col_list] = parsed_series
            df[col_str] = df[col_list].apply(lambda items: " ".join(clean_text(item) for item in items))

    # Clean raw text columns
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Build final combined text
    final_cols = []
    for col in text_cols:
        if col in df.columns:
            final_cols.append(col)
    for col in json_cols:
        col_str = f"{col}_str"
        if col_str in df.columns:
            final_cols.append(col_str)

    df[combined_col] = df.apply(
        lambda row: " ".join(str(row[c]) for c in final_cols),
        axis=1
    )
