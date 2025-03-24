import ast
import pandas as pd

def parse_json_column(df, col_name, key='name'):
    """
    A generic function that parses a DataFrame column containing JSON-like or 
    dictionary/list structures and extracts a specified key's value.

    Handles the following cases:
    1. Column does not exist in the DataFrame -> raises KeyError
    2. Raw data is a string (attempts ast.literal_eval)
    3. Raw data is already a dict
    4. Raw data is a list of dicts
    5. Any parsing error or unexpected format -> returns an empty list

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the column to parse.
    col_name : str
        The name of the column holding JSON-like data.
    key : str, optional (default='name')
        The dictionary key whose values will be extracted.

    Returns
    -------
    pd.Series
        A Series where each element is a list of values extracted from 'key'.
    """

    # Check if the column exists
    if col_name not in df.columns:
        raise KeyError(f"Column '{col_name}' not found in the DataFrame.")

    results = []
    for raw_value in df[col_name]:
        # Start with an empty container in case of failure or nulls
        extracted_values = []

        if pd.isna(raw_value):
            # NaNs or None become empty lists
            pass

        elif isinstance(raw_value, str):
            # Try to parse string as JSON-like data
            try:
                parsed_data = ast.literal_eval(raw_value)
            except (ValueError, SyntaxError):
                parsed_data = []  # If parsing fails, fallback

            if isinstance(parsed_data, dict):
                parsed_data = [parsed_data]
            elif not isinstance(parsed_data, list):
                parsed_data = []

            extracted_values = [
                item.get(key, '') for item in parsed_data if isinstance(item, dict)
            ]

        elif isinstance(raw_value, dict):
            # Convert single dict to list for uniform processing
            extracted_values = [raw_value.get(key, '')]

        elif isinstance(raw_value, list):
            # Assume list of dicts
            extracted_values = [
                item.get(key, '') for item in raw_value if isinstance(item, dict)
            ]

        # Append final list of extracted values
        results.append(extracted_values)

    return pd.Series(results, index=df.index)
