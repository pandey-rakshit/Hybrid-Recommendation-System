import pandas as pd
from utils.helpers import get_file_extension_from_path


def create_dataframe(data_url: str, **kwargs) -> pd.DataFrame:
    """
    Reads data from a file path or URL and returns a DataFrame.

    Args:
        data_url (str): Path to the data file (local or remote).
        **kwargs: Additional arguments for Pandas read functions (e.g., encoding, separator).

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        ValueError: If the file format is unsupported.
        RuntimeError: If data loading fails.
    """
    # Mapping extensions to Pandas read functions
    read_functions = {
        "csv": pd.read_csv,
        "xlsx": pd.read_excel,
        "json": pd.read_json,
        "parquet": pd.read_parquet,
    }

    file_ext = get_file_extension_from_path(data_url)

    if file_ext not in read_functions:
        raise ValueError(f"Unsupported file format: {file_ext}")

    try:
        df = read_functions[file_ext](data_url, **kwargs)
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading data from {data_url}: {e}")
