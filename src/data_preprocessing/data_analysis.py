from utils.constants import MISSING_VALUES, DATASET_KEYS, SUMMARIES, SKEWNESS


def analyze_dataset(df, exclude_columns=None, include_stats=True):
    """
    Performs exploratory analysis on the dataset.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        exclude_columns (list, optional): List of columns to exclude from numerical analysis.
        include_stats (bool, optional): Whether to compute summary statistics.

    Returns:
        dict: A dictionary containing dataset insights.
    """
    results = {
        DATASET_KEYS["MISSING_VALUES"]: summarize_missing_values(df),
        DATASET_KEYS["DUPLICATE_ROWS"]: df.duplicated().sum(),
        DATASET_KEYS["DATA_TYPES"]: df.dtypes.reset_index(),
    }

    # General dataset info
    results[DATASET_KEYS["ROWS"]], results[DATASET_KEYS["COLUMNS"]] = df.shape
    results[DATASET_KEYS["DATA_TYPES"]].columns = ["Column", "DataType"]

    if include_stats:
        stats = compute_summary_statistics(df, exclude_columns)

        results[SUMMARIES["STATISTICS"]] = stats
        results[SUMMARIES["OBSERVATIONS"]] = analyze_skewness(stats)

    return results


def summarize_missing_values(df):
    """Summarizes missing values in the dataset."""
    missing_count = df.isnull().sum()
    missing_data = missing_count[missing_count > 0]

    return {
        MISSING_VALUES["TOTAL"]: missing_data.sum(),
        MISSING_VALUES["PERCENTAGE"]: (missing_data.sum() / df.size) * 100,
        MISSING_VALUES["DETAILS"]: missing_data.to_dict(),
    }


def compute_summary_statistics(df, exclude_columns):
    """Computes descriptive statistics for numerical columns."""
    numeric_df = df.select_dtypes(include=["number"])
    if exclude_columns:
        numeric_df = numeric_df.drop(columns=exclude_columns, errors="ignore")

    stats = numeric_df.describe().T

    return stats.round(2)


def analyze_skewness(stats):
    """Identifies skewness in numerical features."""
    observations = {}
    for col in stats.index:
        mean, median = stats.loc[col, ["mean", "50%"]]
        if mean < median:
            observations[col] = SKEWNESS["LEFT_SKEWED"]
        elif mean > median:
            observations[col] = SKEWNESS["RIGHT_SKEWED"]
        else:
            observations[col] = SKEWNESS["SYMMETRIC"]

    return observations
