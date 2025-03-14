from .visualize_correlation import visualize_correlation_matrix


def analyze_correlation_matrix(df, columns=None, threshold=None):
    """
    Analyze the correlation matrix of a DataFrame and optionally filter correlations by a threshold.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        columns (list, optional): List of columns to include in the correlation analysis. If None, all numeric columns are used.
        threshold (float, optional): Correlation threshold to filter significant correlations. If None, no filtering is applied.

    Returns:
        dict: A dictionary containing the full correlation matrix and the filtered correlation matrix.
    """

    # Use only the specified columns or default to all numeric columns
    if columns is not None:
        # Check if the provided columns are numeric
        non_numeric_columns = [
            col for col in columns if df[col].dtype not in ["int64", "float64"]
        ]

        if non_numeric_columns:
            print(
                f"Warning: The following non-numeric columns were excluded from skewness analysis: {', '.join(non_numeric_columns)}"
            )
            # Remove non-numeric columns from the columns list
            columns = [col for col in columns if col not in non_numeric_columns]

    else:
        columns = df.select_dtypes(include="number").columns.tolist()

    if not columns:
        raise ValueError("No numeric columns available for correlation analysis.")

    # Compute the correlation matrix for the selected columns
    filtered_df = df[columns]
    correlation_matrix = filtered_df.corr()

    # Filter the correlation matrix based on the threshold
    if threshold is not None:
        filtered_matrix = correlation_matrix[
            abs(correlation_matrix) >= threshold
        ].fillna(0)
    else:
        filtered_matrix = (
            correlation_matrix  # No filtering applied if threshold is None
        )

    visualize_correlation_matrix(df, correlation_matrix=correlation_matrix)

    # Return both the full and filtered correlation matrices
    return {
        "correlation_matrix": correlation_matrix,
        "filtered_correlation_matrix": filtered_matrix,
    }
