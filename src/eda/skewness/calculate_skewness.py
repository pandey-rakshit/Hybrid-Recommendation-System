def calculate_skewness(df, columns=None):
    """
    Calculate skewness for specific columns or all numeric columns in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        columns (list): A list of column names for which skewness needs to be calculated.
                          If None, skewness will be calculated for all numeric columns.

    Returns:
        dict: A dictionary with column names as keys and skewness values as values.
    """
    if columns is None:
        # If no column names are provided, calculate skewness for all numeric columns
        numeric_columns = df.select_dtypes(include=["number"]).columns
    else:
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

    # Calculate skewness for the selected columns
    skewness_dict = {col: df[col].skew() for col in numeric_columns}
    return skewness_dict
