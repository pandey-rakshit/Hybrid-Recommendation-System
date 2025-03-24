from .visualize_skewness import visualize_skewness_with_chart


def analyze_skewness(df, columns=None):
    """
    Analyze the skewness of numeric columns in the dataset and generate a summary of skewness categories.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        columns (list, optional): List of columns to include in the skewness analysis.
                                   If None, all numeric columns will be considered.

    Returns:
        dict: A dictionary containing the skewness analysis result categorized by 'high', 'moderate', and 'low'.
        plt.Figure: A Matplotlib figure object displaying histograms of the skewness.
    """
    # Determine the columns to analyze (use all numeric columns if no specific columns are provided)
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

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

    # If there are no numeric columns to analyze, raise an error
    if not columns:
        raise ValueError(
            "No numeric columns available in the dataset for skewness analysis."
        )

    # Calculate skewness for each numeric column
    skewness_values = df[columns].apply(lambda x: x.skew()).to_dict()

    # Categorize columns based on skewness values
    high_skew = [col for col, skew in skewness_values.items() if abs(skew) > 1]
    moderate_skew = [
        col for col, skew in skewness_values.items() if 0.5 < abs(skew) <= 1
    ]
    low_skew = [col for col, skew in skewness_values.items() if abs(skew) <= 0.5]

    # Prepare the result object
    skewness_result = {
        "high_skew": high_skew,
        "moderate_skew": moderate_skew,
        "low_skew": low_skew,
        "skewness_values": skewness_values,  # For detailed skewness values of each column
    }

    # print(skewness_result)
    # Generate and visualize skewness distribution for the specified columns
    visualize_skewness_with_chart(df, columns)

    return skewness_result
