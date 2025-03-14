import pandas as pd


def remove_outliers(df, col, lower, upper):
    """Remove rows with outliers for the given column."""
    return df[(df[col] >= lower) & (df[col] <= upper)]


def flag_outliers(df, col, lower, upper):
    """Flag outliers for the given column by adding a new column."""
    df[f"{col}_outlier"] = (df[col] < lower) | (df[col] > upper)
    return df


def transform_outliers(df, col, lower, upper):
    """Clip outliers for the given column to the bounds."""
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def handle_outliers_in_data(df, columns=None, method="remove", threshold=1.5):
    """
    Detect and handle outliers in the DataFrame using the IQR method without using if-else.

    Args:
        df (pd.DataFrame): The DataFrame to check for outliers.
        columns (list, optional): List of columns to check for outliers. If None, all numeric columns are used.
        method (str): Method to handle outliers - 'remove', 'flag', or 'transform'.
        threshold (float): The IQR multiplier for detecting outliers.

    Returns:
        pd.DataFrame: DataFrame with outliers handled (removed, flagged, or transformed).
        pd.DataFrame: DataFrame containing only the outliers.
    """
    # Select numeric columns if no specific columns are provided
    columns = columns or df.select_dtypes(include=["number"]).columns.tolist()

    # DataFrames for results
    df_result = df.copy()
    outliers_list = []

    # Map methods to functions
    method_actions = {
        "remove": remove_outliers,
        "flag": flag_outliers,
        "transform": transform_outliers,
    }

    # Raise error for invalid method
    if method not in method_actions:
        raise ValueError(
            f"Invalid method '{method}'. Choose from 'remove', 'flag', or 'transform'."
        )

    # Loop through each column and handle outliers
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in the DataFrame. Skipping...")
            continue

        # Calculate IQR bounds
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Identify outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        if not outliers.empty:
            outliers_list.append(outliers)

        print(f"Outliers detected for '{col}': {len(outliers)}")

        # Call the appropriate method
        df_result = method_actions[method](df_result, col, lower_bound, upper_bound)

    # Concatenate all outliers into a single DataFrame
    # df_outliers = pd.concat(outliers_list).drop_duplicates()
    df_outliers = (
        pd.concat(outliers_list, ignore_index=True).drop_duplicates()
        if outliers_list
        else pd.DataFrame()
    )

    return df_result, df_outliers
