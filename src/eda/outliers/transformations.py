import numpy as np

from .outlier_handling import handle_outliers_in_data


def apply_transformation(df, column, transformation_type):
    """
    Apply the specified transformation to a column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to transform.
        column (str): The name of the column to transform.
        transformation_type (str): The type of transformation to apply.
                                   Supported: 'log', 'sqrt', 'square', 'reciprocal'.

    Returns:
        pd.Series: Transformed column as a pandas Series.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    # Transformation mapping
    transformations = {
        "log": lambda x: np.log1p(x),
        "sqrt": lambda x: np.sqrt(x.clip(lower=0)),
        "square": lambda x: np.square(x),
        "reciprocal": lambda x: 1 / x.replace(0, np.nan),
    }

    # Get the transformation function
    transform_func = transformations.get(transformation_type)
    if not transform_func:
        raise ValueError(f"Unsupported transformation type: '{transformation_type}'.")

    # Apply the transformation
    return transform_func(df[column])


def best_transformation_with_outliers(df, skew_categories, handle_outliers=False, ):
    """
    Apply the best transformation based on skewness for each column already categorized into high, moderate, or low skew.
    Optionally, handle outliers by removing or transforming them.

    Args:
        df (pd.DataFrame): The DataFrame containing numeric columns.
        skew_categories (dict): Dictionary with keys 'high', 'moderate', 'low' mapping to lists of column names.
        handle_outliers (bool): Whether to handle outliers by removal after transformation.
        skew_threshold (float): The threshold for determining the degree of skewness for transformation.
        outlier_threshold (float): The threshold for detecting outliers based on IQR method.

    Returns:
        pd.DataFrame: DataFrame with transformations applied to columns and outliers handled (if applicable).
    """
    transformed_df = df.copy()  # Create a copy to apply transformations

    # Dictionary of transformations for each skew category
    transformations = {"high_skew": "log", "moderate_skew": "sqrt"}

    # 'low': 'boxcox'  # Optional, or we can skip transformation for low skew

    transformed_col = []
    # Apply transformations based on skew category
    for skew_category, transformation_type in transformations.items():
        columns = skew_categories.get(skew_category, [])
        for col in columns:
            print(
                f"Applying {transformation_type} transformation to {col} due to {skew_category} skewness."
            )
            transformed_df[col] = apply_transformation(
                transformed_df, col, transformation_type
            )
            transformed_col.append(col)

    # Handle outliers if required
    if handle_outliers:
        transformed_df = handle_outliers_in_data(transformed_df, transformed_col)

    return transformed_df
