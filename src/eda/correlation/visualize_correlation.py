from utils.visualization import visualize_chart, plot_functions


def visualize_correlation_matrix(df, columns=None, correlation_matrix=None):
    """
    Visualize the correlation matrix of numeric columns using a heatmap.

    Args:
        df (pd.DataFrame): The DataFrame to analyze and visualize.
        columns (list, optional): List of columns to calculate correlation matrix.
                                  If None, the entire DataFrame will be used.
        correlation_matrix (pd.DataFrame, optional): Precomputed correlation matrix to visualize.
                                                     If None, the correlation matrix will be computed from 'columns'.

    Returns:
        fig: The generated figure object.
    """
    # Compute the correlation matrix based on the inputs (columns or precomputed correlation_matrix)
    if correlation_matrix is None:
        # If correlation_matrix is not provided, compute it from the given columns or all numeric columns
        numeric_columns = (
            columns if columns else df.select_dtypes(include=["number"]).columns
        )
        correlation_matrix = df[numeric_columns].corr(numeric_only=True)

    # Create chart_objs with the heatmap visualization for the correlation matrix
    chart_objs = [
        {
            "plot_function": plot_functions["heatmap"],
            "title": "Relationship between Variables: Correlation Matrix",
            "xlabel": "Features",
            "ylabel": "Features",
            "x": correlation_matrix,  # Using the correlation matrix for visualization
            "kwargs": {
                "annot": True,  # Show correlation coefficients in the heatmap
                "cmap": "viridis",  # Color map for the heatmap
                "fmt": ".2f",  # Format for the correlation values
                "linewidths": 0.7,  # Line thickness between cells
                "vmin": -1,
                "vmax": 1,
                # 'cbar_kws': {'shrink': 0.75}  # Color bar size adjustment
            },
        }
    ]

    # Visualize the correlation matrix using the visualize_chart function
    return visualize_chart(chart_objs, nrows=1, ncols=1)
