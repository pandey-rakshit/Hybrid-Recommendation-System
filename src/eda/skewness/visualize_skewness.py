from utils.visualization import visualize_chart, plot_functions


def visualize_skewness_with_chart(df, numeric_columns):
    """
    Visualize skewness of all numeric columns using histograms with KDE.

    Args:
        df (pd.DataFrame): The DataFrame to analyze and visualize.

    Returns:
        fig: The generated figure object.
    """
    # Identify numeric columns
    # numeric_columns = df.select_dtypes(include=['number']).columns

    # Calculate skewness for each numeric column
    skewness = df[numeric_columns].skew().round(2)

    # print(skewness)

    # Create chart_objs with customized titles and other options for each column
    chart_objs = []
    for col in numeric_columns:
        # print(col, skewness[col])
        chart_objs.append(
            {
                "plot_function": plot_functions["hist"],
                "title": f"Skewness of {col}: {skewness[col]}",  # Title for each individual column
                "xlabel": col,
                "ylabel": "Frequency",
                "x": df[col],
                "kwargs": {"kde": True, "color": "purple", "element": "poly"},
            }
        )

    # print(chart_objs)

    # Visualize skewness using the visualize_chart function
    return visualize_chart(chart_objs, nrows=(len(numeric_columns) // 3 + 1), ncols=3)


# # Example: Visualize the skewness of all numeric columns in the DataFrame
# fig = visualize_skewness_with_chart(df)
# plt.show()
