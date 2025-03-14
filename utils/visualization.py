from matplotlib import pyplot as plt
import seaborn as sns

plot_functions = {
    "scatter": sns.scatterplot,
    "line": sns.lineplot,
    "bar": sns.barplot,
    "box": sns.boxplot,
    "hist": sns.histplot,
    "pie": plt.pie,
    "count": sns.countplot,
    "heatmap": sns.heatmap,
}


def visualize_chart(chart_objs, nrows=1, ncols=1, **kwargs):
    """
    Create a custom visualization chart with optional subplots.

    Parameters:
    - chart_objs (list): List of dictionaries, where each dictionary contains
                          chart-specific information like 'plot_function', 'titles', etc.
    - nrows (int): Number of rows for the subplot grid. Default is 1.
    - ncols (int): Number of columns for the subplot grid. Default is 1.
    - **kwargs: Additional common keyword arguments passed to the plotting function.

    Returns:
    - fig (matplotlib.figure.Figure): The created figure object.
    """

    width = 18 if ncols == 1 else ncols * 5.43
    height = 6 if nrows == 1 else nrows * 4

    plt.figure(figsize=(width, height))

    # print(axes)

    # Loop through chart_objs to plot the respective data
    for i, chart in enumerate(chart_objs, 1):

        plot_function = chart["plot_function"]
        title = chart["title"]
        xlabel = chart.get("xlabel", None)
        ylabel = chart.get("ylabel", None)
        x = chart["x"]
        y = chart.get("y", None)
        chart_kwargs = chart.get("kwargs", {})

        plt.subplot(nrows, ncols, i)

        # Construct the plotting function arguments
        # Adjust the plot_args dynamically based on the plot function
        plot_args = (
            {"data": x} if plot_function == plot_functions["heatmap"] else {"x": x}
        )

        if y is not None:
            plot_args["y"] = y

        # Add any additional keyword arguments for this specific chart
        plot_args.update(chart_kwargs)

        # Call the plot function
        ax = plot_function(**plot_args)

        plt.title(title, fontsize=16, pad=20)

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        if plot_function in [plot_functions["bar"], plot_functions["count"]]:
            for i in ax.containers:
                ax.bar_label(i, fmt="%.2f")

    # Adjust layout for better readability
    plt.tight_layout()

    fig = plt.gcf()

    return fig
