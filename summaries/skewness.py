import pandas as pd


def skewness_summary(skewness_object):
    """
    Generates a textual and tabular summary of skewness analysis.

    Args:
        skewness_object (dict): A dictionary containing skewness analysis results.

    Returns:
        str: Formatted skewness summary.
    """
    categories = {
        "high_skew": "Highly skewed columns (suggesting log transformation)",
        "moderate_skew": "Moderately skewed columns (suggesting square root transformation)",
        "low_skew": "Columns with low skewness (no transformation needed)",
    }

    summary = [
        f"- {desc}: {', '.join(skewness_object[key])}"
        for key, desc in categories.items()
        if skewness_object.get(key)
    ]

    if skewness_object.get("skewness_values"):
        skewness_df = (
            pd.DataFrame.from_dict(
                skewness_object["skewness_values"], orient="index", columns=["Skewness"]
            )
            .rename_axis("Column")
            .reset_index()
        )
        summary.append(
            "\nDetailed Skewness Values:\n" + skewness_df.to_string(index=False)
        )

    return "\n".join(summary) if summary else "No skewness analysis results available."
