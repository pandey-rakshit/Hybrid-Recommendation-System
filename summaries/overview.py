from utils.summary_constants import OverviewConstants
from utils.constants import DATASET_KEYS, MISSING_VALUES, SUMMARIES


def overview_step(result):
    """
    Generates an overview summary from the dataset analysis results.

    Args:
        result (dict): The analysis results dictionary.

    Returns:
        str: Overview summary.
    """
    summary = []

    # General dataset information
    summary.append(
        OverviewConstants.OVERVIEW.format(
            ROWS=result[DATASET_KEYS["ROWS"]], COLUMNS=result[DATASET_KEYS["COLUMNS"]]
        )
    )

    # Missing values
    missing_values = result[DATASET_KEYS["MISSING_VALUES"]]
    summary.append(
        OverviewConstants.MISSING_VALUES.format(
            TOTAL=missing_values[MISSING_VALUES["TOTAL"]],
            COLUMNS_LENGTH=len(missing_values[MISSING_VALUES["DETAILS"]]),
        )
    )
    summary.append(
        OverviewConstants.PERCENTAGE.format(
            MISSING_VALUES_PERCENTAGE=missing_values[MISSING_VALUES["PERCENTAGE"]]
        )
    )
    if missing_values[MISSING_VALUES["DETAILS"]]:
        summary.append(OverviewConstants.MISSING_VALUES_DETAILS)
        for col, count in missing_values[MISSING_VALUES["DETAILS"]].items():
            summary.append(f"  - {col}: {count} missing values")
    summary.append("")  # Add a blank line for spacing

    # Duplicate rows
    duplicate_rows = result[DATASET_KEYS["DUPLICATE_ROWS"]]
    if duplicate_rows > 0:
        summary.append(
            OverviewConstants.DUPLICATE_ROWS.format(
                DUPLICATE_ROWS=DATASET_KEYS["DUPLICATE_ROWS"]
            )
        )
    else:
        summary.append(OverviewConstants.NO_DUPLICATE_ROWS)
    summary.append("")  # Add a blank line for spacing

    # Data types
    summary.append(OverviewConstants.DATA_TYPES)
    data_types = result[DATASET_KEYS["DATA_TYPES"]]

    summary.append(data_types.to_string(index=False))

    # for _, row in data_types.iterrows():
    #     summary.append(f"  - {row['Column']}: {row['DataType']}")

    summary.append("")  # Add a blank line for spacing

    # Summary Statistics
    summary.append(OverviewConstants.SUMMARY_STATISTICS)
    statistics = result[SUMMARIES["STATISTICS"]]

    summary.append(
        statistics.to_string()
    )  # Use pandas' `to_string` for a clean table-like output

    return "\n".join(summary)
