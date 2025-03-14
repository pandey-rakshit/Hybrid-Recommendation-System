class ObservationsConstants:
    OBSERVATIONS_NOT_FOUND = "No numerical observations were found in the dataset."
    OBSERVATION_HEADER = "Observations based on the dataset:\n"


class OverviewConstants:
    OVERVIEW = "The dataset contains {ROWS} rows and {COLUMNS} columns.\n"
    MISSING_VALUES = "There are {TOTAL} missing values across {COLUMNS_LENGTH} columns."
    PERCENTAGE = (
        "Missing values account for {MISSING_VALUES_PERCENTAGE:.2f}% of the dataset."
    )
    MISSING_VALUES_DETAILS = "Columns with missing values and their counts:"
    DUPLICATE_ROWS = "There are {DUPLICATE_ROWS} duplicate rows in the dataset."
    NO_DUPLICATE_ROWS = "There are no duplicate rows in the dataset."
    DATA_TYPES = "Data Types:\n"
    SUMMARY_STATISTICS = "Summary Statistics:"
