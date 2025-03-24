from utils.constants import SUMMARIES
from utils.summary_constants import ObservationsConstants


def observations_step(result):
    """
    Generates observations based on numerical analysis.

    Args:
        result (dict): The analysis results dictionary.

    Returns:
        str: Observations summary.
    """
    observations = result.get(SUMMARIES["OBSERVATIONS"], {})

    if not observations:
        return ObservationsConstants.OBSERVATIONS_NOT_FOUND

    summary = [ObservationsConstants.OBSERVATION_HEADER] + [
        f"  - {col}: {observation}" for col, observation in observations.items()
    ]

    return "\n".join(summary)
