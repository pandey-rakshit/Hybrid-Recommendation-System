from .summary_factory import SummaryFactory
from .overview import overview_step
from .observations import observations_step
from .skewness import skewness_summary

from utils.constants import SUMMARIES

# Initialize factory
summary_factory = SummaryFactory()

# Register summary steps
summary_factory.register_step(SUMMARIES["OVERVIEW"], overview_step)
summary_factory.register_step(SUMMARIES["OBSERVATIONS"], observations_step)
summary_factory.register_step(SUMMARIES["SKEWNESS"], skewness_summary)

# Expose factory for use in other modules
__all__ = ["summary_factory"]
