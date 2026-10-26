"""Grid geometry, topology, connectivity, NNC, and quality specialist."""

from . import DeterministicSpecialist


class GridQualitySpecialist(DeterministicSpecialist):
    name = "grid_quality"
    tool_name = "analyze_grid_quality"


__all__ = ["GridQualitySpecialist"]
