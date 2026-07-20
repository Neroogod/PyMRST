"""Backward-compatible aggregation path for AD-core model classes."""

from pymrst.modules.ad_core.models.facilities.facility_model import FacilityModel
from pymrst.modules.ad_core.models.physical_model import PhysicalModel
from pymrst.modules.ad_core.models.reservoir_model import ReservoirModel

__all__ = ["FacilityModel", "PhysicalModel", "ReservoirModel"]
